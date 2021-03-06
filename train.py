from __future__ import absolute_import, division, print_function

import os
import math
import numpy as np
from time import time

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.autograd import Variable

from envs import create_atari_env
from model import ES


def do_rollouts(args, models, random_seeds, return_queue, envs, are_negative, virtual_batch, matching_env_seed=False):
    """
    For each model, do a rollout. Supports multiple models per thread but
    don't do it -- it's inefficient (it's mostly a relic of when I would run
    both a perturbation and its antithesis on the same thread).
    """
    all_returns = []
    all_num_frames = []
    for iii,(model,env) in enumerate(zip(models,envs)):
        if args.gpu: model = model.cuda()
        if args.virtual_batch_norm:
            if args.gpu: virtual_batch = virtual_batch.cuda()
            model.do_virtual_batch_norm(virtual_batch)
        if matching_env_seed: env.seed(matching_env_seed)
        state = env.reset()
        state = torch.from_numpy(state)
        this_model_return = 0
        this_model_num_frames = 0
        # Rollout
        for step in range(args.max_episode_length):
            if args.gpu: state = state.cuda()
            prob = model(Variable(state.unsqueeze(0), volatile=True))

            if args.gpu: prob = prob.cpu()
            if args.act_by_argmax:
                action = prob.max(1)[1].data.numpy()[0, 0]
            else:
                action = torch.multinomial(prob).data.numpy()[0, 0]
            state, reward, done, _ = env.step(action)
            this_model_return += reward
            this_model_num_frames += 1
            if done: 
                break
            state = torch.from_numpy(state)
        all_returns.append(this_model_return)
        all_num_frames.append(this_model_num_frames)
    return_queue.put((random_seeds, all_returns, all_num_frames, are_negative))


def perturb_model(args, model, random_seed, env):
    """
    Modifies the given model with a perturbation of its parameters,
    as well as the negative perturbation, and returns both perturbed
    models.
    """
    new_model = ES(env.observation_space,env.action_space,
                    use_a3c_net=args.a3c_net, use_virtual_batch_norm=args.virtual_batch_norm)
    anti_model = ES(env.observation_space,env.action_space,
                    use_a3c_net=args.a3c_net, use_virtual_batch_norm=args.virtual_batch_norm)
    new_model.load_state_dict(model.state_dict())
    anti_model.load_state_dict(model.state_dict())
    np.random.seed(random_seed)
    eps = args.sigma * np.random.normal(0.0, 1.0, size=model.count_parameters())
    new_model.adjust_es_params(add=eps)
    anti_model.adjust_es_params(add=-eps)
    # for (k, v), (anti_k, anti_v) in zip(new_model.get_es_params(),
    #                                     anti_model.get_es_params()):
    #     eps = np.random.normal(0, 1, v.size())
    #     v += torch.from_numpy(args.sigma*eps).float()
    #     anti_v += torch.from_numpy(args.sigma*-eps).float()
    return [new_model, anti_model]


# from: https://github.com/openai/evolution-strategies-starter/blob/7585f01cc64890aefbbf3cccda326e2ca90ba6f8/es_distributed/es.py#L69
def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort(kind='mergesort')] = np.arange(len(x))
    return ranks

def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y

def fitness_shaping(returns):
    """
    A rank transformation on the rewards, which reduces the chances
    of falling into local optima early in training.
    """
    sorted_returns_backwards = sorted(returns)[::-1]
    lamb = len(returns)
    shaped_returns = []
    denom = sum([max(0, math.log2(lamb/2 + 1) -
                     math.log2(sorted_returns_backwards.index(r) + 1))
                 for r in returns])
    for r in returns:
        num = max(0, math.log2(lamb/2 + 1) -
                  math.log2(sorted_returns_backwards.index(r) + 1))
        shaped_returns.append(num/denom + 1/lamb)
    return shaped_returns

def unperturbed_rank(returns, unperturbed_results):
    nth_place = 1
    for r in returns:
        if r > unperturbed_results:
            nth_place += 1
    rank_diag = ('%d out of %d (1 means gradient '
                 'is uninformative)' % (nth_place,
                                         len(returns) + 1))
    return rank_diag, nth_place


class Optimizer:
    def __init__(self,args):
        if args.momentum:
            self.prev_update = 0.0
    
    def gradient_update(self, args, synced_model, virtual_batch, returns, random_seeds, neg_list,
                        num_eps, num_frames, chkpt_dir, unperturbed_results, start_time):
        batch_size = len(returns)
        assert batch_size == args.n
        assert len(random_seeds) == batch_size
        
        # Compute rank transform
        if args.alt_rank:
            shaped_returns = list(compute_centered_ranks(np.asarray(returns)))
        else:
            shaped_returns = fitness_shaping(returns)

        # Consolidate updates
        consolidated_seeds = {}
        for seed,neg,shaped_return in zip(random_seeds,neg_list,shaped_returns):
            consolidated_seeds[seed] = consolidated_seeds.get(seed,0.) + (-1)**neg*shaped_return
        consolidated_seeds = {seed:weight for seed,weight in consolidated_seeds.items() if abs(weight) > 0.0}
        if not consolidated_seeds:
            # TODO should we apply momentum and weight decay even without productive updates?
            return synced_model
        
        # For each model, generate the same random numbers as we did
        # before, and update parameters.
        weighted_eps_sum = 0.
        for seed,weight in consolidated_seeds.items():
            # print(seed,weight,weight==0.0)
            if weight == 0.0: continue
            np.random.seed(seed)
            eps = args.sigma * np.random.normal(0.0, 1.0, size=synced_model.count_parameters())
            weighted_eps_sum += weight * eps
        
        # Perform update
        update = (args.n*args.sigma) * weighted_eps_sum
        if args.momentum:
            update += args.momentum * self.prev_update
            self.prev_update = update
        synced_model.adjust_es_params(multiply=args.weight_decay, add=update)
        args.lr *= args.lr_decay
        
        # Print diagnostic info
        rewards = [reward for reward,length in returns]
        # rank_diag, rank = unperturbed_rank(returns, unperturbed_results)
        if not args.silent:
            print('\n'
                  'Update: %d\n'
                  'Time: %.2f sec\n'
                  'Mean reward: %.3f (%.3f)\n'
                  'Max reward: %.3f\n'
                  'Median reward: %.3f\n'
                  'Min reward: %.3f\n'
                  'Model norm: %.3f:\n'
                  'Gradient norm: %.3f\n'
                  'Cumulative episodes: %d\n'
                  'Cumulative frames: %d'
                  %
                  ( num_eps/batch_size, time()-start_time,
                    np.mean(rewards), np.var(rewards),
                    max(rewards), np.median(rewards), min(rewards),
                    synced_model.get_param_norm(), np.linalg.norm(update),
                    num_eps, num_frames,
                  ))
        
        # Save model state
        if args.virtual_batch_norm:
            if args.gpu:
                virtual_batch = virtual_batch.cpu()
                synced_model = synced_model.cpu()
            synced_model.do_virtual_batch_norm(virtual_batch)
        torch.save(synced_model.state_dict(),
                   os.path.join(chkpt_dir, 'latest.pth'))
        return synced_model


def render_env(args, model, env):
    while True:
        state = env.reset()
        state = torch.from_numpy(state)
        this_model_return = 0
        done = False
        while not done:
            prob = model(
                (Variable(state.unsqueeze(0), volatile=True)))

            action = prob.max(1)[1].data.numpy()
            state, reward, done, _ = env.step(action[0, 0])
            this_model_return += reward
            state = torch.from_numpy(state)
        print('Reward: %f' % this_model_return)


def generate_seeds_and_models(args, synced_model, env):
    """
    Returns a seed and 2 perturbed models
    """
    np.random.seed()
    random_seed = np.random.randint(2**30)
    two_models = perturb_model(args, synced_model, random_seed, env)
    return random_seed, two_models


def gather_for_virtual_batch_norm(env, batch_size=128, skip_steps=10, seed=409):
    """
    Gather a set of frames for virtual batch normalization.
    """
    np.random.seed(seed)
    env.seed(seed)
    env.reset()
    virtual_batch = []
    for _ in range(batch_size):
        for _ in range(skip_steps):
            action = np.random.randint(env.action_space.n)
            state,_,done,_ = env.step(action)
            if done: env.reset()
        virtual_batch += [state]
    return np.stack(virtual_batch)

def flatten(raw_results, index):
    notflat_results = [result[index] for result in raw_results]
    return [item for sublist in notflat_results for item in sublist]

def train_loop(args, synced_model, env, chkpt_dir, virtual_batch=None):
    # init stats
    num_eps = 0
    total_num_frames = 0
    
    # init optimizer
    opt = Optimizer(args)
    
    # init an environment for every process
    init_envs = [create_atari_env(args.env_name, frame_stack_size=args.stack_images, noop_init=args.noop_init, image_dim=args.image_dim) 
                for _ in range(args.n+1)]
    
    # train
    for iteration in range(args.max_gradient_updates):
        # init
        start_time = time()
        processes = []
        return_queue = mp.Queue()
        all_seeds, all_models, all_are_negative, all_envs = [], [], [], []
        
        # set a matching environment seed
        if args.match_env_seeds:
            env_seed = iteration
        else:
            env_seed = False
        
        # generate perturbations on the synced model
        for j in range(int(args.n/2)):
            random_seed, two_models = generate_seeds_and_models(args,synced_model,env)
            all_seeds.append(random_seed)
            all_seeds.append(random_seed)
            all_models += two_models
            all_are_negative += [False,True]
            all_envs.append(init_envs[j*2])
            all_envs.append(init_envs[j*2+1])
        assert len(all_seeds) == len(all_models)
        
        # add all peturbed models to the queue
        while all_models:
            # get model/seed/env from the queue
            perturbed_models = [all_models.pop() for _ in range(args.models_per_thread)]
            seeds = [all_seeds.pop() for _ in range(args.models_per_thread)]
            envs = [all_envs.pop() for _ in range(args.models_per_thread)]
            are_negative = [all_are_negative.pop() for _ in range(args.models_per_thread)]
            
            # start process
            p = mp.Process(target=do_rollouts, 
                    args=(args,perturbed_models,seeds,return_queue,envs,are_negative,virtual_batch,env_seed))
            p.start()
            processes.append(p)
        assert len(all_seeds) == 0
        
        # evaluate the unperturbed model too
        # p = mp.Process(target=do_rollouts,
        #         args=(args,[synced_model],['dummy_seed'],return_queue,envs[-1:],['dummy_neg'],virtual_batch,env_seed))
        # p.start()
        # processes.append(p)
        
        # wait for processes to finish
        for p in processes: p.join()
        raw_results = [return_queue.get() for p in processes]
        seeds, rewards, num_frames, neg_list = [flatten(raw_results, index) for index in [0, 1, 2, 3]]
        returns = [x for x in zip(rewards,num_frames)]
        
        # Separate the unperturbed results from the perturbed results
        # _ = unperturbed_index = seeds.index('dummy_seed')
        # seeds.pop(unperturbed_index)
        # unperturbed_results = returns.pop(unperturbed_index)
        # _ = num_frames.pop(unperturbed_index)
        # _ = neg_list.pop(unperturbed_index)
        unperturbed_results = None
        
        # update stats
        total_num_frames += sum(num_frames)
        num_eps += len(returns)
        
        # perform model update
        synced_model = opt.gradient_update(args, synced_model, virtual_batch, returns, seeds,
                                       neg_list, num_eps, total_num_frames,
                                       chkpt_dir, unperturbed_results, start_time)
        
        # update variable episode length
        if args.variable_ep_len:
            args.max_episode_length = int(2*sum(num_frames)/len(num_frames))
