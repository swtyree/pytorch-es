from __future__ import absolute_import, division, print_function

import numpy as np

import os
import argparse

import torch

from envs import create_atari_env
from model import ES
from train import train_loop, render_env, gather_for_virtual_batch_norm, torchify

parser = argparse.ArgumentParser(description='ES')
parser.add_argument('--env-name', default='PongDeterministic-v3',
                    metavar='ENV', help='environment')
parser.add_argument('--noop-init', type=int, default=30, metavar='N',
                    help='maximum number of random no-ops at start of episode')
parser.add_argument('--lr', type=float, default=0.5, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='B',
                    help='momentum rate')
parser.add_argument('--lr-decay', type=float, default=1.0, metavar='LRD',
                    help='learning rate decay')
parser.add_argument('--sigma', type=float, default=0.01, metavar='SD',
                    help='noise standard deviation')
parser.add_argument('--weight-decay', type=float, default=0.9, metavar='WD',
                    help='amount of weight decay')
parser.add_argument('--n', type=int, default=12, metavar='N',
                    help='batch size, must be even')
parser.add_argument('--max-episode-length', type=int, default=100000,
                    metavar='MEL', help='maximum length of an episode')
parser.add_argument('--max-gradient-updates', type=int, default=100000,
                    metavar='MGU', help='maximum number of updates')
parser.add_argument('--restore', default='', metavar='RES',
                    help='checkpoint from which to restore')
parser.add_argument('--a3c-net', action='store_true',
                    help='use A3C network')
parser.add_argument('--alt-rank-trans', action='store_true',
                    help='use alternative rank transformation')
parser.add_argument('--stack-images', type=int, default=1, metavar='S',
                    help='input a stack of recent frames')
parser.add_argument('--virtual-batch-norm', type=int, default=128, metavar='V',
                    help='Use virtual batch normalization with V frames')
parser.add_argument('--variable-ep-len', action='store_true',
                    help="change max episode length during training")
parser.add_argument('--silent', action='store_true',
                    help='silence print statements during training')
parser.add_argument('--test', action='store_true',
                    help='just render the env, no training')
parser.add_argument('--gpu', action='store_true',
                    help='use GPU')
parser.add_argument('--models-per-thread', type=int, default=1, metavar='M',
                    help='models evaluated by each thread')


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.n % 2 == 0
    
    env = create_atari_env(args.env_name, frame_stack_size=args.stack_images, noop_init=args.noop_init)
    chkpt_dir = 'checkpoints/%s/' % args.env_name
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
    synced_model = ES(env.observation_space.shape[0], env.action_space,
        use_a3c_net=args.a3c_net, use_virtual_batch_norm=args.virtual_batch_norm)
    for param in synced_model.parameters():
        param.requires_grad = False
    if args.restore:
        state_dict = torch.load(args.restore)
        synced_model.load_state_dict(state_dict)
    
    if args.virtual_batch_norm:
        print('Computing batch for virtual batch normalization')
        virtual_batch = gather_for_virtual_batch_norm(env, batch_size=args.virtual_batch_norm)
        virtual_batch = torchify(virtual_batch, unsqueeze=False)
    else:
        virtual_batch = None

    if args.test:
        render_env(args, synced_model, env)
    else:
        train_loop(args, synced_model, env, chkpt_dir, virtual_batch=virtual_batch)
