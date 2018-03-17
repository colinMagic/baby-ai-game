import argparse

import torch


def get_args():

    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed (default: 1234).')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='mini-batch size')
    parser.add_argument(
        '--env-name',
        type=str,
        default='MiniGrid-Empty-6x6-v0',
        help='gym environment to load')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='parameter for GAE (default: 0.95)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help="Coefficient associated with the critic loss.")
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help="Coefficient associated with the policy entropy.")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max value of the gradient norm (default: 0.5)')
    parser.add_argument('--max-iters', type=int, default=10000000,
                        help='maximum training iterations (default: 10000000)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number max of forward steps in AC (default: 5).' +
        ' Use 0 to go through complete episodes before updating.')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--num-checkpoints', default=5,
                        type=int, help="Number of check points (default: 5).")
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='interval between training status logs (default: 10)')
    parser.add_argument(
        '--expt-dir',
        type=str,
        default='./experiment',
        help='Path to experiment directory. If load_checkpoint ' +
        'is True, then path to checkpoint directory has to be ' +
        'provided')
    parser.add_argument('--num-processes', type=int, default=16,
                        help='how many training processes to use ' +
                        '(default: 16)')
    parser.add_argument(
        '--load-checkpoint',
        action='store',
        help='The name of ' +
        'the checkpoint to load, usually an encoded time string')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Indicates if training has to be resumed from ' +
                        'the latest checkpoint')
    parser.add_argument(
        '--vis-interval',
        type=int,
        default=100,
        help='vis interval, one log per n updates (default: 100)')

    parser.add_argument('--no-vis', action='store_true', default=False,
                        help='disables visdom visualization')

    parser.add_argument('--num-stack', type=int, default=1,
                        help='number of frames to stack (default: 1)')

    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | ppo | acktr')

    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--ppo-num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--ppo-clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')

    #
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis

    if not args.cuda:
        print('*** WARNING: CUDA NOT ENABLED ***')

    return args
