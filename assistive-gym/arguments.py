import argparse, multiprocessing, torch

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='ppo',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_false', default=True,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.0,
                        help='entropy term coefficient (default: 0.0)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num-processes', type=int, default=0,
                        help='how many training CPU processes to use (default: 0 (max))')
    parser.add_argument('--num-steps', type=int, default=200,
                        help='number of forward steps in A2C/PPO (default: 200)')
    parser.add_argument('--ppo-epoch', type=int, default=10,
                        help='number of ppo epochs (default: 10)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=1,
                        help='log interval, one log per n updates (default: 1)')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='save interval, one save per n updates (default: 10)')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num-env-steps', type=int, default=10000000,
                        help='number of environment steps to train (default: 10000000)')
    parser.add_argument('--env-name', default='ScratchItchJaco-v0',
                        help='environment to train on (default: ScratchItchJaco-v0)')
    parser.add_argument('--log-dir', default='/tmp/gym/',
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--load-policy', default=None,
                        help='Policy to load existing agent logs to continue training (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_false', default=True,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--use-linear-clip-decay', action='store_true', default=False,
                        help='use a linear schedule on the ppo clipping parameter')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='enable visdom visualization')
    parser.add_argument('--port', type=int, default=8097,
                        help='port to run the server on (default: 8097)')
    parser.add_argument('--homotopy_mode', type=str, default='up',
                        help='decide homotopy class') 
    parser.add_argument('--obs_size', type=float, default=1.,
                        help='obstacle size')
    parser.add_argument('--rew_factor', type=float, default=1.,
                        help='factor on the reward') 
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.num_processes <= 0:
        # Determine the maximum number of CPUs available
        args.num_processes = multiprocessing.cpu_count()

    return args
