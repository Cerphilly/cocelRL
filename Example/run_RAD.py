import argparse
import torch
from Algorithm.ImageRL.RAD import RAD

from Trainer.Basic_trainer import Basic_trainer
from Common.Utils import set_seed, dmc_image_env, dmcr_env

def hyperparameters():
    parser = argparse.ArgumentParser(description='Reinforcement learning with Augmented Data (RAD) example')
    #environment
    parser.add_argument('--algorithm', default='SACv2', help='SACv2')
    parser.add_argument('--domain_type', default='dmc/image', type=str, help='gym or dmc')
    parser.add_argument('--env-name', default='cartpole/swingup', help='DM Control Suite domain name + task name')
    parser.add_argument('--discrete', default=False, type=bool, help='Always Continuous')
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--training-start', default=1000, type=int, help='First step to start training')
    parser.add_argument('--max-step', default=100001, type=int, help='Maximum training step')
    parser.add_argument('--eval', default=True, type=bool, help='whether to perform evaluation')
    parser.add_argument('--eval-step', default=10000, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--eval-episode', default=10, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--random-seed', default=1234, type=int, help='Random seed setting')

    parser.add_argument('--frame-stack', default=3, type=int)
    parser.add_argument('--frame-skip', default=8, type=int)
    parser.add_argument('--image-size', default=84, type=int)
    parser.add_argument('--pre-image-size', default=100, type=int)
    parser.add_argument('--data-augs', default='crop', type=str, help='data augmentation: crop, grayscale, random cutout, etc')
    #parser.add_argument('--data-augs', default='no_aug', type=str, help='no aug if you want to do SAC:pixel')

    #sac
    parser.add_argument('--batch-size', default=256, type=int, help='Mini-batch size')
    parser.add_argument('--buffer-size', default=100000, type=int, help='Buffer maximum size')
    parser.add_argument('--train-mode', default='online', help='online')
    parser.add_argument('--training-step', default=1, type=int)
    parser.add_argument('--train-alpha', default=True, type=bool)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--actor-lr', default=0.001, type=float)
    parser.add_argument('--critic-lr', default=0.001, type=float)
    parser.add_argument('--v-lr', default=0.001, type=float)
    parser.add_argument('--alpha-lr', default=0.0001, type=float)
    parser.add_argument('--encoder-lr', default=0.001, type=float)

    parser.add_argument('--tau', default=0.01, type=float)
    parser.add_argument('--actor-update', default=1, type=int)
    parser.add_argument('--critic-update', default=2, type=int)
    parser.add_argument('--hidden-dim', default=(512, 512), help='hidden dimension of network')
    parser.add_argument('--log_std_min', default=-10, type=int, help='For squashed gaussian actor')
    parser.add_argument('--log_std_max', default=2, type=int, help='For squashed gaussian actor')

    #rad&encoder
    parser.add_argument('--layer-num', default=4, type=int)
    parser.add_argument('--filter-num', default=32, type=int)
    parser.add_argument('--encoder-tau', default=0.01, type=float)
    parser.add_argument('--feature-dim', default=50, type=int)

    parser.add_argument('--cpu-only', default=False, type=bool, help='force to use cpu only')
    parser.add_argument('--log', default=True, type=bool, help='use tensorboard summary writer to log, if false, cannot use the features below')
    parser.add_argument('--tensorboard', default=True, type=bool, help='when logged, write in tensorboard')
    parser.add_argument('--file', default=True, type=bool, help='when logged, write log')
    parser.add_argument('--numpy', default=True, type=bool, help='when logged, save log in numpy')

    parser.add_argument('--model', default=True, type=bool, help='when logged, save model')
    parser.add_argument('--model-freq', default=10000, type=int, help='model saving frequency')
    parser.add_argument('--buffer', default=False, type=bool, help='when logged, save buffer')
    parser.add_argument('--buffer-freq', default=100000, type=int, help='buffer saving frequency')


    args = parser.parse_args()

    return args

def main(args):
    if args.cpu_only == True:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Device: ", device)
    # random seed setting
    random_seed = set_seed(args.random_seed)
    print("Random Seed:", random_seed)

    if 'crop' not in args.data_augs.split('-'):
        assert args.pre_image_size == args.image_size
    else:
        assert args.pre_image_size != args.image_size

    if args.domain_type == 'dmc/image':
        env, test_env = dmc_image_env(args.env_name, args.pre_image_size, args.frame_stack, args.frame_skip, random_seed)

    elif args.domain_type == 'dmcr':
        env, test_env = dmcr_env(args.env_name, args.pre_image_size, args.frame_skip, random_seed, mode='classic')

    state_dim = (3 * args.frame_stack, args.image_size, args.image_size)
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]

    if args.algorithm == 'SACv1':
        raise NotImplementedError
    elif args.algorithm == 'SACv2':
        algorithm = RAD(state_dim, action_dim, device, args)

    print("Training of", args.domain_type + '_' + args.env_name)
    print("Algorithm:", algorithm.name)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Max action:", max_action)
    print("Min action:", min_action)

    trainer = Basic_trainer(env, test_env, algorithm, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    args = hyperparameters()
    main(args)

