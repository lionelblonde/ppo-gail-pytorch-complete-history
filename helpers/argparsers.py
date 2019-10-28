import argparse

from helpers.misc_util import boolean_flag


def argparser(description="PPO/GAIL Experiment"):
    """Create an argparse.ArgumentParser"""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--uuid', type=str, default=None)
    boolean_flag(parser, 'cuda', default=False)
    boolean_flag(parser, 'pixels', default=False)
    boolean_flag(parser, 'recurrent', default=False)
    parser.add_argument('--extractor', type=str, default=None)
    parser.add_argument('--env_id', help='environment identifier', default='Hopper-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--checkpoint_dir', type=str, help='directory to save the models',
                        default='data/checkpoints')
    parser.add_argument('--log_dir', type=str, default='data/logs')
    boolean_flag(parser, 'enable_visdom', default=False)
    parser.add_argument('--visdom_dir', type=str, default='data/summaries')
    parser.add_argument('--visdom_server', type=str, default=None)
    parser.add_argument('--visdom_port', type=int, default=None)
    parser.add_argument('--visdom_username', type=str, default=None)
    parser.add_argument('--visdom_password', type=str, default=None)
    boolean_flag(parser, 'render', help='whether to render the interaction traces', default=False)
    boolean_flag(parser, 'record', help='whether to record the interaction traces', default=False)
    parser.add_argument('--video_dir', default='data/videos')
    parser.add_argument('--video_len', help='duration of the video to record',
                        type=int, default=200)
    parser.add_argument('--task', type=str, choices=['train', 'evaluate'], default=None)
    parser.add_argument('--algo', type=str, choices=['ppo', 'gail'], default=None)
    parser.add_argument('--save_frequency', help='save model every xx iterations',
                        type=int, default=100)
    parser.add_argument('--num_iters', help='cummulative number of iterations since launch',
                        type=int, default=int(1e6))
    parser.add_argument('--rollout_len', help='number of interactions per iteration',
                        type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_trajs', help='number of trajectories to evaluate',
                        type=int, default=10)
    parser.add_argument('--iter_num', help='iteration to evaluate the model at',
                        type=int, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--p_lr', type=float, default=3e-4)
    boolean_flag(parser, 'with_scheduler', default=False)
    parser.add_argument('--d_lr', type=float, default=3e-4)
    parser.add_argument('--hidden_state_size', type=int, default=None)
    boolean_flag(parser, 'with_layernorm', default=False)
    parser.add_argument('--optim_epochs_per_iter', type=int, default=10)
    parser.add_argument('--gamma', help='discount factor', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--eps', help='ppo ratio clipping value', type=float, default=0.2)
    parser.add_argument('--clip_norm', type=float, default=None)
    boolean_flag(parser, 'state_only', default=False)
    boolean_flag(parser, 'minimax_only', default=True)
    parser.add_argument('--p_ent_reg_scale', help='p entropy reg coeff', type=float, default=0.)
    parser.add_argument('--d_ent_reg_scale', help='d entropy reg coeff', type=float, default=0.)
    parser.add_argument('--eval_steps_per_iter', type=int, default=50)
    parser.add_argument('--eval_frequency', type=int, default=20)
    parser.add_argument('--d_update_ratio', type=int, default=5,
                        help='number of discriminator update per generator update')
    parser.add_argument('--expert_path', help='.npz archive containing the demos',
                        type=str, default=None)
    parser.add_argument('--num_demos', help='number of expert demo trajs for imitation',
                        type=int, default=None)
    return parser
