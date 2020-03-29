import argparse

from helpers.misc_util import boolean_flag


def argparser(description="PPO/GAIL Experiment"):
    """Create an argparse.ArgumentParser"""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Primary
    parser.add_argument('--wandb_project', help='wandb project name', default='DEFAULT')
    parser.add_argument('--env_id', help='environment identifier', default='Hopper-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', help='demos location', type=str, default=None)

    # Generic
    parser.add_argument('--uuid', type=str, default=None)
    boolean_flag(parser, 'cuda', default=False)
    parser.add_argument('--checkpoint_dir', type=str, default='data/checkpoints')
    parser.add_argument('--log_dir', type=str, default='data/logs')
    boolean_flag(parser, 'render', help='render the interaction traces', default=False)
    boolean_flag(parser, 'record', help='record the interaction traces', default=False)
    parser.add_argument('--task', type=str, choices=['train', 'eval'], default=None)
    parser.add_argument('--algo', type=str, choices=['ppo', 'gail'], default=None)

    # Training
    parser.add_argument('--save_frequency', help='save model every xx iterations',
                        type=int, default=100)
    parser.add_argument('--num_timesteps', help='total number of interactions',
                        type=int, default=int(2e7))
    parser.add_argument('--eval_steps_per_iter', type=int, default=10)
    parser.add_argument('--eval_frequency', type=int, default=10)

    # Model
    parser.add_argument('--perception_stack', type=str, default=None)
    boolean_flag(parser, 'shared_value', default=False)

    # Optimization
    parser.add_argument('--p_lr', type=float, default=3e-4)
    parser.add_argument('--v_lr', type=float, default=1e-3)
    boolean_flag(parser, 'with_scheduler', default=False)
    parser.add_argument('--clip_norm', type=float, default=None)

    # Algorithm
    parser.add_argument('--rollout_len', help='number of interactions per iteration',
                        type=int, default=1024)
    parser.add_argument('--optim_epochs_per_iter', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gamma', help='discount factor', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--eps', help='ppo ratio clipping value', type=float, default=0.2)
    parser.add_argument('--baseline_scale', type=float, default=0.5)
    parser.add_argument('--p_ent_reg_scale', help='p entropy reg coeff', type=float, default=0.)

    # Adversarial imitation
    parser.add_argument('--g_steps', type=int, default=3)
    parser.add_argument('--d_steps', type=int, default=1)
    parser.add_argument('--d_lr', type=float, default=3e-4)
    boolean_flag(parser, 'state_only', default=False)
    boolean_flag(parser, 'minimax_only', default=True)
    parser.add_argument('--d_ent_reg_scale', help='d entropy reg coeff', type=float, default=0.001)
    parser.add_argument('--num_demos', help='number of expert demo trajs for imitation',
                        type=int, default=None)
    boolean_flag(parser, 'spectral_norm', help='whether to use spectral norm', default=True)
    boolean_flag(parser, 'grad_pen', help='whether to use gradient penalty', default=True)
    parser.add_argument('--fake_ls_type', type=str, default='none')
    parser.add_argument('--real_ls_type', type=str, default='random-uniform_0.7_1.2')
    parser.add_argument('--syn_rew_scale', type=float, default=1.0)
    boolean_flag(parser, 'wrap_absorb', default=False)

    # KYE
    boolean_flag(parser, 'kye_p', default=False)
    parser.add_argument('--kye_p_scale', type=float, default=0.1)
    boolean_flag(parser, 'kye_d', default=False)
    parser.add_argument('--kye_d_scale', type=float, default=0.1)
    boolean_flag(parser, 'kye_mixing', default=True)

    # PU
    boolean_flag(parser, 'use_purl', default=False)
    parser.add_argument('--purl_eta', type=float, default=0.25)

    # Evaluation
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--num_trajs', help='number of trajectories to evaluate',
                        type=int, default=10)
    parser.add_argument('--iter_num', help='iteration to evaluate the model at',
                        type=int, default=None)

    return parser
