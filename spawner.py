import argparse
from copy import deepcopy
import os
import os.path as osp
import numpy as np
import subprocess
import yaml

from helpers import logger
from helpers.misc_util import zipsame, boolean_flag
from helpers.experiment import uuid as create_uuid


parser = argparse.ArgumentParser(description="Job Spawner")
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--envset', type=str, default=None)
parser.add_argument('--num_demos', '--list', nargs='+', type=str, default=None)
boolean_flag(parser, 'call', default=False, help="launch immediately?")
boolean_flag(parser, 'sweep', default=False, help="hp search?")
args = parser.parse_args()

# Retrieve config from filesystem
CONFIG = yaml.safe_load(open(args.config))

# Extract parameters from config
NUM_SEEDS = CONFIG['parameters']['num_seeds']
NEED_DEMOS = (CONFIG['parameters']['algo'] == 'gail')
if NEED_DEMOS:
    NUM_DEMOS = [int(i) for i in args.num_demos]
else:
    NUM_DEMOS = [0]  # arbitrary, only used for dim checking
CLUSTER = CONFIG['resources']['cluster']
WANDB_PROJECT = CONFIG['resources']['wandb_project'].upper() + '-' + CLUSTER.upper()
CONDA = CONFIG['resources']['conda_env']
# Define experiment type
TYPE = 'sweep' if args.sweep else 'fixed'
# Write out the boolean arguments (using the 'boolean_flag' function)
BOOL_ARGS = ['cuda', 'render', 'record', 'with_scheduler', 'layer_norm', 'shared_value',
             'state_only', 'minimax_only', 'spectral_norm', 'grad_pen', 'one_sided_pen',
             'wrap_absorb', 'd_batch_norm',
             'kye_p', 'kye_mixing', 'adaptive_aux_scaling',
             'red_batch_norm', 'rnd_explo', 'rnd_batch_norm', 'kye_batch_norm', 'dyn_batch_norm',
             'use_purl']

# Create the list of environments from the indicated benchmark
BENCH = CONFIG['parameters']['benchmark']
if BENCH == 'mujoco':
    TOC = {
        'debug': ['Hopper-v3'],
        'flareon': ['InvertedPendulum-v2',
                    'InvertedDoublePendulum-v2',
                    'Hopper-v3'],
        'glaceon': ['Hopper-v3',
                    'Walker2d-v3',
                    'HalfCheetah-v3',
                    'Ant-v3'],
        'humanoid': ['Humanoid-v3'],
        'suite': ['InvertedPendulum-v2',
                  'InvertedDoublePendulum-v2',
                  'Hopper-v3',
                  'Walker2d-v3',
                  'HalfCheetah-v3',
                  'Ant-v3'],
    }
    ENVS = TOC[args.envset]

    if CLUSTER == 'baobab':
        # Define per-environement partitions map
        PEP = {
            'InvertedPendulum': 'shared-EL7,mono-shared-EL7',
            'Reacher': 'shared-EL7,mono-shared-EL7',
            'InvertedDoublePendulum': 'shared-EL7,mono-shared-EL7',
            'Hopper': 'shared-EL7,mono-shared-EL7',
            'Walker2d': 'shared-EL7,mono-shared-EL7',
            'HalfCheetah': 'shared-EL7,mono-shared-EL7',
            'Ant': 'shared-EL7,mono-shared-EL7',
            'Humanoid': 'shared-EL7,mono-shared-EL7',
        }
        # Define per-environment ntasks map
        PEC = {
            'InvertedPendulum': 8,
            'Reacher': 8,
            'InvertedDoublePendulum': 8,
            'Hopper': 16 if NEED_DEMOS else 32,
            'Walker2d': 16 if NEED_DEMOS else 32,
            'HalfCheetah': 16 if NEED_DEMOS else 32,
            'Ant': 16 if NEED_DEMOS else 32,
            'Humanoid': 16 if NEED_DEMOS else 32,
        }
        # Define per-environment timeouts map
        PET = {
            'InvertedPendulum': '0-06:00:00',
            'Reacher': '0-06:00:00',
            'InvertedDoublePendulum': '0-06:00:00',
            'Hopper': '0-06:00:00',
            'Walker2d': '0-06:00:00',
            'HalfCheetah': '0-06:00:00',
            'Ant': '0-06:00:00',
            'Humanoid': '0-12:00:00',
        }

elif BENCH == 'dmc':
    TOC = {
        'debug': ['Hopper-Hop-Feat-v0'],
        'flareon': ['Hopper-Hop-Feat-v0',
                    'Walker-Run-Feat-v0']
    }
    ENVS = TOC[args.envset]

    if CLUSTER == 'baobab':
        # Define per-environement partitions map
        PEP = {
            'Hopper-Hop-Feat': 'shared-EL7,mono-shared-EL7',
        }
        # Define per-environment ntasks map
        PEC = {
            'Hopper-Hop-Feat': 8,
        }
        # Define per-environment timeouts map
        PET = {
            'Hopper-Hop-Feat': '0-06:00:00',
        }

elif BENCH == 'safety':
    TOC = {
        'debug_nohazards': ['Safexp-PointGoal0-v0'],
        'debug_hazards': ['Safexp-PointGoal1-v0'],
        'point_nohazards': ['Safexp-PointGoal0-v0',
                            'Safexp-PointPush0-v0'],
        'point_hazards': ['Safexp-PointGoal1-v0',
                          'Safexp-PointGoal2-v0',
                          'Safexp-PointPush1-v0',
                          'Safexp-PointPush2-v0'],
        'car_nohazards': ['Safexp-CarGoal0-v0',
                          'Safexp-CarPush0-v0'],
        'car_hazards': ['Safexp-CarGoal1-v0',
                        'Safexp-CarGoal2-v0',
                        'Safexp-CarPush1-v0',
                        'Safexp-CarPush2-v0'],
        'doggo_nohazards': ['Safexp-DoggoGoal0-v0',
                            'Safexp-DoggoPush0-v0'],
        'doggo_hazards': ['Safexp-DoggoGoal1-v0',
                          'Safexp-DoggoGoal2-v0',
                          'Safexp-DoggoPush1-v0',
                          'Safexp-DoggoPush2-v0'],
    }
    ENVS = TOC[args.envset]

    if CLUSTER == 'baobab':
        # Define per-environement partitions map
        PEP = {
            'Safexp-PointGoal1': 'shared-EL7,mono-shared-EL7',
            'Safexp-PointGoal2': 'shared-EL7,mono-shared-EL7',
            'Safexp-PointPush1': 'shared-EL7,mono-shared-EL7',
            'Safexp-PointPush2': 'shared-EL7,mono-shared-EL7',
            'Safexp-CarGoal1': 'shared-EL7,mono-shared-EL7',
            'Safexp-CarGoal2': 'shared-EL7,mono-shared-EL7',
            'Safexp-CarPush1': 'shared-EL7,mono-shared-EL7',
            'Safexp-CarPush2': 'shared-EL7,mono-shared-EL7',
            'Safexp-DoggoGoal1': 'shared-EL7,mono-shared-EL7',
            'Safexp-DoggoGoal2': 'shared-EL7,mono-shared-EL7',
            'Safexp-DoggoPush1': 'shared-EL7,mono-shared-EL7',
            'Safexp-DoggoPush2': 'shared-EL7,mono-shared-EL7',
        }
        # Define per-environment ntasks map
        PEC = {
            'Safexp-PointGoal1': 16 if NEED_DEMOS else 32,
            'Safexp-PointGoal2': 16 if NEED_DEMOS else 32,
            'Safexp-PointPush1': 16 if NEED_DEMOS else 32,
            'Safexp-PointPush2': 16 if NEED_DEMOS else 32,
            'Safexp-CarGoal1': 16 if NEED_DEMOS else 32,
            'Safexp-CarGoal2': 16 if NEED_DEMOS else 32,
            'Safexp-CarPush1': 16 if NEED_DEMOS else 32,
            'Safexp-CarPush2': 16 if NEED_DEMOS else 32,
            'Safexp-DoggoGoal1': 16 if NEED_DEMOS else 32,
            'Safexp-DoggoGoal2': 16 if NEED_DEMOS else 32,
            'Safexp-DoggoPush1': 16 if NEED_DEMOS else 32,
            'Safexp-DoggoPush2': 16 if NEED_DEMOS else 32,
        }
        # Define per-environment timeouts map
        PET = {
            'Safexp-PointGoal1': '0-12:00:00',
            'Safexp-PointGoal2': '0-12:00:00',
            'Safexp-PointPush1': '0-12:00:00',
            'Safexp-PointPush2': '0-12:00:00',
            'Safexp-CarGoal1': '0-12:00:00',
            'Safexp-CarGoal2': '0-12:00:00',
            'Safexp-CarPush1': '0-12:00:00',
            'Safexp-CarPush2': '0-12:00:00',
            'Safexp-DoggoGoal1': '0-12:00:00',
            'Safexp-DoggoGoal2': '0-12:00:00',
            'Safexp-DoggoPush1': '0-12:00:00',
            'Safexp-DoggoPush2': '0-12:00:00',
        }

elif BENCH == 'atari':
    TOC = {
        'easy': ['Pong'],
        'normal': ['Qbert',
                   'MsPacman',
                   'SpaceInvaders',
                   'Frostbite',
                   'Freeway',
                   'BeamRider',
                   'Asteroids'],
        'hard_exploration': ['MontezumaRevenge',
                             'Pitfall',
                             'PrivateEye'],
    }
    if args.envset == 'all':
        ENVS = TOC['easy'] + TOC['normal'] + TOC['hard_exploration']
    else:
        ENVS = TOC[args.envset]
    ENVS = ["{}NoFrameskip-v4".format(name) for name in ENVS]
elif BENCH == 'pycolab':
    TOC = {
        'box_world': ['BoxWorld-v0'],
        'cliff_walk': ['CliffWalk-v0'],
    }
    ENVS = TOC[args.envset]

    if CLUSTER == 'baobab':
        # Define per-environement partitions map
        PEP = {
            'BoxWorld': 'shared-EL7,mono-shared-EL7',
            'CliffWalk': 'shared-EL7,mono-shared-EL7',
        }
        # Define per-environment ntasks map
        PEC = {
            'BoxWorld': 16 if NEED_DEMOS else 32,
            'CliffWalk': 16 if NEED_DEMOS else 32,
        }
        # Define per-environment timeouts map
        PET = {
            'BoxWorld': '0-12:00:00',
            'CliffWalk': '0-6:00:00',
        }

else:
    raise NotImplementedError("benchmark not covered by the spawner.")

# If needed, create the list of demonstrations needed
if NEED_DEMOS:
    demo_dir = os.environ['DEMO_DIR']
    DEMOS = {k: osp.join(demo_dir, k) for k in ENVS}


def copy_and_add_seed(hpmap, seed):
    hpmap_ = deepcopy(hpmap)
    # Add the seed and edit the job uuid to only differ by the seed
    hpmap_.update({'seed': seed})
    # Enrich the uuid with extra information
    if NEED_DEMOS:
        hpmap_.update({'uuid': "{}.{}.demos{}.seed{}".format(hpmap['uuid'],
                                                             hpmap['env_id'],
                                                             str(hpmap['num_demos']).zfill(3),
                                                             str(seed).zfill(2))})
    else:
        hpmap_.update({'uuid': "{}.{}.seed{}".format(hpmap['uuid'],
                                                     hpmap['env_id'],
                                                     str(seed).zfill(2))})
    return hpmap_


def copy_and_add_env(hpmap, env):
    hpmap_ = deepcopy(hpmap)
    # Add the env and if demos are needed, add those too
    hpmap_.update({'env_id': env})
    if NEED_DEMOS:
        hpmap_.update({'expert_path': DEMOS[env]})
    return hpmap_


def copy_and_add_num_demos(hpmap, num_demos):
    hpmap_ = deepcopy(hpmap)
    # Add the num of demos
    hpmap_.update({'num_demos': num_demos})
    return hpmap_


def get_hps(sweep):
    """Return a list of maps of hyperparameters"""
    # Create a uuid to identify the current job
    uuid = create_uuid()
    # Assemble the hyperparameter map
    if sweep:
        # Random search
        hpmap = {
            # Primary
            'wandb_project': WANDB_PROJECT,

            # Generic
            'uuid': uuid,
            'cuda': CONFIG['parameters']['cuda'],
            'checkpoint_dir': CONFIG['logging']['checkpoint_dir'],
            'log_dir': CONFIG['logging']['log_dir'],
            'video_dir': CONFIG['logging']['video_dir'],
            'render': False,
            'record': CONFIG['logging'].get('record', False),
            'task': CONFIG['parameters']['task'],
            'algo': CONFIG['parameters']['algo'],

            # Training
            'save_frequency': CONFIG['parameters'].get('save_frequency', 400),
            'num_timesteps': int(float(CONFIG['parameters'].get('num_timesteps', 2e7))),
            'eval_steps_per_iter': CONFIG['parameters'].get('eval_steps_per_iter', 10),
            'eval_frequency': CONFIG['parameters'].get('eval_frequency', 10),

            # Model
            'perception_stack': CONFIG['parameters']['perception_stack'],
            'layer_norm': CONFIG['parameters']['layer_norm'],
            'shared_value': CONFIG['parameters']['shared_value'],

            # Optimization
            'p_lr': float(np.random.choice([1e-3, 3e-4])),
            'v_lr': float(np.random.choice([3e-3, 1e-3])),
            'with_scheduler': CONFIG['parameters']['with_scheduler'],
            'clip_norm': np.random.choice([.5, 1., 20., 40.]),

            # Algorithm
            'rollout_len': np.random.choice([1024, 2048]),
            'optim_epochs_per_iter': np.random.choice([1, 2, 6, 10]),
            'batch_size': np.random.choice([32, 64, 128]),
            'gamma': np.random.choice([0.99, 0.995]),
            'gae_lambda': np.random.choice([0.95, 0.98, 0.99]),
            'eps': np.random.choice([0.1, 0.2, 0.4]),
            'baseline_scale': float(np.random.choice([0.1, 0.3, 0.5])),
            'p_ent_reg_scale': CONFIG['parameters'].get('p_ent_reg_scale', 0.),

            # Adversarial imitation
            'g_steps': CONFIG['parameters'].get('g_steps', 3),
            'd_steps': CONFIG['parameters'].get('d_steps', 1),
            'd_lr': float(CONFIG['parameters'].get('d_lr', 3e-4)),
            'state_only': CONFIG['parameters'].get('state_only', False),
            'minimax_only': CONFIG['parameters'].get('minimax_only', True),
            'd_ent_reg_scale': CONFIG['parameters'].get('d_ent_reg_scale', 0.001),
            'spectral_norm': CONFIG['parameters'].get('spectral_norm', True),
            'grad_pen': CONFIG['parameters'].get('grad_pen', True),
            'grad_pen_type': CONFIG['parameters'].get('grad_pen_type', 'wgan'),
            'one_sided_pen': CONFIG['parameters'].get('one_sided_pen', True),
            'fake_ls_type': np.random.choice(['"random-uniform_0.7_1.2"',
                                              '"soft_labels_0.1"',
                                              '"none"']),
            'real_ls_type': np.random.choice(['"random-uniform_0.7_1.2"',
                                              '"soft_labels_0.1"',
                                              '"none"']),
            'wrap_absorb': CONFIG['parameters'].get('wrap_absorb', False),
            'd_batch_norm': CONFIG['parameters'].get('d_batch_norm', False),
            'red_batch_norm': CONFIG['parameters'].get('red_batch_norm', False),

            'kye_p': CONFIG['parameters'].get('kye_p', False),
            'kye_p_scale': np.random.choice([0.01, 0.1, 0.5]),
            'kye_mixing': CONFIG['parameters'].get('kye_mixing', False),
            'adaptive_aux_scaling': CONFIG['parameters'].get('adaptive_aux_scaling', False),

            'reward_type': CONFIG['parameters'].get('reward_type', 'gail'),

            'red_epochs': CONFIG['parameters'].get('red_epochs', 200),
            'red_lr': CONFIG['parameters'].get('red_lr', 5e-4),
            'proportion_of_exp_per_red_update': CONFIG['parameters'].get('proportion_of_exp_per_red_update', 1.),

            'rnd_explo': CONFIG['parameters'].get('rnd_explo', False),
            'rnd_batch_norm': CONFIG['parameters'].get('rnd_batch_norm', True),
            'rnd_lr': CONFIG['parameters'].get('rnd_lr', 5e-4),
            'proportion_of_exp_per_rnd_update': CONFIG['parameters'].get('proportion_of_exp_per_rnd_update', 1.),

            'kye_batch_norm': CONFIG['parameters'].get('kye_batch_norm', True),
            'kye_lr': CONFIG['parameters'].get('kye_lr', 5e-4),
            'proportion_of_exp_per_kye_update': CONFIG['parameters'].get('proportion_of_exp_per_kye_update', 1.),

            'dyn_batch_norm': CONFIG['parameters'].get('dyn_batch_norm', True),
            'dyn_lr': CONFIG['parameters'].get('dyn_lr', 5e-4),
            'proportion_of_exp_per_dyn_update': CONFIG['parameters'].get('proportion_of_exp_per_dyn_update', 1.),

            'use_purl': CONFIG['parameters'].get('use_purl', False),
            'purl_eta': float(CONFIG['parameters'].get('purl_eta', 0.25)),
        }
    else:
        # No search, fixed map
        hpmap = {
            # Primary
            'wandb_project': WANDB_PROJECT,

            # Generic
            'uuid': uuid,
            'cuda': CONFIG['parameters']['cuda'],
            'checkpoint_dir': CONFIG['logging']['checkpoint_dir'],
            'log_dir': CONFIG['logging']['log_dir'],
            'video_dir': CONFIG['logging']['video_dir'],
            'render': False,
            'record': CONFIG['logging'].get('record', False),
            'task': CONFIG['parameters']['task'],
            'algo': CONFIG['parameters']['algo'],

            # Training
            'save_frequency': CONFIG['parameters'].get('save_frequency', 400),
            'num_timesteps': int(float(CONFIG['parameters'].get('num_timesteps', 2e7))),
            'eval_steps_per_iter': CONFIG['parameters'].get('eval_steps_per_iter', 10),
            'eval_frequency': CONFIG['parameters'].get('eval_frequency', 10),

            # Model
            'perception_stack': CONFIG['parameters']['perception_stack'],
            'layer_norm': CONFIG['parameters']['layer_norm'],
            'shared_value': CONFIG['parameters']['shared_value'],

            # Optimization
            'p_lr': float(CONFIG['parameters'].get('p_lr', 3e-4)),
            'v_lr': float(CONFIG['parameters'].get('v_lr', 1e-3)),
            'with_scheduler': CONFIG['parameters']['with_scheduler'],
            'clip_norm': CONFIG['parameters'].get('clip_norm', 5.0),

            # Algorithm
            'rollout_len': CONFIG['parameters'].get('rollout_len', 2048),
            'optim_epochs_per_iter': CONFIG['parameters'].get('optim_epochs_per_iter', 10),
            'batch_size': CONFIG['parameters'].get('batch_size', 128),
            'gamma': CONFIG['parameters'].get('gamma', 0.995),
            'gae_lambda': CONFIG['parameters'].get('gae_lambda', 0.95),
            'eps': CONFIG['parameters'].get('eps', 0.2),
            'baseline_scale': float(CONFIG['parameters'].get('baseline_scale', 0.5)),
            'p_ent_reg_scale': CONFIG['parameters'].get('p_ent_reg_scale', 0.),

            # Adversarial imitation
            'g_steps': CONFIG['parameters'].get('g_steps', 3),
            'd_steps': CONFIG['parameters'].get('d_steps', 1),
            'd_lr': float(CONFIG['parameters'].get('d_lr', 3e-4)),
            'state_only': CONFIG['parameters'].get('state_only', False),
            'minimax_only': CONFIG['parameters'].get('minimax_only', True),
            'd_ent_reg_scale': CONFIG['parameters'].get('d_ent_reg_scale', 0.001),
            'spectral_norm': CONFIG['parameters'].get('spectral_norm', True),
            'grad_pen': CONFIG['parameters'].get('grad_pen', True),
            'grad_pen_type': CONFIG['parameters'].get('grad_pen_type', 'wgan'),
            'one_sided_pen': CONFIG['parameters'].get('one_sided_pen', True),
            'fake_ls_type': CONFIG['parameters'].get('fake_ls_type', 'none'),
            'real_ls_type': CONFIG['parameters'].get('real_ls_type', 'random-uniform_0.7_1.2'),
            'wrap_absorb': CONFIG['parameters'].get('wrap_absorb', False),
            'd_batch_norm': CONFIG['parameters'].get('d_batch_norm', False),
            'red_batch_norm': CONFIG['parameters'].get('red_batch_norm', False),

            'kye_p': CONFIG['parameters'].get('kye_p', False),
            'kye_p_scale': CONFIG['parameters'].get('kye_p_scale', 0.1),
            'kye_mixing': CONFIG['parameters'].get('kye_mixing', False),
            'adaptive_aux_scaling': CONFIG['parameters'].get('adaptive_aux_scaling', False),

            'reward_type': CONFIG['parameters'].get('reward_type', 'gail'),

            'red_epochs': CONFIG['parameters'].get('red_epochs', 200),
            'red_lr': CONFIG['parameters'].get('red_lr', 5e-4),
            'proportion_of_exp_per_red_update': CONFIG['parameters'].get('proportion_of_exp_per_red_update', 1.),

            'rnd_explo': CONFIG['parameters'].get('rnd_explo', False),
            'rnd_batch_norm': CONFIG['parameters'].get('rnd_batch_norm', True),
            'rnd_lr': CONFIG['parameters'].get('rnd_lr', 5e-4),
            'proportion_of_exp_per_rnd_update': CONFIG['parameters'].get('proportion_of_exp_per_rnd_update', 1.),

            'kye_batch_norm': CONFIG['parameters'].get('kye_batch_norm', True),
            'kye_lr': CONFIG['parameters'].get('kye_lr', 5e-4),
            'proportion_of_exp_per_kye_update': CONFIG['parameters'].get('proportion_of_exp_per_kye_update', 1.),

            'dyn_batch_norm': CONFIG['parameters'].get('dyn_batch_norm', True),
            'dyn_lr': CONFIG['parameters'].get('dyn_lr', 5e-4),
            'proportion_of_exp_per_dyn_update': CONFIG['parameters'].get('proportion_of_exp_per_dyn_update', 1.),

            'use_purl': CONFIG['parameters'].get('use_purl', False),
            'purl_eta': float(CONFIG['parameters'].get('purl_eta', 0.25)),
        }

    # Duplicate for each environment
    hpmaps = [copy_and_add_env(hpmap, env)
              for env in ENVS]

    if NEED_DEMOS:
        # Duplicate for each number of demos
        hpmaps = [copy_and_add_num_demos(hpmap_, num_demos)
                  for hpmap_ in hpmaps
                  for num_demos in NUM_DEMOS]

    # Duplicate for each seed
    hpmaps = [copy_and_add_seed(hpmap_, seed)
              for hpmap_ in hpmaps
              for seed in range(NUM_SEEDS)]

    # Verify that the correct number of configs have been created
    assert len(hpmaps) == NUM_SEEDS * len(ENVS) * len(NUM_DEMOS)

    return hpmaps


def unroll_options(hpmap):
    """Transform the dictionary of hyperparameters into a string of bash options"""
    indent = 4 * ' '  # indents are defined as 4 spaces
    arguments = ""

    for k, v in hpmap.items():
        if k in BOOL_ARGS:
            if v is False:
                argument = "no-{}".format(k)
            else:
                argument = "{}".format(k)
        else:
            argument = "{}={}".format(k, v)

        arguments += "{}--{} \\\n".format(indent, argument)

    return arguments


def create_job_str(name, command, envkey):
    """Build the batch script that launches a job"""

    # Prepend python command with python binary path
    command = osp.join(os.environ['CONDA_PREFIX'], "bin", command)

    if CLUSTER == 'baobab':
        # Set sbatch config
        bash_script_str = ('#!/usr/bin/env bash\n\n')
        bash_script_str += ('#SBATCH --job-name={jobname}\n'
                            '#SBATCH --partition={partition}\n'
                            '#SBATCH --ntasks={ntasks}\n'
                            '#SBATCH --cpus-per-task=1\n'
                            '#SBATCH --time={timeout}\n'
                            '#SBATCH --mem=16000\n'
                            '#SBATCH --output=./out/run_%j.out\n'
                            '#SBATCH --constraint="V3|V4|V5|V6|V7"\n')
        if CONFIG['parameters']['cuda']:
            contraint = "COMPUTE_CAPABILITY_6_0|COMPUTE_CAPABILITY_6_1"
            bash_script_str += ('#SBATCH --gres=gpu:1\n'
                                '#SBATCH --constraint="{}"\n'.format(contraint))
        bash_script_str += ('\n')
        # Load modules
        bash_script_str += ('module load GCC/8.3.0 OpenMPI/3.1.4\n')
        if CONFIG['parameters']['cuda']:
            bash_script_str += ('module load CUDA\n')
        bash_script_str += ('\n')
        # Launch command
        bash_script_str += ('srun {command}')

        bash_script_str = bash_script_str.format(jobname=name,
                                                 partition=PEP[envkey],
                                                 ntasks=PEC[envkey],
                                                 timeout=PET[envkey],
                                                 command=command)

    elif CLUSTER == 'local':
        # Set header
        bash_script_str = ('#!/usr/bin/env bash\n\n')
        bash_script_str += ('# job name: {}\n\n')
        # Launch command
        bash_script_str += ('mpiexec -n {} {}')
        bash_script_str = bash_script_str.format(name,
                                                 CONFIG['resources']['num_workers'],
                                                 command)
    else:
        raise NotImplementedError("cluster selected is not covered.")

    return bash_script_str[:-2]  # remove the last `\` and `\n` tokens


def run(args):
    """Spawn jobs"""

    # Create directory for spawned jobs
    os.makedirs("spawn", exist_ok=True)
    if CLUSTER == 'local':
        os.makedirs("tmux", exist_ok=True)

    # Get the hyperparameter set(s)
    if args.sweep:
        hpmaps_ = [get_hps(sweep=True)
                   for _ in range(CONFIG['parameters']['num_trials'])]
        # Flatten into a 1-dim list
        hpmaps = [x for hpmap in hpmaps_ for x in hpmap]
    else:
        hpmaps = get_hps(sweep=False)

    # Create associated task strings
    commands = ["python main.py \\\n{}".format(unroll_options(hpmap)) for hpmap in hpmaps]
    if not len(commands) == len(set(commands)):
        # Terminate in case of duplicate experiment (extremely unlikely though)
        raise ValueError("bad luck, there are dupes -> Try again (:")
    # Create the job maps
    names = ["{}.{}".format(TYPE, hpmap['uuid']) for i, hpmap in enumerate(hpmaps)]
    # Create environment keys for envionment-specific hyperparameter selection
    envkeys = [hpmap['env_id'].split('-v')[0] for hpmap in hpmaps]

    # Finally get all the required job strings
    jobs = [create_job_str(name, command, envkey)
            for name, command, envkey in zipsame(names, commands, envkeys)]

    # Spawn the jobs
    for i, (name, job) in enumerate(zipsame(names, jobs)):
        logger.info(">>>>>>>>>>>>>>>>>>>> Job #{} ready to submit. Config below.".format(i))
        logger.info(job + "\n")
        dir_ = name.split('.')[1]
        os.makedirs("spawn/{}".format(dir_), exist_ok=True)
        job_name = "spawn/{}/{}.sh".format(dir_, name)
        with open(job_name, 'w') as f:
            f.write(job)
        if args.call and not CLUSTER == 'local':
            # Spawn the job!
            stdout = subprocess.run(["sbatch", job_name]).stdout
            logger.info("[STDOUT]\n{}".format(stdout))
            logger.info(">>>>>>>>>>>>>>>>>>>> Job #{} submitted.".format(i))
    # Summarize the number of jobs spawned
    logger.info(">>>>>>>>>>>>>>>>>>>> {} jobs were spawned.".format(len(jobs)))

    if CLUSTER == 'local':
        dir_ = hpmaps[0]['uuid'].split('.')[0]  # arbitrarilly picked index 0
        session_name = "{}-{}seeds-{}".format(TYPE, str(NUM_SEEDS).zfill(2), dir_)
        yaml_content = {'session_name': session_name,
                        'windows': []}
        if NEED_DEMOS:
            yaml_content.update({'environment': {'DEMO_DIR': os.environ['DEMO_DIR']}})
        for i, name in enumerate(names):
            executable = "{}.sh".format(name)
            pane = {'shell_command': ["source activate {}".format(CONDA),
                                      "chmod u+x spawn/{}/{}".format(dir_, executable),
                                      "spawn/{}/{}".format(dir_, executable)]}
            window = {'window_name': "job{}".format(str(i).zfill(2)),
                      'focus': False,
                      'panes': [pane]}
            yaml_content['windows'].append(window)
        # Dump the assembled tmux config into a yaml file
        job_config = "tmux/{}.yaml".format(session_name)
        with open(job_config, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        if args.call:
            # Spawn all the jobs in the tmux session!
            stdout = subprocess.run(["tmuxp", "load", "-d", "{}".format(job_config)]).stdout
            logger.info("[STDOUT]\n{}".format(stdout))


if __name__ == "__main__":
    # Create (and optionally launch) the jobs!
    run(args)
