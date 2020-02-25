import argparse
from copy import deepcopy
import os
import os.path as osp
import numpy as np
from subprocess import check_output
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
BOOL_ARGS = ['cuda', 'norm_obs', 'clip_obs', 'binned_aux_loss', 'squared_aux_loss',
             'render', 'record', 'with_scheduler', 'shared_value',
             'state_only', 'minimax_only', 'grad_pen',
             'use_purl']

# Create the list of environments from the indicated benchmark
BENCH = CONFIG['parameters']['benchmark']
if BENCH == 'mujoco':
    TOC = {'debug': ['Hopper-v3'],
           'easy': ['InvertedPendulum-v2',
                    'InvertedDoublePendulum-v2'],
           'hard': ['Hopper-v3',
                    'Walker2d-v3',
                    'HalfCheetah-v3',
                    'Ant-v3',
                    'Humanoid-v3']
           }
    if args.envset == 'all':
        ENVS = TOC['easy'] + TOC['hard']
    else:
        ENVS = TOC[args.envset]

    if CLUSTER == 'baobab':
        # Define per-environement partitions map
        PEP = {'InvertedPendulum': 'shared-EL7,mono-shared-EL7',
               'Reacher': 'shared-EL7,mono-shared-EL7',
               'InvertedDoublePendulum': 'shared-EL7,mono-shared-EL7',
               'Hopper': 'shared-EL7,mono-shared-EL7',
               'Walker2d': 'shared-EL7,mono-shared-EL7',
               'HalfCheetah': 'shared-EL7,mono-shared-EL7',
               'Ant': 'shared-EL7,mono-shared-EL7',
               'Humanoid': 'shared-EL7,mono-shared-EL7'}
        # Define per-environment ntasks map
        PEC = {'InvertedPendulum': '20',
               'Reacher': '20',
               'InvertedDoublePendulum': '20',
               'Hopper': '40',
               'Walker2d': '40',
               'HalfCheetah': '40',
               'Ant': '40',
               'Humanoid': '60'}
        # Define per-environment timeouts map
        PET = {'InvertedPendulum': '0-06:00:00',
               'Reacher': '0-06:00:00',
               'InvertedDoublePendulum': '0-06:00:00',
               'Hopper': '0-12:00:00',
               'Walker2d': '0-12:00:00',
               'HalfCheetah': '0-12:00:00',
               'Ant': '0-12:00:00',
               'Humanoid': '0-12:00:00'}

elif BENCH == 'atari':
    map_ = {'easy': ['Pong'],
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
    map_ = {'box_world': ['BoxWorld-v0'],
            'cliff_world': ['CliffWalk-v0'],
            }
    ENVS = TOC[args.envset]
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
    hpmap_.update({'uuid': "{}.{}.demos{}.seed{}".format(hpmap['uuid'],
                                                         hpmap['env_id'],
                                                         str(hpmap['num_demos']).zfill(3),
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
            'shared_value': CONFIG['parameters']['shared_value'],

            # Optimization
            'p_lr': float(np.random.choice([1e-3, 3e-4])),
            'v_lr': float(np.random.choice([3e-3, 1e-3])),
            'with_scheduler': CONFIG['parameters']['with_scheduler'],
            'clip_norm': np.random.choice([.5, 1., 20., 40.]),

            # Algorithm
            'norm_obs': CONFIG['parameters'].get('norm_obs', False),
            'clip_obs': CONFIG['parameters'].get('clip_obs', False),
            'rollout_len': np.random.choice([1024, 2048]),
            'optim_epochs_per_iter': np.random.choice([1, 2, 6, 10]),
            'batch_size': np.random.choice([32, 64, 128]),
            'gamma': np.random.choice([0.99, 0.995]),
            'gae_lambda': np.random.choice([0.95, 0.98, 0.99]),
            'eps': np.random.choice([0.1, 0.2, 0.4]),
            'baseline_scale': float(np.random.choice([0.1, 0.3, 0.5])),
            'p_ent_reg_scale': CONFIG['parameters'].get('p_ent_reg_scale', 0.),
            'binned_aux_loss': CONFIG['parameters'].get('binned_aux_loss', False),
            'squared_aux_loss': CONFIG['parameters'].get('squared_aux_loss', False),
            'ss_aux_loss_scale': np.random.choice([0.001, 0.01, 0.1]),

            # Adversarial imitation
            'd_lr': float(CONFIG['parameters'].get('d_lr', 3e-4)),
            'state_only': CONFIG['parameters'].get('state_only', False),
            'minimax_only': CONFIG['parameters'].get('minimax_only', True),
            'd_ent_reg_scale': CONFIG['parameters'].get('d_ent_reg_scale', 0.),
            'd_update_ratio': CONFIG['parameters'].get('d_update_ratio', 2),
            'num_demos': CONFIG['parameters'].get('num_demos', 0),
            'grad_pen': CONFIG['parameters'].get('grad_pen', True),
            'fake_ls_type': np.random.choice(['"random-uniform_0.7_1.2"',
                                              '"soft_labels_0.1"',
                                              '"none"']),
            'real_ls_type': np.random.choice(['"random-uniform_0.7_1.2"',
                                              '"soft_labels_0.1"',
                                              '"none"']),

            # PU
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
            'shared_value': CONFIG['parameters']['shared_value'],

            # Optimization
            'p_lr': float(CONFIG['parameters'].get('p_lr', 3e-4)),
            'v_lr': float(CONFIG['parameters'].get('v_lr', 1e-3)),
            'with_scheduler': CONFIG['parameters']['with_scheduler'],
            'clip_norm': CONFIG['parameters'].get('clip_norm', 5.0),

            # Algorithm
            'norm_obs': CONFIG['parameters'].get('norm_obs', False),
            'clip_obs': CONFIG['parameters'].get('clip_obs', False),
            'rollout_len': CONFIG['parameters'].get('rollout_len', 2048),
            'optim_epochs_per_iter': CONFIG['parameters'].get('optim_epochs_per_iter', 10),
            'batch_size': CONFIG['parameters'].get('batch_size', 128),
            'gamma': CONFIG['parameters'].get('gamma', 0.995),
            'gae_lambda': CONFIG['parameters'].get('gae_lambda', 0.95),
            'eps': CONFIG['parameters'].get('eps', 0.2),
            'baseline_scale': float(CONFIG['parameters'].get('baseline_scale', 0.5)),
            'p_ent_reg_scale': CONFIG['parameters'].get('p_ent_reg_scale', 0.),
            'binned_aux_loss': CONFIG['parameters'].get('binned_aux_loss', False),
            'squared_aux_loss': CONFIG['parameters'].get('squared_aux_loss', False),
            'ss_aux_loss_scale': CONFIG['parameters'].get('ss_aux_loss_scale', 0.1),

            # Adversarial imitation
            'd_lr': float(CONFIG['parameters'].get('d_lr', 3e-4)),
            'state_only': CONFIG['parameters'].get('state_only', False),
            'minimax_only': CONFIG['parameters'].get('minimax_only', True),
            'd_ent_reg_scale': CONFIG['parameters'].get('d_ent_reg_scale', 0.),
            'd_update_ratio': CONFIG['parameters'].get('d_update_ratio', 2),
            'num_demos': CONFIG['parameters'].get('num_demos', 0),
            'grad_pen': CONFIG['parameters'].get('grad_pen', True),
            'fake_ls_type': CONFIG['parameters'].get('fake_ls_type', 'none'),
            'real_ls_type': CONFIG['parameters'].get('real_ls_type', 'random-uniform_0.7_1.2'),

            # PU
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
                            '#SBATCH --mem=32000\n'
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
    envkeys = [hpmap['env_id'].split('-')[0] for hpmap in hpmaps]

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
            check_output(["sbatch", job_name])
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
            check_output(["tmuxp", "load", "-d", "{}".format(job_config)])


if __name__ == "__main__":
    # Create (and optionally launch) the jobs!
    run(args)
