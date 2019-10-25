import yaml

import gym

from helpers import logger
from helpers.wrappers import PixelObsEnv, FrameStack


def get_benchmark(env_id):
    """Verify that the specified env is amongst the admissible ones"""
    envs = yaml.safe_load(open("admissible_envs.yaml"))['environments']
    benchmark = None
    for k, v in envs.items():
        if env_id in list(v.keys()):
            benchmark = k
    assert benchmark is not None, "env not found in 'project_root/admissible_envs.yml'"
    logger.info("env_id = {} <- admissibility check passed!".format(env_id))
    return benchmark


def make_mujoco_env(env_id, seed, pixels, width=84, height=84, grayscale=True, k=4):
    """Create a wrapped gym.Env for MuJoCo"""
    env = gym.make(env_id)
    env.seed(seed)
    # Apply wrappers
    if pixels:
        env = PixelObsEnv(env, width, height, grayscale)
        env = FrameStack(env, k)
    return env


def make_classic_env(env_id, seed):
    """Create a wrapped gym.Env for VM"""
    env = gym.make(env_id)
    env.seed(seed)
    return env


def make_env(env_id, seed, pixels, width, height, grayscale, k):
    """Create an environment"""
    benchmark = get_benchmark(env_id)
    if benchmark == 'mujoco':
        return make_mujoco_env(env_id, seed, pixels, width, height, grayscale, k)
    elif benchmark == 'classic':
        return make_classic_env(env_id, seed)
    else:
        raise RuntimeError("unknown benchmark")