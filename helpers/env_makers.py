import gym
import safety_gym  # noqa

from helpers.atari_wrappers import wrap_atari

from helpers.pycolab_envs import make_pycolab

import environments


def get_benchmark(env_id):
    """Verify that the specified env is amongst the admissible ones"""
    for k, v in environments.BENCHMARKS.items():
        if env_id in v:
            benchmark = k
            continue
    assert benchmark is not None, "unsupported environment"
    return benchmark


def make_env(env_id, seed):
    """Create an environment"""
    benchmark = get_benchmark(env_id)
    if benchmark == 'pycolab':
        env = make_pycolab(env_id)
        return env
    env = gym.make(env_id)
    env.seed(seed)
    if benchmark in ['mujoco', 'safety']:
        pass
    elif benchmark == 'atari':
        env = wrap_atari(env)
    else:
        raise ValueError('unsupported benchmark')
    return env
