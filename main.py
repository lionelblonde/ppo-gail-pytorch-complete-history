import random

from mpi4py import MPI
import numpy as np
import torch

from helpers import logger
from helpers.argparsers import argparser
from helpers.experiment import ExperimentInitializer
from helpers.distributed_util import setup_mpi_gpus
from helpers.env_makers import make_env
from agents import orchestrator
from helpers.dataset import DemoDataset
from agents.ppo_agent import PPOAgent
from agents.gail_agent import GAILAgent


def train(args):
    """Train an agent"""

    # Get the current process rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    torch.set_num_threads(1)

    # Initialize and configure experiment
    experiment = ExperimentInitializer(args, rank=rank, world_size=world_size)
    experiment.configure_logging()
    # Create experiment name
    experiment_name = experiment.get_name()

    # Set device-related knobs
    if args.cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda:0")
        setup_mpi_gpus()
    else:
        device = torch.device("cpu")
    logger.info("device in use: {}".format(device))

    # Seedify
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    worker_seed = args.seed + (1000000 * (rank + 1))
    eval_seed = args.seed + 1000000

    # Create environment
    env = make_env(args.env_id, worker_seed)

    expert_dataset = None

    # Create an agent wrapper
    if args.algo == 'ppo':
        def agent_wrapper():
            return PPOAgent(env=env, device=device, hps=args)

    elif args.algo == 'gail':
        # Create the expert demonstrations dataset from expert trajectories
        expert_dataset = DemoDataset(
            expert_path=args.expert_path,
            num_demos=args.num_demos,
            env=env,
            wrap_absorb=args.wrap_absorb,
        )

        def agent_wrapper():
            return GAILAgent(env=env, device=device, hps=args,
                             expert_dataset=expert_dataset)

    else:
        raise NotImplementedError("algorithm not covered")

    # Create an evaluation environment not to mess up with training rollouts
    eval_env = None
    if rank == 0:
        eval_env = make_env(args.env_id, eval_seed)

    # Train
    orchestrator.learn(args=args,
                       rank=rank,
                       world_size=world_size,
                       env=env,
                       eval_env=eval_env,
                       agent_wrapper=agent_wrapper,
                       experiment_name=experiment_name)

    # Close environment
    env.close()

    # Close the eval env
    if eval_env is not None:
        assert rank == 0
        eval_env.close()


def evaluate(args):
    """Evaluate an agent"""

    # Initialize and configure experiment
    experiment = ExperimentInitializer(args)
    experiment.configure_logging()
    # Create experiment name
    experiment_name = experiment.get_name()

    # Seedify
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create environment
    env = make_env(args.env_id, args.seed)

    # Create an agent wrapper
    if args.algo == 'ppo':
        def agent_wrapper():
            return PPOAgent(env=env, device='cpu', hps=args)

    elif args.algo == 'gail':
        def agent_wrapper():
            return GAILAgent(env=env, device='cpu', hps=args)

    else:
        raise NotImplementedError("algorithm not covered")

    # Evaluate
    orchestrator.evaluate(args=args,
                          env=env,
                          agent_wrapper=agent_wrapper,
                          experiment_name=experiment_name,
                          num_trajs=args.num_trajs,
                          iter_num=args.iter_num,
                          model_path=args.model_path)

    # Close environment
    env.close()


if __name__ == '__main__':
    _args = argparser().parse_args()
    if _args.task == 'train':
        train(_args)
    elif _args.task == 'eval':
        evaluate(_args)
    else:
        raise NotImplementedError
