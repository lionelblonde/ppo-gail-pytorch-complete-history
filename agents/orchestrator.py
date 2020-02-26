import time
from copy import deepcopy
import os
import os.path as osp
from collections import defaultdict

import wandb
import numpy as np

from gym import spaces

from helpers import logger
from helpers.distributed_util import mpi_mean_reduce
# from helpers.distributed_util import sync_check
from agents.memory import RingBuffer
from helpers.env_makers import get_benchmark
from helpers.console_util import timed_cm_wrapper, log_iter_info


def rollout_generator(env, agent, rollout_len):

    t = 0
    done = True
    # Reset agent's env
    ob = np.array(env.reset())
    # Init collections
    obs0 = RingBuffer(rollout_len, shape=agent.ob_shape)
    obs1 = RingBuffer(rollout_len, shape=agent.ob_shape)
    acs = RingBuffer(rollout_len, shape=((1,)
                                         if isinstance(agent.ac_space, spaces.Discrete)
                                         else agent.ac_shape))
    vs = RingBuffer(rollout_len, shape=(1,), dtype=np.float32)
    logps = RingBuffer(rollout_len, shape=(1,), dtype=np.float32)
    syn_rews = RingBuffer(rollout_len, shape=(1,), dtype=np.float32)
    env_rews = RingBuffer(rollout_len, shape=(1,), dtype=np.float32)
    dones = RingBuffer(rollout_len, shape=(1,), dtype=np.int32)
    # Init current episode statistics
    cur_ep_len = 0
    cur_ep_env_ret = 0
    if agent.hps.algo == 'gail':
        cur_ep_syn_ret = 0
    # Init global episodic statistics
    ep_lens = []
    ep_env_rets = []
    if agent.hps.algo == 'gail':
        ep_syn_rets = []

    while True:

        # Predict
        ac, v, logp = agent.predict(ob, sample_or_mode=True)

        if not isinstance(agent.ac_space, spaces.Discrete):
            # NaN-proof and clip
            ac = np.nan_to_num(ac)
            ac = np.clip(ac, env.action_space.low, env.action_space.high)
        else:
            ac = ac if isinstance(ac, int) else np.asscalar(ac)

        if t > 0 and t % rollout_len == 0:
            out = {
                # Data collected during the rollout
                "obs0": obs0.data.reshape(-1, *agent.ob_shape),
                "obs1": obs1.data.reshape(-1, *agent.ob_shape),
                "acs": (acs.data.reshape(-1, *((1,)
                        if isinstance(agent.ac_space, spaces.Discrete)
                        else agent.ac_shape))),
                "vs": vs.data.reshape(-1, 1),
                "logps": logps.data.reshape(-1, 1),
                "env_rews": env_rews.data.reshape(-1, 1),
                "dones": dones.data.reshape(-1, 1),
                "next_v": v * (1 - done),
                # Global episodic statistics
                "ep_lens": ep_lens,
                "ep_env_rets": ep_env_rets,
            }
            if agent.hps.algo == 'gail':
                out.update({
                    "syn_rews": syn_rews.data.reshape(-1, 1),
                    "ep_syn_rets": ep_syn_rets,
                })
            # Yield
            yield out
            # Reset global episodic statistics
            ep_lens = []
            ep_env_rets = []
            if agent.hps.algo == 'gail':
                ep_syn_rets = []

        # Interact with env(s)
        new_ob, env_rew, done, _ = env.step(ac)

        # Populate collections
        obs0.append(ob)
        obs1.append(new_ob)
        acs.append(ac)
        vs.append(v)
        logps.append(logp)
        env_rews.append(env_rew)
        dones.append(done)

        if agent.hps.algo == 'gail':
            # Get synthetic rewards
            syn_rew = np.asscalar(agent.get_reward(ob, ac, new_ob)[0].cpu().numpy().flatten())
            syn_rews.append(syn_rew)

        # Set current state with the next
        ob = np.array(deepcopy(new_ob))

        # Update current episode statistics
        cur_ep_len += 1
        cur_ep_env_ret += env_rew
        if agent.hps.algo == 'gail':
            cur_ep_syn_ret += syn_rew

        if done:
            # Update the global episodic statistics and
            # reset current episode statistics
            ep_lens.append(cur_ep_len)
            cur_ep_len = 0
            ep_env_rets.append(cur_ep_env_ret)
            cur_ep_env_ret = 0
            if agent.hps.algo == 'gail':
                ep_syn_rets.append(cur_ep_syn_ret)
                cur_ep_syn_ret = 0
            # Reset env
            ob = np.array(env.reset())

        t += 1


def ep_generator(env, agent, render, record):
    """Generator that spits out a trajectory collected during a single episode
    `append` operation is also significantly faster on lists than numpy arrays,
    they will be converted to numpy arrays once complete and ready to be yielded.
    """

    if record:
        benchmark = get_benchmark(agent.hps.env_id)

        def bgr_to_rgb(x):
            _b = np.expand_dims(x[..., 0], -1)
            _g = np.expand_dims(x[..., 1], -1)
            _r = np.expand_dims(x[..., 2], -1)
            rgb_x = np.concatenate([_r, _g, _b], axis=-1)
            del x, _b, _g, _r
            return rgb_x

        kwargs = {'mode': 'rgb_array'}
        if benchmark == 'atari':
            def _render():
                return bgr_to_rgb(env.render(**kwargs))
        elif benchmark in ['mujoco', 'pycolab']:
            def _render():
                return env.render(**kwargs)
        else:
            raise ValueError('unsupported benchmark')

    ob = np.array(env.reset())

    if record:
        ob_orig = _render()

    cur_ep_len = 0
    cur_ep_env_ret = 0
    obs = []
    if record:
        obs_render = []
    acs = []
    vs = []
    env_rews = []

    while True:
        ac, v, _ = agent.predict(ob, sample_or_mode=False)

        if not isinstance(agent.ac_space, spaces.Discrete):
            # NaN-proof and clip
            ac = np.nan_to_num(ac)
            ac = np.clip(ac, env.action_space.low, env.action_space.high)
        else:
            ac = ac if isinstance(ac, int) else np.asscalar(ac)

        obs.append(ob)
        if record:
            obs_render.append(ob_orig)
        acs.append(ac)
        vs.append(v)
        new_ob, env_rew, done, _ = env.step(ac)

        if render:
            env.render()

        if record:
            ob_orig = _render()

        env_rews.append(env_rew)
        cur_ep_len += 1
        cur_ep_env_ret += env_rew
        ob = np.array(deepcopy(new_ob))
        if done:
            obs = np.array(obs)
            if record:
                obs_render = np.array(obs_render)
            acs = np.array(acs)
            env_rews = np.array(env_rews)
            out = {"obs": obs,
                   "acs": acs,
                   "vs": vs,
                   "env_rews": env_rews,
                   "ep_len": cur_ep_len,
                   "ep_env_ret": cur_ep_env_ret}
            if record:
                out.update({"obs_render": obs_render})

            yield out

            cur_ep_len = 0
            cur_ep_env_ret = 0
            obs = []
            if record:
                obs_render = []
            acs = []
            env_rews = []
            ob = np.array(env.reset())

            if record:
                ob_orig = _render()


def evaluate(env,
             agent_wrapper,
             num_trajs,
             iter_num,
             render,
             model_path):

    # Rebuild the computational graph
    # Create an agent
    agent = agent_wrapper()
    # Create episode generator
    ep_gen = ep_generator(env, agent, render)
    # Initialize and load the previously learned weights into the freshly re-built graph

    # Load the model
    agent.load(model_path, iter_num)
    logger.info("model loaded from path:\n  {}".format(model_path))

    # Initialize the history data structures
    ep_lens = []
    ep_env_rets = []
    # Collect trajectories
    for i in range(num_trajs):
        logger.info("evaluating [{}/{}]".format(i + 1, num_trajs))
        traj = ep_gen.__next__()
        ep_len, ep_env_ret = traj['ep_len'], traj['ep_env_ret']
        # Aggregate to the history data structures
        ep_lens.append(ep_len)
        ep_env_rets.append(ep_env_ret)
    # Log some statistics of the collected trajectories
    ep_len_mean = np.mean(ep_lens)
    ep_env_ret_mean = np.mean(ep_env_rets)
    logger.record_tabular("ep_len_mean", ep_len_mean)
    logger.record_tabular("ep_env_ret_mean", ep_env_ret_mean)
    logger.dump_tabular()


def learn(args,
          rank,
          world_size,
          env,
          eval_env,
          agent_wrapper,
          experiment_name):

    # Create an agent
    agent = agent_wrapper()

    # Create context manager that records the time taken by encapsulated ops
    timed = timed_cm_wrapper(logger)

    num_iters = args.num_timesteps // args.rollout_len
    iters_so_far = 0
    timesteps_so_far = 0
    tstart = time.time()

    # Create dictionary to collect stats
    d = defaultdict(list)

    # Set up model save directory
    if rank == 0:
        ckpt_dir = osp.join(args.checkpoint_dir, experiment_name)
        os.makedirs(ckpt_dir, exist_ok=True)

    # Setup wandb
    if rank == 0:
        while True:
            try:
                wandb.init(project=args.wandb_project,
                           name=experiment_name,
                           group='.'.join(experiment_name.split('.')[:-2]),
                           job_type=experiment_name.split('.')[-2],
                           config=args.__dict__)
            except ConnectionRefusedError:
                pause = 5
                logger.info("[WARN] wandb co error. Retrying in {} secs.".format(pause))
                time.sleep(pause)
            else:
                logger.info("[WARN] wandb co established!")
                break

    # Create rollout generator for training the agent
    roll_gen = rollout_generator(env, agent, args.rollout_len)
    if eval_env is not None:
        assert rank == 0, "non-zero rank mpi worker forbidden here"
        # Create episode generator for evaluating the agent
        eval_ep_gen = ep_generator(eval_env, agent, args.render, args.record)

    while iters_so_far <= num_iters:

        log_iter_info(logger, iters_so_far, num_iters, tstart)

        # if iters_so_far % 20 == 0:
        #     # Check if the mpi workers are still synced
        #     sync_check(agent.policy)
        #     if agent.hps.algo == 'gail':
        #         sync_check(agent.discriminator)

        if rank == 0 and iters_so_far % args.save_frequency == 0:
            # Save the model
            agent.save(ckpt_dir, iters_so_far)
            logger.info("saving model @: {}".format(ckpt_dir))

        # Sample mini-batch in env w/ perturbed actor and store transitions
        with timed("interacting"):
            rollout = roll_gen.__next__()
            logger.info("[INFO] {} ".format("timesteps".ljust(20, '.')) +
                        "{}".format(timesteps_so_far + args.rollout_len))

            # Log interaction stats
            roll_len = mpi_mean_reduce(rollout['ep_lens'])
            roll_env_ret = mpi_mean_reduce(rollout['ep_env_rets'])
            logger.record_tabular('roll_len', roll_len)
            logger.record_tabular('roll_env_ret', roll_env_ret)
            if agent.hps.algo == 'gail':
                roll_syn_ret = mpi_mean_reduce(rollout['ep_syn_rets'])
                logger.record_tabular('roll_syn_ret', roll_syn_ret)
            logger.info("[CSV] dumping roll stats in .csv file")
            logger.dump_tabular()

        with timed("training"):
            # Train the policy and value
            losses, lrnow = agent.train(
                rollout=rollout,
                iters_so_far=iters_so_far,
            )
            # Store the losses and gradients in their respective deques
            d['pol_losses'].append(losses['pol'])
            d['val_losses'].append(losses['val'])
            if agent.hps.algo == 'gail':
                d['dis_losses'].append(losses['dis'])

        if eval_env is not None:
            assert rank == 0, "non-zero rank mpi worker forbidden here"

            if iters_so_far % args.eval_frequency == 0:

                with timed("evaluating"):

                    for eval_step in range(args.eval_steps_per_iter):
                        # Sample an episode w/ non-perturbed actor w/o storing anything
                        eval_ep = eval_ep_gen.__next__()
                        # Aggregate data collected during the evaluation to the buffers
                        d['eval_ep_len'].append(eval_ep['ep_len'])
                        d['eval_ep_env_ret'].append(eval_ep['ep_env_ret'])

                    # Log evaluation stats
                    eval_len = np.mean(d['eval_ep_len'])
                    eval_env_ret = np.mean(d['eval_ep_env_ret'])
                    logger.record_tabular('eval_len', eval_len)
                    logger.record_tabular('eval_env_ret', eval_env_ret)
                    logger.info("[CSV] dumping eval stats in .csv file")
                    logger.dump_tabular()

                    if args.record:
                        # Record the last episode in a video
                        frames = np.split(eval_ep['obs_render'], 1, axis=-1)
                        frames = np.concatenate(np.array(frames), axis=0)
                        frames = np.array([np.squeeze(a, axis=0)
                                           for a in np.split(frames, frames.shape[0], axis=0)])
                        frames = np.transpose(frames, (0, 3, 1, 2))  # from nwhc to ncwh

                        wandb.log({'video': wandb.Video(frames.astype(np.uint8),
                                                        fps=25,
                                                        format='gif',
                                                        caption="Evaluation (last episode)")},
                                  step=timesteps_so_far)

        # Log stats in dashboard
        if rank == 0:

            wandb.log({"num_workers": np.array(world_size)})

            wandb.log({'roll_len': roll_len,
                       'roll_env_ret': roll_env_ret},
                      step=timesteps_so_far)
            if agent.hps.algo == 'gail':
                wandb.log({'roll_syn_ret': roll_syn_ret},
                          step=timesteps_so_far)

            wandb.log({'pol_loss': np.mean(d['pol_losses']),
                       'val_loss': np.mean(d['val_losses']),
                       'lrnow': np.array(lrnow)},
                      step=timesteps_so_far)
            if agent.hps.algo == 'gail':
                wandb.log({'dis_loss': np.mean(d['dis_losses'])},
                          step=timesteps_so_far)

            if iters_so_far % args.eval_frequency == 0:
                wandb.log({'eval_len': eval_len,
                           'eval_env_ret': eval_env_ret},
                          step=timesteps_so_far)

        # Increment counters
        iters_so_far += 1
        timesteps_so_far += args.rollout_len
        # Clear the iteration's running stats
        d.clear()
