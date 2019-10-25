import time
import copy
import os
from collections import namedtuple, deque, OrderedDict

import yaml
import cv2
import visdom
import numpy as np
from gym import spaces

from helpers import logger
from helpers.distributed_util import sync_check
from agents.memory import RingBuffer
from helpers.console_util import (timed_cm_wrapper, pretty_iter,
                                  pretty_elapsed, columnize)


def rollout_generator(env, agent, rollout_len):

    is_disc = isinstance(env.action_space, spaces.Discrete)
    pixels = len(env.observation_space.shape) == 3

    t = 0
    done = True
    env_rew = 0.0
    ob = env.reset()
    if pixels:
        ob = np.array(ob)
    if agent.is_recurrent:
        state = np.zeros((agent.hps.hidden_state_size * 2,))  # LSTM's h and c

    obs = RingBuffer(rollout_len, shape=agent.ob_shape)
    acs = RingBuffer(rollout_len, shape=(1,) if is_disc else agent.ac_shape)
    vs = RingBuffer(rollout_len, shape=(1,), dtype='float32')
    logps = RingBuffer(rollout_len, shape=(1,), dtype='float32')
    if hasattr(agent, 'discriminator'):
        syn_rews = RingBuffer(rollout_len, shape=(1,), dtype='float32')
    env_rews = RingBuffer(rollout_len, shape=(1,), dtype='float32')
    dones = RingBuffer(rollout_len, shape=(1,), dtype='int32')
    if agent.is_recurrent:
        states = RingBuffer(rollout_len, shape=(agent.hps.hidden_state_size * 2,))  # LSTM's h and c

    while True:
        if agent.is_recurrent:
            ac, v, logp, state = agent.predict_recurrent(ob, done, state, sample_or_mode=True)
        else:
            ac, v, logp = agent.predict(ob, sample_or_mode=True)

        # if t <= 100:
        #     print(">>>>>>>>>> Acting randomly in env to init the normalizers.")
        #     ac = env.action_space.sample()
        # XXX GYM BUG. LEAVE HERE. `env.action_space.sample()` NON-DETERMINISTIC FUNCTION.

        if not is_disc:
            # NaN-proof and clip
            ac = np.nan_to_num(ac)
            ac = np.clip(ac, env.action_space.low, env.action_space.high)
        else:
            ac = ac if isinstance(ac, int) else np.asscalar(ac)

        if t > 0 and t % rollout_len == 0:

            obs_ = obs.data.reshape(-1, *agent.ob_shape)

            if not pixels:
                agent.rms_obs.update(obs_)

            out = {"obs": obs_,
                   "acs": acs.data.reshape(-1, *((1,) if is_disc else agent.ac_shape)),
                   "vs": vs.data.reshape(-1, 1),
                   "logps": logps.data.reshape(-1, 1),
                   "env_rews": env_rews.data.reshape(-1, 1),
                   "dones": dones.data.reshape(-1, 1),
                   "next_v": v * (1 - done)}
            if agent.is_recurrent:
                out.update({"states": states.data})
            if hasattr(agent, 'discriminator'):
                out.update({"syn_rews": syn_rews.data})

            yield out

        obs.append(ob)
        acs.append(ac)
        vs.append(v)
        logps.append(logp)
        dones.append(done)
        if agent.is_recurrent:
            states.append(state)

        # Interact with env(s)
        new_ob, env_rew, done, _ = env.step(ac)

        env_rews.append(env_rew)

        if hasattr(agent, 'discriminator'):
            syn_rew = np.asscalar(agent.discriminator.get_reward(ob, ac).cpu().numpy().flatten())
            syn_rews.append(syn_rew)

        ob = copy.copy(new_ob)
        if pixels:
            ob = np.array(ob)

        if done:
            ob = np.array(env.reset())

        t += 1


def ep_generator(env, agent, render):
    """Generator that spits out a trajectory collected during a single episode
    `append` operation is also significantly faster on lists than numpy arrays,
    they will be converted to numpy arrays once complete and ready to be yielded.
    """

    is_disc = isinstance(env.action_space, spaces.Discrete)

    ob = np.array(env.reset())
    done = True
    if agent.is_recurrent:
        state = np.zeros((agent.hps.hidden_state_size * 2,))
    cur_ep_len = 0
    cur_ep_env_ret = 0
    obs = []
    if agent.is_recurrent:
        states = []
    acs = []
    vs = []
    env_rews = []

    while True:
        if agent.is_recurrent:
            ac, v, _, state = agent.predict_recurrent(ob, done, state, sample_or_mode=True)
        else:
            ac, v, _ = agent.predict(ob, sample_or_mode=True)

        if not is_disc:
            # NaN-proof and clip
            ac = np.nan_to_num(ac)
            ac = np.clip(ac, env.action_space.low, env.action_space.high)
        else:
            ac = ac if isinstance(ac, int) else np.asscalar(ac)

        obs.append(ob)
        if agent.is_recurrent:
            states.append(state)
        acs.append(ac)
        vs.append(v)
        new_ob, env_rew, done, _ = env.step(ac)

        if render:
            env.render()

        env_rews.append(env_rew)
        cur_ep_len += 1
        cur_ep_env_ret += env_rew
        ob = np.array(copy.copy(new_ob))
        if done:
            obs = np.array(obs)
            acs = np.array(acs)
            env_rews = np.array(env_rews)
            out = {"obs": obs,
                   "acs": acs,
                   "vs": vs,
                   "env_rews": env_rews,
                   "ep_len": cur_ep_len,
                   "ep_env_ret": cur_ep_env_ret}
            if agent.is_recurrent:
                out.update({"states": states})

            yield out

            cur_ep_len = 0
            cur_ep_env_ret = 0
            obs = []
            if agent.is_recurrent:
                states = []
            acs = []
            env_rews = []
            ob = np.array(env.reset())


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
          experiment_name,
          ckpt_dir,
          save_frequency,
          enable_visdom,
          visdom_dir,
          visdom_server,
          visdom_port,
          visdom_username,
          visdom_password,
          rollout_len,
          eval_steps_per_iter,
          eval_frequency,
          render,
          expert_dataset,
          max_iters):

    # Create an agent
    agent = agent_wrapper()

    # Create context manager that records the time taken by encapsulated ops
    timed = timed_cm_wrapper(logger)

    # Create rollout generator for training the agent
    roll_gen = rollout_generator(env, agent, rollout_len)
    if eval_env is not None:
        assert rank == 0, "non-zero rank mpi worker forbidden here"
        # Create episode generator for evaluating the agent
        eval_ep_gen = ep_generator(eval_env, agent, render)

    iters_so_far = 0
    tstart = time.time()

    # Define rolling buffers for experiental data collection
    maxlen = 100
    keys = ['ac', 'v', 'policy_losses', 'value_losses', 'gradnorm']
    if hasattr(agent, 'discriminator'):
        keys.extend(['d_losses'])
    if eval_env is not None:
        assert rank == 0, "non-zero rank mpi worker forbidden here"
        keys.extend(['eval_ac', 'eval_v', 'eval_len', 'eval_env_ret'])
    Deques = namedtuple('Deques', keys)
    deques = Deques(**{k: deque(maxlen=maxlen) for k in keys})

    # Set up model save directory
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)

    # Setup Visdom
    if rank == 0 and enable_visdom:

        # Setup the Visdom directory
        os.makedirs(visdom_dir, exist_ok=True)

        # Create visdom
        viz = visdom.Visdom(env="job_{}_{}".format(int(time.time()), experiment_name),
                            log_to_filename=os.path.join(visdom_dir, "vizlog.txt"),
                            server=visdom_server,
                            port=visdom_port,
                            username=visdom_username,
                            password=visdom_password)
        assert viz.check_connection(timeout_seconds=4), "viz co not great"

        viz.text("World size: {}".format(world_size))
        iter_win = viz.text("will be overridden soon")
        viz.text(yaml.safe_dump(args.__dict__, default_flow_style=False))

        keys = ['eval_len', 'eval_env_ret', 'eval_frames']
        keys.extend(['policy_loss', 'value_loss'])
        if hasattr(agent, 'discriminator'):
            keys.append('d_loss')

        # Create (empty) visdom windows
        VizWins = namedtuple('VizWins', keys)
        vizwins = VizWins(**{k: viz.line(X=[0], Y=[np.nan]) for k in keys})
        # HAXX: NaNs ignored by visdom

    while iters_so_far <= max_iters:

        pretty_iter(logger, iters_so_far)
        pretty_elapsed(logger, tstart)

        if iters_so_far % 20 == 0:
            # Check if the mpi workers are still synced
            sync_check(agent.policy)

        if rank == 0 and iters_so_far % save_frequency == 0:
            # Save the model
            agent.save(ckpt_dir, iters_so_far)
            logger.info("saving model:\n  @: {}".format(ckpt_dir))

        # Sample mini-batch in env w/ perturbed actor and store transitions
        with timed("interacting"):
            rollout = roll_gen.__next__()

        # Extend deques with collected experiential data
        deques.ac.extend(rollout['acs'])
        deques.v.extend(rollout['vs'])

        with timed("training"):

            train = agent.train_recurrent if args.hidden_state_size is not None else agent.train

            # Train the policy and value
            losses, gradnorm = train(rollout, timesteps_so_far=iters_so_far * rollout_len)

            # Store the losses and gradients in their respective deques
            deques.policy_losses.append(losses['policy'])
            deques.value_losses.append(losses['value'])
            deques.gradnorm.append(gradnorm)
            if hasattr(agent, 'discriminator'):
                deques.d_losses.append(losses['discriminator'])

        if eval_env is not None:
            assert rank == 0, "non-zero rank mpi worker forbidden here"

            if iters_so_far % eval_frequency == 0:

                with timed("evaluating"):

                    # Use the running stats of the training environment to normalize
                    if hasattr(eval_env, 'running_moments'):
                        eval_env.running_moments = copy.deepcopy(env.running_moments)

                    for eval_step in range(eval_steps_per_iter):

                        # Sample an episode w/ non-perturbed actor w/o storing anything
                        eval_ep = eval_ep_gen.__next__()

                        # Aggregate data collected during the evaluation to the buffers
                        deques.eval_ac.extend(eval_ep['acs'])
                        deques.eval_v.extend(eval_ep['vs'])
                        deques.eval_len.append(eval_ep['ep_len'])
                        deques.eval_env_ret.append(eval_ep['ep_env_ret'])

                    # When using pixels, create a video with the last eval loop
                    if hasattr(env, 'grayscale') and args.verbose_eval:
                        # Unstack the frames if stacked, while leaving colors unaltered
                        frames = np.split(eval_ep['obs'],
                                          env.k if hasattr(env, 'k') else 1,
                                          axis=-1)
                        frames = np.concatenate(np.array(frames), axis=0)
                        frames = [np.squeeze(a, axis=0)
                                  for a in np.split(frames, frames.shape[0], axis=0)]
                        # Create OpenCV video writer
                        writer = cv2.VideoWriter(filename='__eval__.mp4',
                                                 fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                                                 fps=25,
                                                 frameSize=(env.width, env.height),
                                                 isColor=(False
                                                          if (hasattr(env, 'grayscale') and
                                                              env.grayscale)
                                                          else True))
                        for frame in frames:
                            # Add frame to video
                            writer.write(frame)
                        writer.release()
                        cv2.destroyAllWindows()

        # Log statistics

        logger.info("logging misc training stats")

        stats = OrderedDict()
        # Add min, max and mean of the components of the average action
        ac_np_mean = np.mean(deques.ac, axis=0)  # vector
        stats.update({'min_ac_comp': np.amin(ac_np_mean)})
        stats.update({'max_ac_comp': np.amax(ac_np_mean)})
        stats.update({'mean_ac_comp': np.mean(ac_np_mean)})
        # Add values mean and std
        stats.update({'v_value': np.mean(deques.v)})
        stats.update({'v_deviation': np.std(deques.v)})
        # Add gradient norms
        stats.update({'gradnorm': np.mean(deques.gradnorm)})
        # Add losses
        stats.update({'p_loss': np.mean(deques.policy_losses)})
        stats.update({'v_loss': np.mean(deques.value_losses)})
        if hasattr(agent, 'discriminator'):
            stats.update({'d_loss': np.mean(deques.d_losses)})

        # Log dictionary content
        logger.info(columnize(['name', 'value'], stats.items(), [24, 16]))

        if eval_env is not None:
            assert rank == 0, "non-zero rank mpi worker forbidden here"

            if iters_so_far % eval_frequency == 0:

                # Use the logger object to log the eval stats (will appear in `progress{}.csv`)
                logger.info("logging misc eval stats")
                # Add min, max and mean of the components of the average action
                ac_np_mean = np.mean(deques.eval_ac, axis=0)  # vector
                logger.record_tabular('min_ac_comp', np.amin(ac_np_mean))
                logger.record_tabular('max_ac_comp', np.amax(ac_np_mean))
                logger.record_tabular('mean_ac_comp', np.mean(ac_np_mean))
                # Add V values mean and std
                logger.record_tabular('v_value', np.mean(deques.eval_v))
                logger.record_tabular('v_deviation', np.std(deques.eval_v))
                # Add episodic stats
                logger.record_tabular('ep_len', np.mean(deques.eval_len))
                logger.record_tabular('ep_env_ret', np.mean(deques.eval_env_ret))
                logger.dump_tabular()

        # Mark the end of the iter in the logs
        logger.info('')

        iters_so_far += 1

        if rank == 0 and enable_visdom:

            viz.text("Current iter: {}".format(iters_so_far), win=iter_win, append=False)

            if iters_so_far % eval_frequency == 0:

                viz.line(X=[iters_so_far],
                         Y=[np.mean(deques.eval_len)],
                         win=vizwins.eval_len,
                         update='append',
                         opts=dict(title='Eval Episode Length'))

                viz.line(X=[iters_so_far],
                         Y=[np.mean(deques.eval_env_ret)],
                         win=vizwins.eval_env_ret,
                         update='append',
                         opts=dict(title='Eval Episodic Return'))

                # When using pixels, send eval frames
                if hasattr(env, 'grayscale') and args.verbose_eval:
                    viz.images(np.array(frames).transpose(0, 3, 1, 2),
                               win=vizwins.eval_frames,
                               opts=dict(title='Last Eval Episode'))

            viz.line(X=[iters_so_far],
                     Y=[np.mean(deques.policy_losses)],
                     win=vizwins.policy_loss,
                     update='append',
                     opts=dict(title="Policy Loss"))

            viz.line(X=[iters_so_far],
                     Y=[np.mean(deques.value_losses)],
                     win=vizwins.value_loss,
                     update='append',
                     opts=dict(title="Value Loss"))

            if hasattr(agent, 'discriminator'):
                viz.line(X=[iters_so_far],
                         Y=[np.mean(deques.d_losses)],
                         win=vizwins.d_loss,
                         update='append',
                         opts=dict(title="D Loss"))
