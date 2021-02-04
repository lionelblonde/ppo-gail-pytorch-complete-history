# PyTorch implementations of PPO and GAIL

The repository covers the following algorithms:
* **PPO** (Reinforcement Learning)
    * MLP and CNN policies
    * Gaussian policies (continuous actions) and categorical policies (discrete actions)
* **GAIL** (Imitation Learning)
    * MLP policies
    * Gaussian policies (continuous actions)

Benchmarks and environments covered:
* MuJoCo
* DeepMind Control Suite
* Safety Gym
* Atari
* Pycolab

## Dependencies

TODO

## Running Experiments
While one can launch any job via `main.py`, it is advised to use `spawner.py`,
designed to spawn a swarm of experiments over multiple seeds and environments in one command.
To get its usage description, type `python spawner.py -h`.
```bash
usage: spawner.py [-h] [--config CONFIG] [--conda_env CONDA_ENV]
                  [--env_bundle ENV_BUNDLE] [--num_workers NUM_WORKERS]
                  [--deployment {tmux,slurm}] [--num_seeds NUM_SEEDS]
                  [--caliber {short,long,verylong,veryverylong}]
                  [--deploy_now] [--no-deploy_now] [--sweep] [--no-sweep]
                  [--wandb_upgrade] [--no-wandb_upgrade]
                  [--num_demos NUM_DEMOS [NUM_DEMOS ...]] [--debug]
                  [--no-debug] [--wandb_dryrun] [--no-wandb_dryrun]
                  [--debug_lvl DEBUG_LVL]

Job Spawner

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG
  --conda_env CONDA_ENV
  --env_bundle ENV_BUNDLE
  --num_workers NUM_WORKERS
  --deployment {tmux,slurm}
                        deploy how?
  --num_seeds NUM_SEEDS
  --caliber {short,long,verylong,veryverylong}
  --deploy_now          deploy immediately?
  --no-deploy_now
  --sweep               hp search?
  --no-sweep
  --wandb_upgrade       upgrade wandb?
  --no-wandb_upgrade
  --num_demos NUM_DEMOS [NUM_DEMOS ...], --list NUM_DEMOS [NUM_DEMOS ...]
  --debug               toggle debug/verbose mode in spawner
  --no-debug
  --wandb_dryrun        toggle wandb offline mode
  --no-wandb_dryrun
  --debug_lvl DEBUG_LVL
                        set the debug level for the spawned runs
```

Here is an example:
```bash
python spawner.py --config tasks/train_mujoco_ppo.yaml --env_bundle debug --wandb_upgrade --no-sweep --deploy_now --caliber short --num_workers 2 --num_seeds 3 --deployment tmux --conda_env pytorch --wandb_dryrun --debug_lvl 2
```
Check the argument parser in `spawner.py` to know what each of these arguments mean.
