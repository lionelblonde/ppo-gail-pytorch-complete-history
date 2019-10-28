#!/usr/bin/env bash
# Example: ./minigrid_ppo_train.sh <env_id> <num_learners> <seed> <visdom_server> <visdom_port> <visdom_username> <visdom_password>
cd ..

mpiexec -n $2 python main.py \
    --no-cuda \
    --pixels \
    --no-recurrent \
    --extractor='minigrid_cnn' \
    --env_id=$1 \
    --seed=$3 \
    --checkpoint_dir="data/checkpoints" \
    --log_dir="data/logs" \
    --enable_visdom \
    --visdom_dir="data/summaries" \
    --visdom_server=$4 \
    --visdom_port=$5 \
    --visdom_username=$6 \
    --visdom_password=$7 \
    --task="train" \
    --algo="ppo" \
    --save_frequency=100 \
    --num_iters=10000000 \
    --optim_epochs_per_iter=10 \
    --eval_steps_per_iter=1 \
    --eval_frequency=1 \
    --no-render \
    --rollout_len=2048 \
    --batch_size=128 \
    --with_layernorm \
    --p_lr=3e-4 \
    --no-with_scheduler \
    --clip_norm=.5 \
    --gamma=0.99 \
    --gae_lambda=0.95 \
    --eps=0.2 \
    --p_ent_reg_scale=0.
