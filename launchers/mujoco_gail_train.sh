#!/usr/bin/env bash
# Example: ./mujoco_gail_train.sh <env_id> <num_learners> <expert_path> <num_demos> <seed> <visdom_server> <visdom_port> <visdom_username> <visdom_password>
cd ..

mpiexec -n $2 python main.py \
    --no-cuda \
    --no-pixels \
    --no-recurrent \
    --feat_x_p='shallow_mlp' \
    --feat_x_v='shallow_mlp' \
    --env_id=$1 \
    --seed=$5 \
    --checkpoint_dir="data/checkpoints" \
    --log_dir="data/logs" \
    --enable_visdom \
    --visdom_dir="data/summaries" \
    --visdom_server=$6 \
    --visdom_port=$7 \
    --visdom_username=$8 \
    --visdom_password=$9 \
    --task="train" \
    --algo="gail" \
    --save_frequency=100 \
    --num_iters=10000000 \
    --optim_epochs_per_iter=2 \
    --eval_steps_per_iter=10 \
    --eval_frequency=1 \
    --no-render \
    --rollout_len=2048 \
    --batch_size=128 \
    --with_layernorm \
    --d_update_ratio=1 \
    --p_lr=3e-4 \
    --no-with_scheduler \
    --d_lr=3e-4 \
    --clip_norm=.5 \
    --no-state_only \
    --minimax_only \
    --gamma=0.99 \
    --gae_lambda=0.95 \
    --eps=0.2 \
    --p_ent_reg_scale=0. \
    --d_ent_reg_scale=0. \
    --expert_path=$3 \
    --num_demos=$4
