#!/bin/bash -x
algo=$1
gpu=$2

if [ $algo = 1 ]
then
    algo='ppo'
    rollout_length='1024'
    evaluate_interval="10"
    ckpt_interval='50'
    rl_activation="tanh"
    num_batches="50"
    log_interval="1"
elif [ $algo = 2 ]
then
    algo='sac'
    rollout_length="10000"
    evaluate_interval="1000"
    ckpt_interval='100000'
    rl_activation="relu"
    num_batches="1"
    log_interval="1000"
fi

workers="1"
prefix="05.19.BASELINE.SAWYER.LIFT"
max_global_step="60000000"
env="sawyer-lift-robosuite-v0"
gpu=$gpu
rl_hid_size="256"
max_episode_step="1000"
entropy_loss_coef="1e-3"
buffer_size="1000000"
lr_actor="3e-4"
lr_critic="3e-4"
debug="False"
batch_size="256"
clip_param="0.2"
seed='1235'
ctrl_reward='1e-2'
reward_type='dense'
comment='Baseline'
start_steps='10000'
actor_num_hid_layers='2'
log_root_dir="./logs"
group='05.19.SAWYER.LIFT.BASELINE'
env_debug='False'
log_freq='1000'
reward_scale='10.'
vis_replay="False"
# success_reward='0.'
# has_terminal='True'
# max_grad_norm='0.5'

#mpiexec -n $workers
python -m rl.main \
    --log_root_dir $log_root_dir \
    --wandb True \
    --prefix $prefix \
    --max_global_step $max_global_step \
    --env $env \
    --gpu $gpu \
    --rl_hid_size $rl_hid_size \
    --max_episode_step $max_episode_step \
    --evaluate_interval $evaluate_interval \
    --entropy_loss_coef $entropy_loss_coef \
    --buffer_size $buffer_size \
    --num_batches $num_batches \
    --lr_actor $lr_actor \
    --lr_critic $lr_critic \
    --debug $debug \
    --rollout_length $rollout_length \
    --batch_size $batch_size \
    --clip_param $clip_param \
    --rl_activation $rl_activation \
    --algo $algo \
    --seed $seed \
    --ctrl_reward $ctrl_reward \
    --reward_type $reward_type \
    --comment $comment \
    --start_steps $start_steps \
    --actor_num_hid_layers $actor_num_hid_layers \
    --group $group \
    --env_debug $env_debug \
    --log_freq $log_freq \
    --log_interval $log_interval \
    --reward_scale $reward_scale \
    --vis_replay $vis_replay \
    # --success_reward $success_reward \
    # --has_terminal $has_terminal \