#python -m rl.main --log_root_dir ./logs --wandb True --prefix hl.sst.curr_sub.v2 --max_global_step 6000000 --hrl True --ll_type mp --planner_type sst --planner_objective state_const_integral --range 1.0 --threshold 0.5 --timelimit 0.5 --env simple-reacher-obstacle-v0  --hl_type subgoal --gpu 1 --rl_hid_size 256 --meta_update_target both --hrl_network_to_update HL --max_episode_step 150 --evaluate_interval 1 --meta_tanh_policy True  --meta_subgoal_rew -0.5 --max_meta_len 30 --max_grad_norm 0.5
#python -m rl.main --log_root_dir ./logs --wandb True --prefix hl.sst.curr_sub.v1 --max_global_step 6000000 --hrl True --ll_type mp --planner_type sst --planner_objective state_const_integral --range 1.0 --threshold 0.5 --timelimit 0.5 --env simple-reacher-obstacle-v0  --hl_type subgoal --gpu 2 --rl_hid_size 256 --meta_update_target both --hrl_network_to_update HL --max_episode_step 150 --evaluate_interval 1 --meta_tanh_policy True  --meta_subgoal_rew -0.5 --max_meta_len 50 --max_grad_norm 0.5
mpiexec -n 20 python -m rl.main --log_root_dir ./logs --wandb True --prefix hl.sst.v3 --max_global_step 6000000 --hrl True --ll_type mp --planner_type sst --planner_objective state_const_integral --range 1.0 --threshold 0.5 --timelimit 0.5 --env simple-reacher-obstacle-v0  --hl_type subgoal --gpu 3 --rl_hid_size 512 --meta_update_target both --hrl_network_to_update HL --max_episode_step 150 --evaluate_interval 1 --meta_tanh_policy True  --meta_subgoal_rew -1 --max_meta_len 30
