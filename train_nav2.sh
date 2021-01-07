## relaxing, set --experiment_name to MO_{source task name}_{target task name}_None, --source_env {source task name}, --target_env ContinuousNone{target task name}-v0
python train_nav2.py --experiment_dir output --experiment_name MO_LL2RL_None --source_env LL --target_env ContinuousNoneRL-v0

## compute relaxing iterations number, with the only parameter as the --experiment_name in the previous command
python compute_iterations_relax.py MO_LL2RL_None

## obstacle size, set --experiment_name to MO_{source task name}_{target task name}_barrier_set_size, --source_env {source task name}, --target_env ContinuousAdjust{target task name}-v0
python train_nav2.py --experiment_dir output --experiment_name MO_LL2LR_barrier_set_size --source_env LL --target_env ContinuousAdjustLR-v0

## reward curriculum, set --experiment_name to MO_{source task name}_{target task name}_reward_w, --source_env {source task name}, --target_env ContinuousAdjust{target task name}-v0
python train_nav2.py --experiment_dir output --experiment_name MO_LL2LR_reward_w --source_env LL --target_env ContinuousAdjustRLR-v0

##compute curriculum iterations number, with the only parameter as the --experiment_name in the previous two commands
python compute_iterations.py output/MO_LL2LR_barrier_set_size
python compute_iterations.py output/MO_LL2LR_reward_w
