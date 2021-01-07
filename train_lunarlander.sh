## train policy for each homotopy class
python train_lunarlander.py --experiment_dir output --experiment_name L --target_env HomotopyLunarLanderLObstacle-v0 --seed 128

## check which model converges
python compute_iterations_relax.py output/L 

## relaxing stage
python train_lunarlander.py --experiment_dir output --experiment_name L2R_None --source_env L --target_env HomotopyLunarLanderR-v0 --seed 128

## check which model converges
python compute_iterations_relax.py output/L2R_None

## curriculum learning stage
### reward weight approach
python train_lunarlander.py --experiment_dir output --experiment_name L2R_reward_w --source_env L --target_env HomotopyLunarLanderRObstacle-v0

### check when converges in each step
python compute_iterations.py output/L2R_reward_w

### barrier size set approach
python train_lunarlander.py --experiment_dir output --experiment_name L2R_obs_size --source_env L --target_env HomotopyLunarLanderRObstacle-v0

### check when converges in each step
python compute_iterations.py output/L2R_obs_size
