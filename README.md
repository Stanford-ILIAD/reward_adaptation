## Overview 
* `driving` contains the driving environment. The specific Merging Environments can be found in 
`driving/driving_envs/envs/merging_env.py`. This defines all the rewards for different curricula.
* `reward_curriculum_expts/reward_curriculum.py` contains the code for training the reward curriculum.
This file also contains code for directly training on a single domain.
* `trainer.py` contains the main training code that is commonly used across different experiments for training.
* `utils.py` contains helper code, including the `evaluate` function when training.

## Training reward curriculum
* Specify reward in `merging_env.py` by changing `weight` arguments. For example, for the curriculum 
[-0.5, 0, 0.5, 1.0], I would specify those weights for MergingEnv2, MergingEnv3, .., MergingEnv5. 
* In `reward_curriculum.py`, comment out MergingEnvironments that you do not use where `self.curriculum` is defined 
* To train, run `python reward_curriculum_expts/reward_curriculum.py`
    * To train a curriculum, choose the `train_curriculum` function. The pre-trained model is specified at the bottom of the file.
    * To train on a single domain, choose the `train_single` function. 
 