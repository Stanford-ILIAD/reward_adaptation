# Transfer Reinforcement Learning Across Homotopy Classes

## Introduction
_Authors_: [Zhangjie Cao](https://caozhangjie.github.io/), [Minae Kwon](https://stanford.edu/~mnkwon/), [Dorsa Sadigh](https://dorsa.fyi/)

Source code accompanying our paper. Also see our project web page.

## Requirements
This code requires Python3. We use version 3.7. The Python3 requirements are specified in `requirements.txt`. We recommend creating using a virtual environment and then installing requirements as follows,
```
pip install -r requirements.txt
```

## Training

### Navigation1: Barrier Sizes (Table 1)
* Install the env
  * `cd driving`
  * `pip install -e .`

* When running these scripts, please specify a barrier size [1,3,5,7].
* Ours:
  * run `python train_nav1.py --expt_type ours --bs [1,3,5,7]`
* Fine tune:
  * run `python train_nav1.py --expt_type finetune --bs [1,3,5,7]`
* Random:
  * run `python train_nav1.py --expt_type direct --bs [1,3,5,7]`
* PNN:
  * run `python baselines/PNN/train.py --bs [1,3,5,7]`  
* L2SP:
  * run `python baselines/L2SP/train.py --bs [1,3,5,7]`  
* BSS:
  * run `python baselines/BSS/train.py --bs [1,3,5,7]`  

### Navigation2: Four Homotopy Classes (Table 2)
* Install the env
  * `cd driving`
  * `pip install -e .`

* Ours:
  * run `./train_nav2.sh`

### Lunar Lander (Table 3)
* Install the env
  * `cd lunarlander`
  * `pip install -e .`

* Ours:
  * run `./train_lunarlander.sh`

### Fetch Reach (Table 4)

* When running our method, you will want to change the barrier penalties as part of the curriculum. Make sure that the correct barrier penalty is specified on line 114-117 of `fetch/fetch_envs/fetch_env.py` The exact curriculum we used is detailed in the supplementary materials
* Please also specify whether you are fine-tuning from Right to Left (RL) or from Left to Right (LR).
* Ours:
  * run `python train_fetch.py --expt_type ours --bs [RL, LR]`
* Fine tune:
  * run `python train_fetch.py --expt_type finetune --bs [RL, LR]`
* Random:
  * run `python train_fetch.py  --expt_type direct --bs [RL, LR]`
* PNN:
  * run `python baselines_fetch/PNN/train.py --bs [RL, LR]`  
* L2SP:
  * run `python baselines_fetch/L2SP/train.py --bs [RL, LR]`  
* BSS:
  * run `python baselines_fetch/BSS/train.py --bs [RL, LR]`  

### Ant (Table 5)
* Install the env
  * `cd ant/imperfect_envs`
  * `pip install -e .`

* Ours:
  * `cd ant/transfer`
  * run `curriculum.sh`

### Assistive Gym (Table 5)
Please refer to the README in the `assistive-gym` directory for installation and training instructions.
