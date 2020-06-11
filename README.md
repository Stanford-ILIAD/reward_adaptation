# Overview 

### Environments
* `driving` contains the navigation environment. 
* `fetch` contains the fetch reach environment.
* `toy` contains the code for generating the surface plots. 

### Training
* `train.py` contains code to train single models as well as curriculums. 

### Output
* `output` contains all output from our training. Currently stores policies from driving and gridworld environments.

### Experiments
* `experiments` contains environment-specific experiments. 

### Evaluation

* `eval_model.py` contains code to evaluate trained models.  

### Requirements

* `pip install -r requirements.txt`  

# Fetch Reach 

### Reproducing Results (Table 3)

* Before running each experiment, make sure that the homotopy class you are fine-tuning to is correctly specified in line 49 of `fetch/fetch_envs/robot_env.py`
* When running our method, you will want to change the barrier penalties as part of the curriculum. Make sure that the correct barrier penalty is specifeid on line 111 of  `fetch/fetch_envs/fetch_env.py` The exact curriculum we used is detailed in the supplementary materials
* Ours:
  * specify a source model on line 195 in `train.py`
  * run `python train.py --env fetch --expt_type ours`
* Fine tune:
  * specify a source model on line 197 in `train.py`
  * run `python train.py --env fetch --expt_type finetune`
* Random:
  * run `python train.py --env fetch --expt_type direct`
* PNN:
  * specify a source model on line 116 in `baselines_fetch/PNN/train.py`
  * run `python baselines_fetch/PNN/train.py`  
* L2SP:
  * specify a source model on line 123 in `baselines_fetch/L2SP/train.py`
  * run `python baselines_fetch/L2SP/train.py`  
* BSS:
  * specify a source model on line 121 in `baselines_fetch/BSS/train.py`
  * run `python baselines_fetch/BSS/train.py`  

### Pre-trained Models  & Evaluation 

* A list of pre-trained models and their directories are located in `output/fetch2/policies.py` 
* To evaluate saved models, 
  * make sure that the correct homotopy class and barrier penalty are specified
  * specify a desired model to run from `output/fetch2/policies.py` on line 109
  * if running L2SP or PNN, specify the baseline name on line 113; e.g.,  `load_model(model_dir, "HER", baseline='L2SP')`
  * run `python eval_model.py --env fetch`

# Navigation1: Barrier Sizes

### Reproducing Results (Table 1) 

* Uncomment the set of flags (lines 26-33) for the navigation experiment in `train.py`
* Before running each experiment, (1) make sure that the barrier size (1,3,5,7) is correctly specified on line 72 and (2) make sure the correct homotopy class (`right` or `left`) you want to fine-tune to is specified on line 71 of `driving/driving_envs/envs/gridworld_continuous.py`
* Ours:
  * specify a source model on line 179 in `train.py`
  * run `python train.py --env nav1 --expt_type ours`
* Fine tune:
  * specify a source model on line 181 in `train.py`
  * run `python train.py --env nav1 --expt_type finetune`
* Random:
  * run `python train.py --env nav1 --expt_type direct`
* PNN:
  * specify a source model on line 125 in `baselines/PNN/train.py`
  * run `python baselines/PNN/train.py`  
* L2SP:
  * specify a source model on line 126 in `baselines/L2SP/train.py`
  * run `python baselines/L2SP/train.py`  
* BSS:
  * specify a source model on line 128 in `baselines/BSS/train.py`
  * run `python baselines/BSS/train.py`  

