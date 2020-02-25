from eval_model import *
from gridworld_policies.policies import *
from utils import *
from stable_baselines import DQN

source_info = h2v0
source_dir = os.path.join(source_info[0], source_info[1], source_info[2])
target_info = h2v0_h2v1
target_dir = os.path.join(target_info[0], target_info[1], target_info[2])

source = DQN.load(source_dir)
target = DQN.load(target_dir)

param_name = "deepq/model/action_value/fully_connected_2/weights:0"  # CHANGES POLICY!
source_weight = source.get_parameters()[param_name]
target_weight = target.get_parameters()[param_name]

l1_dist = np.abs(source_weight - target_weight)
print(l1_dist)
print(np.sum(l1_dist))

