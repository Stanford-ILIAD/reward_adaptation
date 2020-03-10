# Import the envs module so that envs register themselves
#import tightrope.gym_tightrope.envs
from gym.envs.registration import register

# Import wrappers so it's accessible when installing with pip
#import tightrope.gym_tightrope.wrappers
register(id="Tightrope-v0", entry_point="gym_tightrope.envs:Tightrope")
