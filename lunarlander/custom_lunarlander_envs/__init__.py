from gym.envs.registration import register

register(id='HomotopyLunarLanderL-v0', entry_point='custom_lunarlander_envs.envs:LunarLanderLEnv')
register(id='HomotopyLunarLanderR-v0', entry_point='custom_lunarlander_envs.envs:LunarLanderREnv')
register(id='HomotopyLunarLanderLObstacle-v0', entry_point='custom_lunarlander_envs.envs:LunarLanderLObstacleEnv')
register(id='HomotopyLunarLanderRObstacle-v0', entry_point='custom_lunarlander_envs.envs:LunarLanderRObstacleEnv')
register(id='HomotopyLunarLanderObstacleSuccessor-v0', entry_point='custom_lunarlander_envs.envs:LunarLanderObstacleSuccessor')
