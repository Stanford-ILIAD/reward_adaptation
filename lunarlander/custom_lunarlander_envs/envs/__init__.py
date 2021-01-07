try:
    import Box2D
    from custom_lunarlander_envs.envs.custom_lunarlander import LunarLanderLEnv, LunarLanderREnv, LunarLanderLObstacleEnv, LunarLanderRObstacleEnv, LunarLanderObstacleSuccessor
except ImportError:
    Box2D = None
