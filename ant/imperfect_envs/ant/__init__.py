from gym.envs.registration import register

register(
    id='CustomAnt-v0',
    entry_point='ant.ant:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='HomotopyDirectUpAnt-v0',
    entry_point='ant.ant:HomotopyDirectUpAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='HomotopyDirectDownAnt-v0',
    entry_point='ant.ant:HomotopyDirectDownAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='HomotopyRelaxUpAnt-v0',
    entry_point='ant.ant:HomotopyRelaxUpAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='HomotopyRelaxDownAnt-v0',
    entry_point='ant.ant:HomotopyRelaxDownAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='HomotopyRewardUpAnt-v0',
    entry_point='ant.ant:HomotopyRewardUpAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='HomotopyRewardDownAnt-v0',
    entry_point='ant.ant:HomotopyRewardDownAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='HomotopyObstacleUpAnt-v0',
    entry_point='ant.ant:HomotopyObstacleUpAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='HomotopyObstacleDownAnt-v0',
    entry_point='ant.ant:HomotopyObstacleDownAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

