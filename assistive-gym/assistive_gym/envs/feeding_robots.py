from .feeding import FeedingEnv, FeedingEnvHomotopyUp, FeedingEnvHomotopyDown, FeedingEnvHomotopyUpNo, FeedingEnvHomotopyDownNo, FeedingEnvHomotopyUpAdjust, FeedingEnvHomotopyDownAdjust

class FeedingPR2Env(FeedingEnv):
    def __init__(self):
        super(FeedingPR2Env, self).__init__(robot_type='pr2', human_control=False)

class FeedingBaxterEnv(FeedingEnv):
    def __init__(self):
        super(FeedingBaxterEnv, self).__init__(robot_type='baxter', human_control=False)

class FeedingSawyerEnv(FeedingEnv):
    def __init__(self):
        super(FeedingSawyerEnv, self).__init__(robot_type='sawyer', human_control=False)

class FeedingJacoEnv(FeedingEnv):
    def __init__(self):
        super(FeedingJacoEnv, self).__init__(robot_type='jaco', human_control=False)

class FeedingPR2HumanEnv(FeedingEnv):
    def __init__(self):
        super(FeedingPR2HumanEnv, self).__init__(robot_type='pr2', human_control=True)

class FeedingBaxterHumanEnv(FeedingEnv):
    def __init__(self):
        super(FeedingBaxterHumanEnv, self).__init__(robot_type='baxter', human_control=True)

class FeedingSawyerHumanEnv(FeedingEnv):
    def __init__(self):
        super(FeedingSawyerHumanEnv, self).__init__(robot_type='sawyer', human_control=True)

class FeedingJacoHumanEnv(FeedingEnv):
    def __init__(self):
        super(FeedingJacoHumanEnv, self).__init__(robot_type='jaco', human_control=True)

class FeedingPR2EnvHomotopyUp(FeedingEnvHomotopyUp):
    def __init__(self):
        super(FeedingPR2EnvHomotopyUp, self).__init__(robot_type='pr2', human_control=False)

class FeedingBaxterEnvHomotopyUp(FeedingEnvHomotopyUp):
    def __init__(self):
        super(FeedingBaxterEnvHomotopyUp, self).__init__(robot_type='baxter', human_control=False)

class FeedingSawyerEnvHomotopyUp(FeedingEnvHomotopyUp):
    def __init__(self):
        super(FeedingSawyerEnvHomotopyUp, self).__init__(robot_type='sawyer', human_control=False)

class FeedingJacoEnvHomotopyUp(FeedingEnvHomotopyUp):
    def __init__(self):
        super(FeedingJacoEnvHomotopyUp, self).__init__(robot_type='jaco', human_control=False)

class FeedingPR2HumanEnvHomotopyUp(FeedingEnvHomotopyUp):
    def __init__(self):
        super(FeedingPR2HumanEnvHomotopyUp, self).__init__(robot_type='pr2', human_control=True)

class FeedingBaxterHumanEnvHomotopyUp(FeedingEnvHomotopyUp):
    def __init__(self):
        super(FeedingBaxterHumanEnvHomotopyUp, self).__init__(robot_type='baxter', human_control=True)

class FeedingSawyerHumanEnvHomotopyUp(FeedingEnvHomotopyUp):
    def __init__(self):
        super(FeedingSawyerHumanEnvHomotopyUp, self).__init__(robot_type='sawyer', human_control=True)

class FeedingJacoHumanEnvHomotopyUp(FeedingEnvHomotopyUp):
    def __init__(self):
        super(FeedingJacoHumanEnvHomotopyUp, self).__init__(robot_type='jaco', human_control=True)

class FeedingPR2EnvHomotopyDown(FeedingEnvHomotopyDown):
    def __init__(self):
         super(FeedingPR2EnvHomotopyDown, self).__init__(robot_type='pr2', human_control=False)
 
class FeedingBaxterEnvHomotopyDown(FeedingEnvHomotopyDown):
    def __init__(self):
         super(FeedingBaxterEnvHomotopyDown, self).__init__(robot_type='baxter', human_control=False)
 
class FeedingSawyerEnvHomotopyDown(FeedingEnvHomotopyDown):
    def __init__(self):
         super(FeedingSawyerEnvHomotopyDown, self).__init__(robot_type='sawyer', human_control=False)
 
class FeedingJacoEnvHomotopyDown(FeedingEnvHomotopyDown):
    def __init__(self):
         super(FeedingJacoEnvHomotopyDown, self).__init__(robot_type='jaco', human_control=False)
 
class FeedingPR2HumanEnvHomotopyDown(FeedingEnvHomotopyDown):
    def __init__(self):
         super(FeedingPR2HumanEnvHomotopyDown, self).__init__(robot_type='pr2', human_control=True)
 
class FeedingBaxterHumanEnvHomotopyDown(FeedingEnvHomotopyDown):
    def __init__(self):
         super(FeedingBaxterHumanEnvHomotopyDown, self).__init__(robot_type='baxter', human_control=True)
 
class FeedingSawyerHumanEnvHomotopyDown(FeedingEnvHomotopyDown):
    def __init__(self):
         super(FeedingSawyerHumanEnvHomotopyDown, self).__init__(robot_type='sawyer', human_control=True)
 
class FeedingJacoHumanEnvHomotopyDown(FeedingEnvHomotopyDown):
    def __init__(self):
         super(FeedingJacoHumanEnvHomotopyDown, self).__init__(robot_type='jaco', human_control=True)

class FeedingPR2EnvHomotopyUpNo(FeedingEnvHomotopyUpNo):
    def __init__(self):
        super(FeedingPR2EnvHomotopyUpNo, self).__init__(robot_type='pr2', human_control=False)

class FeedingPR2EnvHomotopyDownNo(FeedingEnvHomotopyDownNo):
    def __init__(self):
        super(FeedingPR2EnvHomotopyDownNo, self).__init__(robot_type='pr2', human_control=False)

class FeedingPR2EnvHomotopyUpAdjust(FeedingEnvHomotopyUpAdjust):
    def __init__(self):
        super(FeedingPR2EnvHomotopyUpAdjust, self).__init__(robot_type='pr2', human_control=False)

class FeedingPR2EnvHomotopyDownAdjust(FeedingEnvHomotopyDownAdjust):
    def __init__(self):
        super(FeedingPR2EnvHomotopyDownAdjust, self).__init__(robot_type='pr2', human_control=False)

