import gym
from gym import spaces
import numpy as np
from tightrope.gym_tightrope import register


class Tightrope(gym.Env):
    def __init__(self):
        self.dt = 0.1
        self.step_num = 0
        self.action_space = spaces.Box(
            np.array([-0.1]), np.array([0.1]), dtype=np.float32
        )
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(1,))
        self.start = -1.0
        self.goal = 1.0

    def step(self, action):
        self.step_num += 1
        self.S += action*self.S

        if self.S < self.start:
            done = True
        if self.S > self.goal:
            done = True

        reward = self.reward()
        return self.S, reward, done, {}

    def reset(self):
        self.S = self.start
        #self.obstacle = ()

    def reward(self):
        dist2goal = self.goal - self.S
        coll_cost = 0
        if self.obstacle and self.S >= self.obstacle[0] and self.S <= self.obstacle[1]:
            coll_cost = -10
        reward = dist2goal + coll_cost
        print("state: {}, dist2goal: {}, coll cost: {}".format(self.S, dist2goal, coll_cost))
        return reward


#register(
    #id='Tightrope-v0',
   # entry_point='gym_tightrope.envs:Tightrope'
#)
