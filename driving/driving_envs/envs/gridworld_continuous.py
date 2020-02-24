import io
from typing import Text
import gym
from gym import spaces
from PIL import Image
import numpy as np
import scipy.special
from driving_envs.world import World
from driving_envs.entities import TextEntity, Entity
from driving_envs.agents import Car, Building
from driving_envs.geometry import Point
from typing import Tuple


class GridworldContinuousEnv(gym.Env):

    def __init__(self,
                 dt: float = 0.1,
                 width: int = 120,
                 height: int = 120,
                 time_limit: float = 150.0):
        super(GridworldContinuousEnv, self).__init__()
        self.dt = dt
        self.width = width
        self.height = height
        self.world = World(self.dt, width=width, height=height, ppm=6)
        self.buildings, self.cars = [], {}
        self.step_num = 0
        self.time_limit = time_limit
        self.action_space = spaces.Box(
            np.array((-0.1)), np.array((0.1)), dtype=np.float32
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,))
        self.correct_pos = []
        self.next_pos = []
        self.start = np.array([5,5])

    def step(self, action: np.ndarray):
        self.step_num += 1

        car = self.world.dynamic_agents[0]
        car.set_control(*action)
        self.world.tick()

        reward = self.reward()

        done = False
        for building in self.buildings:
            if car.collidesWith(building):
                done = True
            if car.y >= self.height or car.y <= 0 or car.x <= 0 or car.x >= self.width:
                done = True
        if self.step_num >= self.time_limit:
            done = True
        return self._get_obs(), reward, done, {}

    def reset(self):
        self.world.reset()

        self.buildings = [
            Building(Point(self.width/2., self.height/2.), Point(3,3), "gray80")
        ]

        self.car = Car(Point(self.start[0], self.start[1]), np.pi/2., "blue")
        self.car.velocity = Point(0, 10)

        for building in self.buildings:
            self.world.add(building)
        self.world.add(self.car)

        self.step_num = 0
        return self._get_obs()

    def _get_obs(self):
        """
        Get state of car
        """
        return self.world.state

    def reward(self):
        return 1

    def render(self):
        self.world.render(self.correct_pos, self.next_pos)