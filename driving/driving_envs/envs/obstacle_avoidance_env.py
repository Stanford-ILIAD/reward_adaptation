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


class ObstacleAvoidanceEnv(gym.Env):
    """Driving gym interface"""

    def __init__(self,
                 dt: float = 0.1,
                 width: int = 120,
                 height: int = 120,
                 time_limit: float = 150.0):
        super(ObstacleAvoidanceEnv, self).__init__()
        self.dt = dt
        self.width = width
        self.height = height
        self.world = World(self.dt, width=width, height=height, ppm=6)
        self.buildings, self.cars = [], {}
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(12,))
        self.step_num = 0
        self.time_limit = time_limit
        self.action_space = spaces.Box(
            np.array((-0.1, -4.)), np.array((0.1, 4.0)), dtype=np.float32
            # np.array((-0.01, -4.)), np.array((0.01, 4.0)), dtype=np.float32
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(14,))
        # self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,))
        self.correct_pos = []
        self.next_pos = []

    def step(self, action: np.ndarray):
        self.step_num += 1

        # update robot actions
        r_action = action
        self.world.dynamic_agents[0].set_control(*r_action)
        self.world.tick()

        # get reward
        r_reward = self.reward()

        # check for dones
        done = False
        # if self.cars["R"].collidesWith(self.cars["H"]):
        #    done = True
        for car_name, car in self.cars.items():
            for building in self.buildings:
                if car.collidesWith(building):
                    done = True
            if car_name == "R" and car.y >= self.height or car.y <= 0 or car.x <= 0 or car.x >= self.width:
                done = True
                # raise ValueError("Car went out of bounds!")
        if self.step_num >= self.time_limit:
            done = True
        # print("step reward: ", r_reward)
        return self._get_obs(), r_reward, done, {'episode': {'r': r_reward, 'l': self.step_num}}

    def reset(self):
        self.world.reset()

        # create buildings
        self.buildings = [
            Building(Point(30, 60), Point(22, 10), "gray80"),
            Building(Point(90, 60), Point(22, 10), "gray80"),
            # Building(Point(62, 90), Point(2, 60), "gray80"),
        ]

        # create cars
        # h_y = 5
        r_y = 5
        self.cars = {
            # "H": Car(Point(58.5, h_y), np.pi / 2, "gray"),
            "R": Car(Point(60., r_y), np.pi / 2, "blue")
        }
        # h_yvel, r_yvel = 10, 10
        r_yvel = 10
        # self.cars["H"].velocity = Point(0, h_yvel)
        self.cars["R"].velocity = Point(0, r_yvel)

        # add the objects
        for building in self.buildings: self.world.add(building)
        self.world.add(self.cars["R"])
        # self.world.add(self.cars["H"])  # order in which dynamic agents are added determines concatenated state/actions

        self.step_num = 0
        return self._get_obs()

    def _get_obs(self):
        """
        Get state of both cars
        """
        # return np.concatenate((self.world.state[:6], self.world.state[7:13]))
        return np.concatenate((self.world.state, np.zeros(7)))
        # return self.world.state

    def reward(self, weight=100):
        car = self.cars["R"]
        coll_cost = -1000
        rew = weight * (car.center.x) + car.center.y + \
              coll_cost * car.collidesWith(self.buildings[1]) + coll_cost * car.collidesWith(self.buildings[0])
        return rew

    def render(self):
        self.world.render(self.correct_pos, self.next_pos)


class ObstacleAvoidanceEnv2(ObstacleAvoidanceEnv):
    def reward(self, weight=10):
        car = self.cars["R"]
        coll_cost = -1000
        rew = weight * (car.center.x) + car.center.y + \
              coll_cost * car.collidesWith(self.buildings[1]) + coll_cost * car.collidesWith(self.buildings[0])
        return rew


class ObstacleAvoidanceEnv3(ObstacleAvoidanceEnv):
    def reward(self, weight=1):
        car = self.cars["R"]
        coll_cost = -1000
        rew = weight * (car.center.x) + car.center.y + \
              coll_cost * car.collidesWith(self.buildings[1]) + coll_cost * car.collidesWith(self.buildings[0])
        return rew


class ObstacleAvoidanceEnv4(ObstacleAvoidanceEnv):
    def reward(self, weight=0.5):
        car = self.cars["R"]
        coll_cost = -1000
        rew = weight * (car.center.x) + car.center.y + \
              coll_cost * car.collidesWith(self.buildings[1]) + coll_cost * car.collidesWith(self.buildings[0])
        return rew


class ObstacleAvoidanceEnv5(ObstacleAvoidanceEnv):
    def reward(self, weight=0.0):
        car = self.cars["R"]
        coll_cost = -1000
        rew = weight * (car.center.x) + car.center.y + \
              coll_cost * car.collidesWith(self.buildings[1]) + coll_cost * car.collidesWith(self.buildings[0])
        return rew


class ObstacleAvoidanceEnv6(ObstacleAvoidanceEnv):
    def reward(self, weight=-0.5):
        car = self.cars["R"]
        coll_cost = -1000
        rew = weight * (car.center.x) + car.center.y + \
              coll_cost * car.collidesWith(self.buildings[1]) + coll_cost * car.collidesWith(self.buildings[0])
        return rew


class ObstacleAvoidanceEnv7(ObstacleAvoidanceEnv):
    def reward(self, weight=-1.0):
        car = self.cars["R"]
        coll_cost = -1000
        rew = weight * (car.center.x) + car.center.y + \
              coll_cost * car.collidesWith(self.buildings[1]) + coll_cost * car.collidesWith(self.buildings[0])
        return rew


class ObstacleAvoidanceEnv8(ObstacleAvoidanceEnv):
    def reward(self, weight=-10):
        car = self.cars["R"]
        coll_cost = -1000
        rew = weight * (car.center.x) + car.center.y + \
              coll_cost * car.collidesWith(self.buildings[1]) + coll_cost * car.collidesWith(self.buildings[0])
        return rew


class ObstacleAvoidanceEnv9(ObstacleAvoidanceEnv):
    def reward(self, weight=-1):
        car = self.cars["R"]
        coll_cost = weight * 100
        if car.center.x <= 30 and car.center.y >= 90:
            rew = 1000000
        else:
            rew = weight * (car.center.x) + car.center.y + \
                  coll_cost * car.collidesWith(self.buildings[1]) + coll_cost * car.collidesWith(self.buildings[0]) + \
                  coll_cost * (car.x <= 0) + coll_cost * (car.x >= self.width) +\
                  coll_cost * (car.y <= 0) + coll_cost * (car.y >= self.height)
        return rew
