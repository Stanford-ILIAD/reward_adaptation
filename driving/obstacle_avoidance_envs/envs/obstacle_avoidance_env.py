import io
from typing import Text
import gym
from gym import spaces
from PIL import Image
import numpy as np
import scipy.special
from obstacle_avoidance_envs.world import World
from obstacle_avoidance_envs.entities import TextEntity
from obstacle_avoidance_envs.agents import Car, Building
from obstacle_avoidance_envs.geometry import Point
from typing import Tuple


class ObstacleAvoidanceEnv(gym.Env):
    """Driving gym interface"""

    def __init__(self,
                 dt: float = 0.1,
                 width: int = 120,
                 height: int = 120,
                 time_limit: float = 60.0):
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
            #np.array((-0.1, -4.)), np.array((0.1, 4.0)), dtype=np.float32
            np.array((-0.001, -4.)), np.array((0.001, 4.0)), dtype=np.float32
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(14,))

    def step(self, action: np.ndarray):
        # update step count
        self.step_num += 1

        # update robot actions
        r_action = action
        #r_action[0] = 0
        #self.world.dynamic_agents[0].set_control(*r_action)
        self.world.tick()

        # get reward
        r_reward = self.reward()

        # check for dones
        done = False
        #if self.cars["R"].collidesWith(self.cars["H"]):
        #    done = True
        for car_name, car in self.cars.items():
            for building in self.buildings:
                if car.collidesWith(building):
                    done = True
            if car_name == "R" and car.y >= self.height or car.y <= 0:
                raise ValueError("Car went out of bounds!")
        if self.step_num >= self.time_limit:
            done = True
        #print("step reward: ", r_reward)
        return self._get_obs(), r_reward, done, {'episode': {'r': r_reward, 'l': self.step_num}}

    def reset(self):
        self.world.reset()

        # create buildings
        #self.buildings = [
        #    Building(Point(28.5, 60), Point(57, 120), "gray80"),
        #    Building(Point(91.5, 60), Point(57, 120), "gray80"),
        #    #Building(Point(62, 90), Point(2, 60), "gray80"),
        #]

        # create cars
        #h_y, 5\
        r_y = 5
        self.cars = {
            #"H": Car(Point(58.5, h_y), np.pi / 2, "gray"),
            "R": Car(Point(61.5, r_y), np.pi / 2, "blue")
        }
        #h_yvel, r_yvel = 10, 10
        r_yvel = 10
        #self.cars["H"].velocity = Point(0, h_yvel)
        self.cars["R"].velocity = Point(0, r_yvel)

        # add the objects
        #for building in self.buildings: self.world.add(building)
        #self.world.add(self.cars["H"])  # order in which dynamic agents are added determines concatenated state/actions
        self.world.add(self.cars["R"])

        self.step_num = 0
        return self._get_obs()

    def _get_obs(self):
        """
        Get state of both cars
        """
        #return np.concatenate((self.world.state[:6], self.world.state[7:13]))
        return self.world.state

    def reward(self):
        car = self.cars["R"]
        #human = self.cars["H"]
        #coll_cost = 100

        # define rewards
        #safe = -1*(car.center.y - human.center.y) + -1*(car.center.x - human.center.x) +\
        #        -10*coll_cost*car.collidesWith(self.buildings[1])

        #eff = (car.center.y - human.center.y) + -1*(car.center.x - human.center.x) +\
        #    -10*coll_cost*car.collidesWith(self.buildings[1])

        #print("\ny dist: ", -1*(car.center.y - human.center.y) )
        #print("x dist: ", -1*(car.center.x - human.center.x) )
        #print("rew: ", safe)
        return 1.0

    def render(self):
        self.world.render()
