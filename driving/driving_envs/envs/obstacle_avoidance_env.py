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
                 time_limit: float = 100.0):
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
            #np.array((-0.001, -4.)), np.array((0.001, 4.0)), dtype=np.float32
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(14,))
        self.goal = np.array([120, 120])
        self.K = 1
        self.r = 100

    def attractive_pf(self):
        att =  -self.K * (np.array([self.cars["R"].center.x, self.cars["R"].center.y]) - self.goal)
        print("att: ", att)
        return att

    def repulsive_pf(self):
        repulsive = np.array([0.0, 0.0])
        for da in self.world.dynamic_agents:
            if da != self.cars['R']:
                #if (isinstance(self.goal, Entity)) and np.all(da == self.goal):
                #    continue
                #else:
                min_dist = self.cars["R"].center.distanceTo(da.center)
                min_dist = max([self.r / 100, min_dist])
                print("min dist: ", min_dist)
                if min_dist <= self.r:
                    #norm_diff = (self.cars['R'].center - da.center) / min_dist
                    norm_diff = (np.array([self.cars['R'].center.x, self.cars['R'].center.y]) - np.array([da.center.x, da.center.y])) / min_dist
                    # repulse = ((1 / min_dist) - (1.0 / r)) * (1.0 / min_dist ** 2) * norm_diff * 50000
                    repulse = ((1 / min_dist) - (1.0 / self.r)) * (1.0 / min_dist ** 2) * norm_diff
                    repulsive += repulse
        for sa in self.world.static_agents:
            #if isinstance(self.goal, Entity) and np.all(sa == self.goal):
            #    continue
            #else:
            min_dist = float(self.cars['R'].center.distanceTo(sa.center))
            min_dist = max([self.r / 100, min_dist])
            if min_dist <= self.r:
                norm_diff = (np.array([self.cars['R'].center.x, self.cars['R'].center.y]) - np.array([sa.center.x, sa.center.y])) / min_dist
                # repulse = ((1 / min_dist) - (1.0 / r)) * (1.0 / min_dist ** 2) * norm_diff * 80000
                repulse = ((1 / min_dist) - (1.0 / self.r)) * (1.0 / min_dist ** 2) * norm_diff
                # repulsive += repulse * 1e3
                repulsive += repulse
        print("REpulsive: ", repulsive)
        return repulsive

    def step(self, action: np.ndarray):
        # update step count
        self.step_num += 1

        print()
        # calculate correct position
        potential_field = self.attractive_pf() #+ self.repulsive_pf()
        print("POT FIELD: ", potential_field)
        correct_position = np.array([self.cars['R'].x, self.cars['R'].y]) + (potential_field*self.dt*5)
        #correct_position = np.array([self.cars['R'].x, self.cars['R'].y]) + (potential_field)
        print("correct position: ", correct_position)

        # update robot actions
        r_action = action
        self.world.dynamic_agents[0].set_control(*r_action)
        self.world.tick()

        # get reward
        r_reward = self.reward(correct_position)

        # check for dones
        done = False
        #if self.cars["R"].collidesWith(self.cars["H"]):
        #    done = True
        for car_name, car in self.cars.items():
            for building in self.buildings:
                if car.collidesWith(building):
                    done = True
            if car_name == "R" and car.y >= self.height or car.y <= 0:
                done = True
                #raise ValueError("Car went out of bounds!")
        if self.step_num >= self.time_limit:
            done = True
        #print("step reward: ", r_reward)
        return self._get_obs(), r_reward, done, {'episode': {'r': r_reward, 'l': self.step_num}}

    def reset(self):
        self.world.reset()

        # create buildings
        self.buildings = [
            Building(Point(28.5, 60), Point(10, 10), "gray80"),
            Building(Point(91.5, 60), Point(20, 10), "gray80"),
            #Building(Point(62, 90), Point(2, 60), "gray80"),
        ]

        # create cars
        h_y = 5
        r_y = 5
        self.cars = {
            "H": Car(Point(58.5, h_y), np.pi / 2, "gray"),
            "R": Car(Point(61.5, r_y), np.pi / 2, "blue")
        }
        h_yvel, r_yvel = 10, 10
        #r_yvel = 10
        self.cars["H"].velocity = Point(0, h_yvel)
        self.cars["R"].velocity = Point(0, r_yvel)

        # add the objects
        for building in self.buildings: self.world.add(building)
        self.world.add(self.cars["R"])
        #self.world.add(self.cars["H"])  # order in which dynamic agents are added determines concatenated state/actions

        self.step_num = 0
        return self._get_obs()

    def _get_obs(self):
        """
        Get state of both cars
        """
        #return np.concatenate((self.world.state[:6], self.world.state[7:13]))
        return np.concatenate((self.world.state, np.zeros(7)))
        #return self.world.state


    def reward(self, correct_position):
        car = self.cars["R"]
        dist = np.linalg.norm(np.array([car.center.x, car.center.y]) - correct_position)
        print("reward: dist: ", dist)
        return -dist

    def render(self):
        self.world.render()
