import io
from typing import Text
import gym
from gym import spaces
from PIL import Image
import numpy as np
import scipy.special
from driving_envs.world import World
from driving_envs.entities import TextEntity
from driving_envs.agents import Car, Building, Barrier1, Barrier2
from driving_envs.geometry import Point
from typing import Tuple
import math

class NavigationEnv(gym.Env):
    """Driving gym interface"""

    def __init__(self,
                 dt: float = 0.1,
                 width: int = 40,
                 height: int = 40,
                 time_limit: float = 300.0):
        super(NavigationEnv, self).__init__()
        self.dt = dt
        self.width = width
        self.height = height
        self.world = World(self.dt, width=width, height=height, ppm=6)
        self.buildings, self.cars = [], {}
        self.step_num = 0
        self.time_limit = time_limit
        self.action_space = spaces.Box(
            np.array((-2, -2)), np.array((2, 2)), dtype=np.float32
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,))

    def step(self, action: np.ndarray):
        # update step count
        self.step_num += 1
        center_backup = self.cars["R"].center
        self.world.dynamic_agents[0].set_control(*action)
        self.world.tick()

        # check for dones
        done = False
        for building in self.buildings:
            if self.cars["R"].collidesWith(building):
                self.collide_building = True
                #self.cars["R"].center.x = center_backup.x
                #self.cars["R"].center.y = center_backup.y
                #done = True
        if self.cars["R"].center.x > 35 and self.cars["R"].center.y > 35:
            done = True
            self.finish = True
        if self.cars["R"].y >= self.height:
            self.cars["R"].center.y = self.height
            self.cars["R"].velocity.y = 0
            self.collide = True
        elif self.cars["R"].y <= 0:
            self.cars["R"].center.y = 0
            self.cars["R"].velocity.y = 0
            self.collide = True
        if self.cars["R"].x >= self.width:
            self.cars["R"].center.x = self.width
            self.cars["R"].velocity.x = 0
            self.collide = True
        elif self.cars["R"].x <= 0:
            self.cars["R"].center.x = 0
            self.cars["R"].velocity.x = 0
            self.collide = True
        if self.step_num >= self.time_limit:
            done = True

        r_reward = self.reward()
        #print("step reward: ", r_reward)
        return self._get_obs(), r_reward, done, {'episode': {'r': r_reward, 'l': self.step_num}}

    def reset(self):
        self.world.reset()

        # create buildings
        self.buildings = [
            #Barrier1(Point(20, 20), 5, "gray80"),
        ]
        '''
        self.buildings = [
            Barrier1(Point(30, 90), 25, "gray80"),
            Barrier1(Point(90, 90), 25, "gray80"),
            Barrier1(Point(60, 30), 25, "gray80"),
        ]
        '''

        # create cars
        self.cars = {
            "R": Car(Point(5, 5), 0, "blue", -math.inf, math.inf)
        }
        self.cars["R"].velocity = Point(2, 2)

        # add the objects
        for building in self.buildings: self.world.add(building)
        self.world.add(self.cars["R"])
        self.collide = False
        self.collide_building = False
        self.finish = False

        self.step_num = 0
        return self._get_obs()

    def _get_obs(self):
        """
        Get state of both cars
        """
        '''
        barrier_states = []
        for building in self.buildings:
            barrier_states.append(np.sqrt(np.square(self.cars["R"].center.x-building.center.x)+np.square(self.cars["R"].center.y-building.center.y))-building.radius)
        barrier_states.extend([abs(self.cars["R"].center.x-120),
                               abs(self.cars["R"].center.y-120),
                               abs(self.cars["R"].center.x),
                               abs(self.cars["R"].center.y)])
        return np.concatenate([self.world.state, np.array(barrier_states)])
        '''
        state = np.copy(self.world.state[0:4])
        state[0] = (state[0] - (self.width / 2)) / (self.width / 2)
        state[1] = (state[1] - (self.height / 2)) / (self.height / 2)
        return state

    def reward(self):
        barrier_states = []
        #for building in self.buildings:
        #    barrier_states.append(np.sqrt(np.square(self.cars["R"].center.x-building.center.x)+np.square(self.cars["R"].center.y-building.center.y))-building.radius)
        barrier_states.extend([abs(self.cars["R"].center.x-self.width) if abs(self.cars["R"].center.x) > abs(self.cars["R"].center.x-self.width) else abs(self.cars["R"].center.x),
                               abs(self.cars["R"].center.y-self.height) if abs(self.cars["R"].center.y) > abs(self.cars["R"].center.y-self.height) else abs(self.cars["R"].center.y),])
        caution = 0
        for barrier_state in barrier_states:
            caution += 50 * (1-np.exp(-barrier_state/(self.width/4)))
        return -np.sqrt(np.square(self.cars["R"].center.x-self.width)+np.square(self.cars["R"].center.y-self.height)) - (10000 if self.collide else 0) 

    def render(self):
        self.world.render()

class NavigationEnv2(NavigationEnv):
    def reset(self):
        self.world.reset()

        # create buildings
        self.buildings = [
            Barrier1(Point(20, 20), 5, "gray80"),
        ]
        '''
        self.buildings = [
            Barrier1(Point(30, 90), 25, "gray80"),
            Barrier1(Point(90, 90), 25, "gray80"),
            Barrier1(Point(60, 30), 25, "gray80"),
        ]
        '''

        # create cars
        self.cars = {
            "R": Car(Point(5, 5), 0, "blue", -math.inf, math.inf)
        }
        self.cars["R"].velocity = Point(2, 2)

        # add the objects
        for building in self.buildings: self.world.add(building)
        self.world.add(self.cars["R"])
        self.collide = False
        self.collide_building = False
        self.finish = False

        self.step_num = 0
        return self._get_obs()

    def reward(self):
        goal_rew = -np.sqrt(np.square(self.cars["R"].center.x-self.width)+np.square(self.cars["R"].center.y-self.height))/5 + (10 if self.finish else 0)
        boundary_rew = -(4 if self.collide else 0)
        building_rew = -(10000 if self.collide_building else 0)
        return goal_rew + boundary_rew + building_rew + prefer_rew


class NavigationEnv3(NavigationEnv2):
    def reward(self):
        goal_rew = -np.sqrt(np.square(self.cars["R"].center.x-self.width)+np.square(self.cars["R"].center.y-self.height))/20 + (1000 if self.finish else 0)
        boundary_rew = -(1 if self.collide else 0)
        building_rew = -(200 if self.collide_building else 0)
        prefer_rew = 0.25 if abs(self.cars["R"].center.x) > abs(self.cars["R"].center.y) else -2.25
        return goal_rew + boundary_rew + building_rew + prefer_rew


class NavigationEnv31(NavigationEnv3):
    def reset(self):
        self.world.reset()

        # create buildings
        self.buildings = [
            Barrier1(Point(20, 20), 8, "gray80"),
        ]
        '''
        self.buildings = [
            Barrier1(Point(30, 90), 25, "gray80"),
            Barrier1(Point(90, 90), 25, "gray80"),
            Barrier1(Point(60, 30), 25, "gray80"),
        ]
        '''

        # create cars
        self.cars = {
            "R": Car(Point(5, 5), 0, "blue", -math.inf, math.inf)
        }
        self.cars["R"].velocity = Point(2, 2)

        # add the objects
        for building in self.buildings: self.world.add(building)
        self.world.add(self.cars["R"])
        self.collide = False
        self.collide_building = False
        self.finish = False

        self.step_num = 0
        return self._get_obs()

class NavigationEnv4(NavigationEnv2):
    def reward(self):
        goal_rew = -np.sqrt(np.square(self.cars["R"].center.x-self.width)+np.square(self.cars["R"].center.y-self.height))/20 + (1000 if self.finish else 0)
        boundary_rew = -(1 if self.collide else 0)
        building_rew = -(200 if self.collide_building else 0)
        prefer_rew = -2.25 if abs(self.cars["R"].center.x) > abs(self.cars["R"].center.y) else 0.25
        return goal_rew + boundary_rew + building_rew + prefer_rew

class NavigationEnv41(NavigationEnv4):
    def reset(self):
        self.world.reset()

        # create buildings
        self.buildings = [
            Barrier1(Point(20, 20), 8, "gray80"),
        ]
        '''
        self.buildings = [
            Barrier1(Point(30, 90), 25, "gray80"),
            Barrier1(Point(90, 90), 25, "gray80"),
            Barrier1(Point(60, 30), 25, "gray80"),
        ]
        '''

        # create cars
        self.cars = {
            "R": Car(Point(5, 5), 0, "blue", -math.inf, math.inf)
        }
        self.cars["R"].velocity = Point(2, 2)

        # add the objects
        for building in self.buildings: self.world.add(building)
        self.world.add(self.cars["R"])
        self.collide = False
        self.collide_building = False
        self.finish = False

        self.step_num = 0
        return self._get_obs()

class NavigationEnv32(NavigationEnv3):
    def reset(self):
        self.world.reset()

        # create buildings
        self.buildings = [
        ]
        '''
        self.buildings = [
            Barrier1(Point(30, 90), 25, "gray80"),
            Barrier1(Point(90, 90), 25, "gray80"),
            Barrier1(Point(60, 30), 25, "gray80"),
        ]
        '''

        # create cars
        self.cars = {
            "R": Car(Point(5, 5), 0, "blue", -math.inf, math.inf)
        }
        self.cars["R"].velocity = Point(2, 2)

        # add the objects
        for building in self.buildings: self.world.add(building)
        self.world.add(self.cars["R"])
        self.collide = False
        self.collide_building = False
        self.finish = False

        self.step_num = 0
        return self._get_obs()

class NavigationEnv33(NavigationEnv32):
    def reward(self):
        goal_rew = -np.sqrt(np.square(self.cars["R"].center.x-self.width)+np.square(self.cars["R"].center.y-self.height))/20 + (1000 if self.finish else 0)
        boundary_rew = -(1 if self.collide else 0)
        building_rew = -(200 if self.collide_building else 0)
        prefer_rew = 1. if abs(self.cars["R"].center.x) > abs(self.cars["R"].center.y) else -2.25
        return goal_rew + boundary_rew + building_rew + prefer_rew

class NavigationEnv42(NavigationEnv4):
    def reset(self):
        self.world.reset()

        # create buildings
        self.buildings = [
        ]
        '''
        self.buildings = [
            Barrier1(Point(30, 90), 25, "gray80"),
            Barrier1(Point(90, 90), 25, "gray80"),
            Barrier1(Point(60, 30), 25, "gray80"),
        ]
        '''

        # create cars
        self.cars = {
            "R": Car(Point(5, 5), 0, "blue", -math.inf, math.inf)
        }
        self.cars["R"].velocity = Point(2, 2)

        # add the objects
        for building in self.buildings: self.world.add(building)
        self.world.add(self.cars["R"])
        self.collide = False
        self.collide_building = False
        self.finish = False

        self.step_num = 0
        return self._get_obs()
