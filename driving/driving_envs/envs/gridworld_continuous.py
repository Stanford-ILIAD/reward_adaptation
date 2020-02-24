import io
from typing import Text
import gym
from gym import spaces
from PIL import Image
import numpy as np
import scipy.special
from driving_envs.world import World
from driving_envs.entities import TextEntity, Entity
from driving_envs.agents import Car, Building, Goal
from driving_envs.geometry import Point
from typing import Tuple

class PidVelPolicy:
    """PID controller for H that maintains its initial velocity."""

    def __init__(self, dt: float, params: Tuple[float, float, float] = (3.0, 1.0, 6.0)):
        self._target_vel = None
        self.previous_error = 0
        self.integral = 0
        self.errors = []
        self.dt = dt
        self.Kp, self.Ki, self.Kd = params

    def action(self, obs):
        my_y_dot = obs[3]
        if self._target_vel is None:
            self._target_vel = my_y_dot
        error = self._target_vel - my_y_dot
        derivative = (error - self.previous_error) * self.dt
        self.integral = self.integral + self.dt * error
        acc = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        self.errors.append(error)
        return acc

    def reset(self):
        self._target_vel = None
        self.previous_error = 0
        self.integral = 0
        self.errors = []

    def __str__(self):
        return "PidVelPolicy({})".format(self.dt)

class GridworldContinuousEnv(gym.Env):

    def __init__(self,
                 dt: float = 0.1,
                 width: int = 50,
                 height: int = 50,
                 time_limit: float = 150.0):
        super(GridworldContinuousEnv, self).__init__()
        self.dt = dt
        self.width = width
        self.height = height
        self.world = World(self.dt, width=width, height=height, ppm=6)
        self.buildings, self.cars = [], {}
        self.step_num = 0
        self.accelerate = PidVelPolicy(dt=self.dt)
        self.time_limit = time_limit
        self.action_space = spaces.Box(
            np.array([-0.01]), np.array([0.01]), dtype=np.float32
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,))
        self.correct_pos = []
        self.next_pos = []
        self.start = np.array([5,5])
        self.goal = np.array([self.width-5., self.height-5.])
        self.max_dist = np.linalg.norm(self.goal-self.start,2)

    def step(self, action: np.ndarray, verbose: bool = False):
        self.step_num += 1

        car = self.world.dynamic_agents[0]
        acc = self.accelerate.action(self._get_obs())
        action = np.append(action, acc)
        car.set_control(*action)
        self.world.tick()

        reward = self.reward(verbose)

        done = False
        #for building in self.buildings:
        #    if car.collidesWith(building):
        #        done = True
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

        self.goal_obj = Goal(Point(self.goal[0], self.goal[1]), 0.0)

        for building in self.buildings:
            self.world.add(building)
        self.world.add(self.car)
        self.world.add(self.goal_obj)

        self.step_num = 0
        return self._get_obs()

    def _get_obs(self):
        """
        Get state of car
        """
        return self.world.state

    def reward(self, verbose):
        dist2goal = self.car.center.distanceTo(self.goal_obj)/self.max_dist
        coll_cost = 0
        for building in self.buildings:
            if self.car.collidesWith(building):
                coll_cost = -10
        reward = 1.0-dist2goal + coll_cost
        if verbose: print("dist to goal: ", dist2goal, "reward: ", reward)
        return reward

    def render(self):
        self.world.render()