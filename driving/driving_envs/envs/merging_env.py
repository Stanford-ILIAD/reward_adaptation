import io
from typing import Text
import gym
from gym import spaces
from PIL import Image
import numpy as np
import scipy.special
from driving_envs.world import World
from driving_envs.entities import TextEntity
from driving_envs.agents import Car, Building
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
        return np.array((0, acc))

    def reset(self):
        self._target_vel = None
        self.previous_error = 0
        self.integral = 0
        self.errors = []

    def __str__(self):
        return "PidVelPolicy({})".format(self.dt)


class MergingEnv(gym.Env):
    """Driving gym interface"""

    def __init__(self,
                 dt: float = 0.1,
                 width: int = 120,
                 height: int = 120,
                 time_limit: float = 60.0):
        super(MergingEnv, self).__init__()
        self.dt = dt
        self.width = width
        self.height = height
        self.world = World(self.dt, width=width, height=height, ppm=6)
        self.buildings, self.cars = [], {}
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(12,))
        self.human_policy = PidVelPolicy(dt=self.dt)
        self.step_num = 0
        self.time_limit = time_limit
        self.action_space = spaces.Box(
            #np.array((-0.1, -4.)), np.array((0.1, 4.0)), dtype=np.float32
            np.array((-0.01, -4.)), np.array((0.01, 4.0)), dtype=np.float32
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(14,))

    def step(self, action: np.ndarray):
        # update step count
        self.step_num += 1

        # update human and robot actions
        h_action = self.human_policy.action(self._get_obs())
        r_action = action
        #r_action[0] = 0
        self.world.dynamic_agents[0].set_control(*h_action)
        self.world.dynamic_agents[1].set_control(*r_action)
        self.world.tick()

        # get reward
        r_reward = self.reward()

        # check for dones
        done = False
        if self.cars["R"].collidesWith(self.cars["H"]):
            done = True
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
        self.buildings = [
            Building(Point(28.5, 60), Point(57, 120), "gray80"),
            Building(Point(91.5, 60), Point(57, 120), "gray80"),
            #Building(Point(62, 90), Point(2, 60), "gray80"),
        ]

        # create cars
        h_y, r_y = 5, 5
        self.cars = {
            "H": Car(Point(58.5, h_y), np.pi / 2, "gray"),
            "R": Car(Point(61.5, r_y), np.pi / 2, "blue")
        }
        h_yvel, r_yvel = 10, 10
        self.cars["H"].velocity = Point(0, h_yvel)
        self.cars["R"].velocity = Point(0, r_yvel)

        # add the objects
        for building in self.buildings: self.world.add(building)
        self.world.add(self.cars["H"])  # order in which dynamic agents are added determines concatenated state/actions
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
        human = self.cars["H"]
        coll_cost = 100

        # define rewards
        safe = -1*(car.center.y - human.center.y) + -1*(car.center.x - human.center.x) +\
                -10*coll_cost*car.collidesWith(self.buildings[1])

        eff = (car.center.y - human.center.y) + -1*(car.center.x - human.center.x) +\
            -10*coll_cost*car.collidesWith(self.buildings[1])
        return safe

    def render(self):
        self.world.render()


class MergingEnv2(MergingEnv):
    def reward(self, weight=-0.8):
        car = self.cars["R"]
        human = self.cars["H"]
        coll_cost = 100

        # define rewards
        safe = weight*(car.center.y - human.center.y) + -1*(car.center.x - human.center.x) + \
               -10*coll_cost*car.collidesWith(self.buildings[1])
        return safe

class MergingEnv3(MergingEnv):
    def reward(self, weight=-0.6):
        car = self.cars["R"]
        human = self.cars["H"]
        coll_cost = 100

        # define rewards
        safe = weight*(car.center.y - human.center.y) + -1*(car.center.x - human.center.x) + \
               -10*coll_cost*car.collidesWith(self.buildings[1])
        return safe

class MergingEnv4(MergingEnv):
    def reward(self, weight=-0.4):
        car = self.cars["R"]
        human = self.cars["H"]
        coll_cost = 100

        # define rewards
        safe = weight*(car.center.y - human.center.y) + -1*(car.center.x - human.center.x) + \
               -10*coll_cost*car.collidesWith(self.buildings[1])
        return safe

class MergingEnv5(MergingEnv):
    def reward(self, weight=-0.2):
        car = self.cars["R"]
        human = self.cars["H"]
        coll_cost = 100

        # define rewards
        safe = weight*(car.center.y - human.center.y) + -1*(car.center.x - human.center.x) + \
               -10*coll_cost*car.collidesWith(self.buildings[1])
        return safe

class MergingEnv6(MergingEnv):
    def reward(self, weight=0):
        car = self.cars["R"]
        human = self.cars["H"]
        coll_cost = 100

        # define rewards
        safe = weight*(car.center.y - human.center.y) + -1*(car.center.x - human.center.x) + \
               -10*coll_cost*car.collidesWith(self.buildings[1])
        return safe

class MergingEnv7(MergingEnv):
    def reward(self, weight=0.2):
        car = self.cars["R"]
        human = self.cars["H"]
        coll_cost = 100

        # define rewards
        safe = weight*(car.center.y - human.center.y) + -1*(car.center.x - human.center.x) + \
               -10*coll_cost*car.collidesWith(self.buildings[1])
        return safe

class MergingEnv8(MergingEnv):
    def reward(self, weight=0.4):
        car = self.cars["R"]
        human = self.cars["H"]
        coll_cost = 100

        # define rewards
        safe = weight*(car.center.y - human.center.y) + -1*(car.center.x - human.center.x) + \
               -10*coll_cost*car.collidesWith(self.buildings[1])
        return safe

class MergingEnv9(MergingEnv):
    def reward(self, weight=0.6):
        car = self.cars["R"]
        human = self.cars["H"]
        coll_cost = 100

        # define rewards
        safe = weight*(car.center.y - human.center.y) + -1*(car.center.x - human.center.x) + \
               -10*coll_cost*car.collidesWith(self.buildings[1])
        return safe

class MergingEnv10(MergingEnv):
    def reward(self, weight=0.8):
        car = self.cars["R"]
        human = self.cars["H"]
        coll_cost = 100

        # define rewards
        safe = weight*(car.center.y - human.center.y) + -1*(car.center.x - human.center.x) + \
               -10*coll_cost*car.collidesWith(self.buildings[1])
        return safe

class MergingEnv11(MergingEnv):
    def reward(self, weight=1.0):
        car = self.cars["R"]
        human = self.cars["H"]
        coll_cost = 100

        # define rewards
        safe = weight*(car.center.y - human.center.y) + -1*(car.center.x - human.center.x) + \
               -10*coll_cost*car.collidesWith(self.buildings[1])
        return safe
