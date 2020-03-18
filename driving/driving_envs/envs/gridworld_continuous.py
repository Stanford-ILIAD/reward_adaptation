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
            np.array([-0.04]), np.array([0.04]), dtype=np.float32
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,))
        self.correct_pos = []
        self.next_pos = []
        self.start = np.array([self.width/2.,5])
        self.goal = np.array([self.width/2., self.height-5.])
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
        #if car.collidesWith(self.goal_obj):
        #    if verbose: print("COLLIDING WITH GOAL!")
        #    done = True
        return self._get_obs(), reward, done, {'episode': {'r': reward, 'l': self.step_num}}

    def reset(self):
        self.world.reset()

        self.buildings = [
            Building(Point(self.width/2., self.height/2.), Point(1,1), "gray80")
        ]

        self.car = Car(Point(self.start[0], self.start[1]), np.pi/2., "blue")
        self.car.velocity = Point(0, 5)

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

    def reward(self, verbose, weight=10.0):
        dist2goal = 1.0 - (self.car.center.distanceTo(self.goal_obj)/self.max_dist)
        coll_cost = 0
        for building in self.buildings:
            if self.car.collidesWith(building):
                coll_cost = -1000

        goal_rew = 0.0
        if self.car.collidesWith(self.goal_obj):
            goal_rew = 10

        # adding preference
        heading = self.world.state[-3]
        #mean_heading = 2.0
        mean_heading = np.pi/2.0
        gamma = 0.9
        homotopy_rew = 0.0
        homotopy_rew += 2*(heading-mean_heading) # left
        #homotopy_rew += -2*(heading-mean_heading) # right
        homotopy_rew *= gamma**(self.step_num)
        dist2goal *= (1.0 - gamma**(self.step_num))

        reward = np.sum(np.array([
                 dist2goal,
                 coll_cost,
                 #goal_rew,
                 homotopy_rew
            ]))
        if verbose: print("dist to goal: ", dist2goal,
                          "homotopy: ", homotopy_rew,
                          "heading: ", heading,
                          "reward: ", reward)
        return reward

    def render(self):
        self.world.render()

class GridworldContinuousMultiObjLLEnv(GridworldContinuousEnv):
    def __init__(self,
                 dt: float = 0.1,
                 width: int = 50,
                 height: int = 100,
                 time_limit: float = 300.0):
        super(GridworldContinuousMultiObjLLEnv, self).__init__(dt=dt, width=width, height=height, time_limit=time_limit)
        self.action_space = spaces.Box(
            np.array([-0.1]), np.array([0.1]), dtype=np.float32
        )

    def reset(self):
        self.check_point1 = False
        self.check_point2 = False
        self.check_point3 = False
        self.check_point4 = False
        self.check_point5 = False
        
        self.world.reset()

        self.buildings = [
            Building(Point(int(self.width/2.), int(self.height*3./5.)), Point(9,4), "gray80"),
            Building(Point(int(self.width/2.), int(self.height*3./10.)), Point(9,4), "gray80"),
            Building(Point(int(self.width-2), int(self.height*3./10.)), Point(4,int(self.height*3./5.)), "gray80"),
            Building(Point(int(2), int(self.height*3./10.)), Point(4,int(self.height*3./5.)), "gray80"),
        ]

        self.car = Car(Point(self.start[0], self.start[1]), np.pi/2., "blue")
        self.car.velocity = Point(0, 5)

        self.goal_obj = Goal(Point(self.goal[0], self.goal[1]), 0.0)

        for building in self.buildings:
            self.world.add(building)
        self.world.add(self.car)
        self.world.add(self.goal_obj)
        
        self.last_heading = np.pi / 2

        self.step_num = 0
        return self._get_obs()

    def reward(self, verbose, weight=10.0):
        checkpoint_portion = 1/4.
        dist2goal = 1.0 - (self.car.center.distanceTo(self.goal_obj)/self.max_dist)
        coll_cost = 0
        for building in self.buildings:
            if self.car.collidesWith(building):
                coll_cost = -1000

        goal_rew = 0.0
        if self.car.collidesWith(self.goal_obj):
            goal_rew = 1000

        # adding preference
        heading = self.world.state[-3]
        max_heading = 2.0
        mean_heading = np.pi / 2
        gamma = 0.9
        #dist2left = 1.5*(self.width-self.car.center.x)/self.width
        homotopy_rew = 0.0
        if self.width / 4. < self.car.x < self.width / 2.:
            homotopy_rew += 0.5
 
        if int(self.height*3./10.) -4. < self.car.y < int(self.height*3./10.):
            if self.width * checkpoint_portion < self.car.x < self.width / 2. and (not self.check_point1):
                homotopy_rew += 500.
                self.check_point1 = True
        elif int(self.height*3./5.) -4. < self.car.y < int(self.height*3./5.):
            if self.width / 2. > self.car.x > self.width * checkpoint_portion and (not self.check_point2):
                homotopy_rew += 500.
                self.check_point2 = True
        elif int(self.height*4./5.) < self.car.y:
            if self.width / 2. -5. < self.car.x < self.width/2. + 5.:
                homotopy_rew += 5.
        
        if abs(heading-mean_heading) > 1.5:
            homotopy_rew += -100000.

        #homotopy_rew *= 0.0 # gamma**(self.step_num)
        #dist2goal *= 0.8 #(1.0 - gamma**(self.step_num))

        boundary_rew = 1.-abs(self.width/2. - self.car.x) / (self.width/2.)
        self.last_heading = heading
        reward = np.sum(np.array([
                 #new_dist2goal,
                 dist2goal,
                 coll_cost,
                 #goal_rew,
                 homotopy_rew,
                 boundary_rew
            ]))


        if verbose: print("dist to goal: ", dist2goal,
                          "homotopy: ", homotopy_rew,
                          "heading: ", heading,
                          "reward: ", reward)
        return reward

class GridworldContinuousMultiObjRREnv(GridworldContinuousMultiObjLLEnv):
    def reward(self, verbose, weight=10.0):
        checkpoint_portion = 1. / 4.        

        dist2goal = 1.0 - (self.car.center.distanceTo(self.goal_obj)/self.max_dist)
        coll_cost = 0
        for building in self.buildings:
            if self.car.collidesWith(building):
                coll_cost = -1000

        goal_rew = 0.0
        if self.car.collidesWith(self.goal_obj):
            goal_rew = 1000

        # adding preference
        heading = self.world.state[-3]
        max_heading = 2.0
        mean_heading = np.pi / 2
        gamma = 0.9
        #dist2left = 1.5*(self.width-self.car.center.x)/self.width
        homotopy_rew = 0.0
        if self.width *3./ 4. > self.car.x > self.width / 2.:
            homotopy_rew += 0.5
 
        if int(self.height*3./10.) -4. < self.car.y < int(self.height*3./10.):
            if self.width * (1-checkpoint_portion) > self.car.x > self.width / 2. and (not self.check_point1):
                homotopy_rew += 500.
                self.check_point1 = True
        elif int(self.height*3./5.) -4. < self.car.y < int(self.height*3./5.):
            if self.width / 2. < self.car.x < self.width *(1-checkpoint_portion) and (not self.check_point2):
                homotopy_rew += 500.
                self.check_point2 = True
        elif int(self.height*4./5.) < self.car.y:
            if self.width / 2. -5. < self.car.x < self.width/2. + 5.:
                homotopy_rew += 5.
        
        if abs(heading-mean_heading) > 1.5:
            homotopy_rew += -100000.

        #homotopy_rew *= 0.0 # gamma**(self.step_num)
        #dist2goal *= 0.8 #(1.0 - gamma**(self.step_num))

        boundary_rew = 1.-abs(self.width/2. - self.car.x) / (self.width/2.)
        self.last_heading = heading
        reward = np.sum(np.array([
                 #new_dist2goal,
                 dist2goal,
                 coll_cost,
                 #goal_rew,
                 homotopy_rew,
                 boundary_rew
            ]))


        if verbose: print("dist to goal: ", dist2goal,
                          "homotopy: ", homotopy_rew,
                          "heading: ", heading,
                          "reward: ", reward)
        return reward


class GridworldContinuousMultiObjLREnv(GridworldContinuousMultiObjLLEnv):
    def reward(self, verbose, weight=10.0):
        checkpoint_portion = 1/4.
        dist2goal = 5*(1.0 - (self.car.center.distanceTo(self.goal_obj)/self.max_dist))
        coll_cost = 0
        for building in self.buildings:
            if self.car.collidesWith(building):
                coll_cost = -1000.

        goal_rew = 0.0
        if self.car.collidesWith(self.goal_obj):
            goal_rew = 100.

        # adding preference
        heading = self.world.state[-3]
        max_heading = 2.0
        mean_heading = np.pi / 2
        gamma = 0.9
        #dist2left = 1.5*(self.width-self.car.center.x)/self.width
        homotopy_rew = 0.0
        if self.car.y < int(self.height*3./10.):
            homotopy_rew += 5*(heading-mean_heading) if heading-mean_heading < 0.5 else 0.
        elif int(self.height*3./10.) <= self.car.y < int(self.height*3./5.):
            homotopy_rew += -5*(heading-mean_heading) if mean_heading - heading < 0.7 else 0.
        else:
            homotopy_rew += 5*(heading-mean_heading) if heading-mean_heading < 0.7 else 0.

        if int(self.height*3./10.) -4. < self.car.y < int(self.height*3./10.):
            if self.width * checkpoint_portion < self.car.x < self.width / 2. and (not self.check_point1):
                homotopy_rew += 500.
                self.check_point1 = True
            elif self.width / 2. < self.car.x and (not self.check_point1):
                homotopy_rew -= 100000.
                self.check_point1 = True
        elif int(self.height*3./5.) -4. < self.car.y < int(self.height*3./5.):
            if self.width / 2. < self.car.x < self.width * (1-checkpoint_portion) and (not self.check_point2):
                homotopy_rew += 500.
                self.check_point2 = True
            elif self.car.x < self.width / 2. and (not self.check_point2):
                homotopy_rew -= 100000.
                self.check_point2 = True
        
        if abs(heading-mean_heading) > 1.5:
            homotopy_rew += -100000.

        boundary_rew = 5*(1.-abs(self.width/2. - self.car.x) / (self.width/2.))
        self.last_heading = heading
        reward = np.sum(np.array([
                 #new_dist2goal,
                 dist2goal,
                 coll_cost,
                 goal_rew,
                 homotopy_rew,
                 boundary_rew
            ]))
        if verbose: print("dist to goal: ", dist2goal,
                          "homotopy: ", homotopy_rew,
                          "heading: ", heading,
                          "reward: ", reward)
        return reward

class GridworldContinuousMultiObjRLEnv(GridworldContinuousMultiObjLLEnv):
    def reward(self, verbose, weight=10.0):
        checkpoint_portion = 1/4.
        dist2goal = 5*(1.0 - (self.car.center.distanceTo(self.goal_obj)/self.max_dist))
        coll_cost = 0
        for building in self.buildings:
            if self.car.collidesWith(building):
                coll_cost -= 1000.

        goal_rew = 0.0
        if self.car.collidesWith(self.goal_obj):
            goal_rew = 1000

        # adding preference
        heading = self.world.state[-3]
        max_heading = 2.0
        mean_heading = np.pi / 2
        gamma = 0.9
        #dist2left = 1.5*(self.width-self.car.center.x)/self.width
        homotopy_rew = 0.0
        if self.car.y < int(self.height*3./10.):
            homotopy_rew += -5*(heading-mean_heading) if mean_heading-heading < 0.5 else 0.
        elif int(self.height*3./10.) <= self.car.y < int(self.height*3./5.):
            homotopy_rew += 5*(heading-mean_heading) if heading - mean_heading < 0.7 else 0.
        else:
            homotopy_rew += -5*(heading-mean_heading) if mean_heading-heading < 0.7 else 0.

        if int(self.height*3./10.) -4. < self.car.y < int(self.height*3./10.):
            if self.width * (1-checkpoint_portion) > self.car.x > self.width / 2. and (not self.check_point1):
                homotopy_rew += 500.
                self.check_point1 = True
            elif self.width / 2. > self.car.x and (not self.check_point1):
                homotopy_rew -= 100000.
                self.check_point1 = True
        elif int(self.height*3./5.) -4. < self.car.y < int(self.height*3./5.):
            if self.width / 2. > self.car.x > self.width * checkpoint_portion and (not self.check_point2):
                homotopy_rew += 500.
                self.check_point2 = True
            elif self.car.x > self.width / 2. and (not self.check_point2):
                homotopy_rew -= 100000.
                self.check_point2 = True
        
        if abs(heading-mean_heading) > 1.5:
            homotopy_rew += -100000.

        boundary_rew = 5*(1.-abs(self.width/2. - self.car.x) / (self.width/2.))
        self.last_heading = heading
        reward = np.sum(np.array([
                 #new_dist2goal,
                 dist2goal,
                 coll_cost,
                 goal_rew,
                 homotopy_rew,
                 boundary_rew
            ]))
        if verbose: print("dist to goal: ", dist2goal,
                          "homotopy: ", homotopy_rew,
                          "heading: ", heading,
                          "reward: ", reward)
        return reward


class GridworldContinuousNoneRLEnv(GridworldContinuousMultiObjRLEnv):
    def reset(self):
        self.check_point1 = False
        self.check_point2 = False
        
        self.world.reset()

        self.buildings = [
            #Building(Point(int(self.width/2.), int(self.height*3./5.)), Point(4,4), "gray80"),
            #Building(Point(int(self.width/2.), int(self.height*3./10.)), Point(4,4), "gray80")
            Building(Point(int(self.width-2), int(self.height*3./10.)), Point(4,int(self.height*3./5.)), "gray80"),
            Building(Point(int(2), int(self.height*3./10.)), Point(4,int(self.height*3./5.)), "gray80"),
        ]

        self.car = Car(Point(self.start[0], self.start[1]), np.pi/2., "blue")
        self.car.velocity = Point(0, 5)

        self.goal_obj = Goal(Point(self.goal[0], self.goal[1]), 0.0)

        for building in self.buildings:
            self.world.add(building)
        self.world.add(self.car)
        self.world.add(self.goal_obj)
        
        self.last_heading = np.pi / 2

        self.step_num = 0
        return self._get_obs()

    def reward(self, verbose, weight=10.0):
        checkpoint_portion = 1/4.
        dist2goal = 5*(1.0 - (self.car.center.distanceTo(self.goal_obj)/self.max_dist))
        coll_cost = 0
        for building in self.buildings:
            if self.car.collidesWith(building):
                coll_cost = -100000.

        goal_rew = 0.0
        if self.car.collidesWith(self.goal_obj):
            goal_rew = 1000

        # adding preference
        heading = self.world.state[-3]
        max_heading = 2.0
        mean_heading = np.pi / 2
        gamma = 0.9
        #dist2left = 1.5*(self.width-self.car.center.x)/self.width
        homotopy_rew = 0.0
        if self.car.y < int(self.height*3./10.):
            homotopy_rew += -5*(heading-mean_heading) if mean_heading-heading < 0.5 else 0.
        elif int(self.height*3./10.) <= self.car.y < int(self.height*3./5.):
            homotopy_rew += 5*(heading-mean_heading) if heading - mean_heading < 0.7 else 0.
        else:
            homotopy_rew += -5*(heading-mean_heading) if mean_heading-heading < 0.7 else 0.

        if int(self.height*3./10.) -4. < self.car.y < int(self.height*3./10.):
            if self.width * (1-checkpoint_portion) > self.car.x > self.width / 2. and (not self.check_point1):
                homotopy_rew += 500.
                self.check_point1 = True
            elif self.width / 2. > self.car.x and (not self.check_point1):
                homotopy_rew -= 100000.
                self.check_point1 = True
        elif int(self.height*3./5.) -4. < self.car.y < int(self.height*3./5.):
            if self.width / 2. > self.car.x > self.width * checkpoint_portion and (not self.check_point2):
                homotopy_rew += 500.
                self.check_point2 = True
            elif self.car.x > self.width / 2. and (not self.check_point2):
                homotopy_rew -= 100000.
                self.check_point2 = True
        
        if abs(heading-mean_heading) > 1.5:
            homotopy_rew += -100000.

        boundary_rew = 5*(1.-abs(self.width/2. - self.car.x) / (self.width/2.))
        self.last_heading = heading
        reward = np.sum(np.array([
                 #new_dist2goal,
                 dist2goal,
                 coll_cost,
                 goal_rew,
                 homotopy_rew,
                 boundary_rew
            ]))
        if verbose: print("dist to goal: ", dist2goal,
                          "homotopy: ", homotopy_rew,
                          "heading: ", heading,
                          "reward: ", reward)
        return reward


class GridworldContinuousNoneLREnv(GridworldContinuousMultiObjLREnv):
    def reset(self):
        self.check_point1 = False
        self.check_point2 = False
        
        self.world.reset()

        self.buildings = [
            #Building(Point(int(self.width/2.), int(self.height*3./5.)), Point(4,4), "gray80"),
            #Building(Point(int(self.width/2.), int(self.height*3./10.)), Point(4,4), "gray80")
            Building(Point(int(self.width-2), int(self.height*3./10.)), Point(4,int(self.height*3./5.)), "gray80"),
            Building(Point(int(2), int(self.height*3./10.)), Point(4,int(self.height*3./5.)), "gray80"),
        ]

        self.car = Car(Point(self.start[0], self.start[1]), np.pi/2., "blue")
        self.car.velocity = Point(0, 5)

        self.goal_obj = Goal(Point(self.goal[0], self.goal[1]), 0.0)

        for building in self.buildings:
            self.world.add(building)
        self.world.add(self.car)
        self.world.add(self.goal_obj)
        
        self.last_heading = np.pi / 2

        self.step_num = 0
        return self._get_obs()

class GridworldContinuousNoneRREnv(GridworldContinuousMultiObjRREnv):
    def reset(self):
        self.check_point1 = False
        self.check_point2 = False
        
        self.world.reset()

        self.buildings = [
            #Building(Point(int(self.width/2.), int(self.height*3./5.)), Point(4,4), "gray80"),
            #Building(Point(int(self.width/2.), int(self.height*3./10.)), Point(4,4), "gray80")
            Building(Point(int(self.width-2), int(self.height*3./10.)), Point(4,int(self.height*3./5.)), "gray80"),
            Building(Point(int(2), int(self.height*3./10.)), Point(4,int(self.height*3./5.)), "gray80"),
        ]

        self.car = Car(Point(self.start[0], self.start[1]), np.pi/2., "blue")
        self.car.velocity = Point(0, 5)

        self.goal_obj = Goal(Point(self.goal[0], self.goal[1]), 0.0)

        for building in self.buildings:
            self.world.add(building)
        self.world.add(self.car)
        self.world.add(self.goal_obj)
        
        self.last_heading = np.pi / 2

        self.step_num = 0
        return self._get_obs()

class GridworldContinuousNoneLLEnv(GridworldContinuousMultiObjLLEnv):
    def reset(self):
        self.check_point1 = False
        self.check_point2 = False
        
        self.world.reset()

        self.buildings = [
            #Building(Point(int(self.width/2.), int(self.height*3./5.)), Point(4,4), "gray80"),
            #Building(Point(int(self.width/2.), int(self.height*3./10.)), Point(4,4), "gray80")
            Building(Point(int(self.width-2), int(self.height*3./10.)), Point(4,int(self.height*3./5.)), "gray80"),
            Building(Point(int(2), int(self.height*3./10.)), Point(4,int(self.height*3./5.)), "gray80"),
        ]

        self.car = Car(Point(self.start[0], self.start[1]), np.pi/2., "blue")
        self.car.velocity = Point(0, 5)

        self.goal_obj = Goal(Point(self.goal[0], self.goal[1]), 0.0)

        for building in self.buildings:
            self.world.add(building)
        self.world.add(self.car)
        self.world.add(self.goal_obj)
        
        self.last_heading = np.pi / 2

        self.step_num = 0
        return self._get_obs()
