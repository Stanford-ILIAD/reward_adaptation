import gym
from gym import spaces
import numpy as np
from minigrid.gym_minigrid.register import register
from minigrid.gym_minigrid.grid import *

class Gridworld(gym.Env):
    def __init__(self):
        self.grid_size = 5
        self.T = self.grid_size * 2 - 2
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=np.array([0,0]), high=np.array([self.grid_size, self.grid_size]))
        self.start = np.array([0, 0])
        self.goal = np.array([self.grid_size-1, self.grid_size-1])
        self.obstacles = []
        self.window = None
        self.moves = {
            0: [-1, 0],  # up
            1: [0, 1],  # right
            2: [1, 0],  # down
            3: [0, -1],  # left
        }

        # begin in start state
        self.reset()

    def step(self, action):
        self.step_count += 1
        #print("step no: ", self.step_count)
        # move agent
        dx, dy = self.moves[action]
        #print("dx, dy: ", dx, dy)
        #print("curr state: ", self.S)
        self.S = np.array([self.S[0] + dx, self.S[1] + dy])
        #print("new state: ", self.S)

        # Set bounds
        # self.S = max(0, self.S[0]), max(0, self.S[1])
        # self.S = (min(self.S[0], self.height - 1),
        #          min(self.S[1], self.width - 1))

        done = False
        if self.S[0] > self.grid_size - 1 or self.S[0] < 0 or self.S[1] > self.grid_size - 1 or self.S[1] < 0:
            done = True
        if self.step_count >= self.T:
            done = True
        if (self.S == self.goal).all():
            done = True
        return self._get_obs(), self._get_reward(), done, {}

    def _get_obs(self):
        # return np.concatendate(([self.S, self.start, self.goal], self.obstacles))
        return self.S

    def _get_reward(self):
        max_dist = 9.0
        dist_goal = np.linalg.norm((self.S - self.goal), 1)
        is_collision = [self.S == obstacle for obstacle in self.obstacles]
        obstacle_penalty = -10 if np.any(is_collision) else 0
        dist_upper_right = np.linalg.norm((self.S - np.array([self.grid_size, 0])), 1)
        #reward = -dist_goal - dist_upper_right + obstacle_penalty
        reward = -(dist_goal/max_dist) + 1.0
        #reward = 1.0 if (self.S == self.goal).all() else 0.0
        reward = reward/self.T
        #print("dist2goal: ", dist_goal, " reward: ", reward)
        return reward

    def reset(self):
        #print("\nreset")
        self.S = self.start
        self.obstacles.append(np.array([3, 3]))
        self.step_count = 0
        self.mission = "go to the goal while avoiding the obstacle"
        self._gen_grid(self.grid_size, self.grid_size)
        return self.S

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        #self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.grid.put_obj(Goal(), self.goal[0], self.goal[1])

    def render(self, close=False, highlight=True, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """
        if close:
            if self.window:
                self.window.close()
            return

        if not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('gym_minigrid')
            self.window.show(block=False)

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.S,
        )

        self.window.show_img(img)
        self.window.set_caption(self.mission)

        return img


register(
    id='Gridworld-v0',
    entry_point='gym_minigrid.envs:Gridworld'
)