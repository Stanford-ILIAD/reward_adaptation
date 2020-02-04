import gym
from gym import spaces
import numpy as np
from minigrid.gym_minigrid.register import register
from minigrid.gym_minigrid.grid import *


class Gridworld(gym.Env):
    def __init__(self):
        self.grid_size = 5
        self.T = self.grid_size * 2 - 1
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([self.grid_size, self.grid_size]))
        self.start = np.array([0, 0])
        self.goal = np.array([self.grid_size - 1, self.grid_size - 1])
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

    def step(self, action, verbose=False):
        self.step_count += 1
        # print("step no: ", self.step_count)

        if verbose: print("state: ", self.S)
        ret = self._get_reward(verbose=verbose)
        if verbose: print("step: ", self.step_count)
        if verbose: print("ret: ", ret)

        # move agent
        dx, dy = self.moves[action]
        self.S = np.array([self.S[0] + dx, self.S[1] + dy])

        # Set bounds
        # self.S = max(0, self.S[0]), max(0, self.S[1])
        # self.S = (min(self.S[0], self.height - 1),
        #          min(self.S[1], self.width - 1))

        done = False
        if self.S[0] > self.grid_size - 1 or self.S[0] < 0 or self.S[1] > self.grid_size - 1 or self.S[1] < 0:
            # if verbose: print("done: out of bounds")
            done = True
        if self.step_count >= self.T:
            # if verbose: print("done: step count")
            done = True
        # if (self.S == self.goal).all():
        #    done = True
        is_collision = np.array([(self.S == obstacle).all() for obstacle in self.obstacles])
        if is_collision.any():
            # if verbose: print("done: collision")
            done = True
        return self._get_obs(), ret, done, {}

    def _get_obs(self):
        # return np.concatendate(([self.S, self.start, self.goal], self.obstacles))
        return self.S

    def _get_reward(self, verbose=False):
        max_dist = 8.0
        max_rew = 8.0 + 7 + 6 + 5 + 4 + 3 + 2 + 1
        dist_goal = np.linalg.norm((self.S - self.goal), 1)
        is_collision = np.all([(self.S == obstacle).all() for obstacle in self.obstacles])
        obstacle_penalty = -1.0 if np.any(is_collision) else 0.0
        dist_bottom = self.S[1] - self.grid_size
        dist_right = self.S[0] - self.grid_size

        reward = -(dist_goal / max_rew) + (2.0 / self.T) + obstacle_penalty
        reward += dist_bottom/50.
        if (self.S == self.goal).all():
            reward += 10
        if (self.S == np.array([0,4])).all() and self.step_count <=5:
            reward += 1
        if verbose: print("dist2goal: ", dist_goal, " reward: ", reward, "pref: ", dist_bottom/50.)  # , "pos: ", self.S)
        return reward

    def reset(self):
        # print("\nreset")
        self.S = self.start
        self.obstacles.append(np.array([int(self.grid_size / 2), int(self.grid_size / 2)]))
        self.step_count = 0
        self.mission = "go to the goal while avoiding the obstacle"
        self._gen_grid(self.grid_size, self.grid_size)
        return self.S

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        self.grid.put_obj(Goal(), self.goal[0], self.goal[1])
        for obstacle in self.obstacles:
            self.grid.put_obj(Lava(), obstacle[0], obstacle[1])

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

    def print_value_all(self, q_table):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for action in range(0, 4):
                    state = [i, j]
                    if str(state) in q_table.keys():
                        temp = q_table[str(state)][action]
                        print(j, i, round(temp, 2), action)


register(
    id='Gridworld-v0',
    entry_point='gym_minigrid.envs:Gridworld'
)
