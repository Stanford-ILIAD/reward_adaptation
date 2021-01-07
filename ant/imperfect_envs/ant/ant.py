from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/ant.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        self.step_id = 0
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

class HomotopyDirectUpAntEnv(AntEnv):
    def __init__(self):
        self.step_id = 0
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/homotopy_ant.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self.step_id += 1
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.3 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()

        joint_angle = ob[5:13]
        homotopy_rew = 0
        if joint_angle[0] > 0:
            homotopy_rew += 1
        else:
            homotopy_rew -= 1
        if -0.2 < joint_angle[0] < 0.2 and self.step_id > 1:
            homotopy_rew -= 1000
        if state[2] < 0.3 or state[2] > 1:
            homotopy_rew -= 2000
        reward += homotopy_rew
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            reward_homotopy=homotopy_rew)

class HomotopyDirectDownAntEnv(AntEnv):
    def __init__(self):
        self.step_id = 0
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/homotopy_ant.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self.step_id += 1
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.3 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()

        joint_angle = ob[5:13]
        homotopy_rew = 0
        if joint_angle[0] < 0:
            homotopy_rew += 1
        else:
            homotopy_rew -= 1
        if -0.2 < joint_angle[0] < 0.2 and self.step_id > 1:
            homotopy_rew -= 1000
        if state[2] < 0.3 or state[2] > 1:
            homotopy_rew -= 2000
        reward += homotopy_rew
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            reward_homotopy=homotopy_rew)

class HomotopyRelaxUpAntEnv(AntEnv):
    def __init__(self):
        self.step_id = 0
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/homotopy_ant.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self.step_id += 1
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.3 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()

        joint_angle = ob[5:13]
        homotopy_rew = 0
        if joint_angle[0] > 0:
            homotopy_rew += 1
        else:
            homotopy_rew -= 1
        #if -0.2 < joint_angle[0] < 0.2 and self.step_id > 1:
        #    homotopy_rew -= 1000
        if state[2] < 0.3 or state[2] > 1:
            homotopy_rew -= 2000
        reward += homotopy_rew
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            reward_homotopy=homotopy_rew)

class HomotopyRelaxDownAntEnv(AntEnv):
    def __init__(self):
        self.step_id = 0
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/homotopy_ant.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self.step_id += 1
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.3 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()

        joint_angle = ob[5:13]
        homotopy_rew = 0
        if joint_angle[0] < 0:
            homotopy_rew += 1
        else:
            homotopy_rew -= 1
        #if -0.2 < joint_angle[0] < 0.2 and self.step_id > 1:
        #    homotopy_rew -= 1000
        if state[2] < 0.3 or state[2] > 1:
            homotopy_rew -= 2000
        reward += homotopy_rew
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            reward_homotopy=homotopy_rew)

class HomotopyRewardUpAntEnv(AntEnv):
    def __init__(self):
        self.ratio = 1.
        self.step_id = 0
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/homotopy_ant.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def set_param(self, ratio):
        self.ratio = ratio

    def step(self, a):
        self.step_id += 1
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.3 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()

        joint_angle = ob[5:13]
        homotopy_rew = 0
        if joint_angle[0] > 0:
            homotopy_rew += 1
        else:
            homotopy_rew -= 1
        if -0.2 < joint_angle[0] < 0.2 and self.step_id > 1:
            homotopy_rew -= 1000*self.ratio
        if state[2] < 0.3 or state[2] > 1:
            homotopy_rew -= 2000
        reward += homotopy_rew
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            reward_homotopy=homotopy_rew)

class HomotopyRewardDownAntEnv(AntEnv):
    def __init__(self):
        self.ratio = 1.
        self.step_id = 0
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/homotopy_ant.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def set_param(self, ratio):
        self.ratio = ratio

    def step(self, a):
        self.step_id += 1
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.3 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()

        joint_angle = ob[5:13]
        homotopy_rew = 0
        if joint_angle[0] < 0:
            homotopy_rew += 1
        else:
            homotopy_rew -= 1
        if -0.2 < joint_angle[0] < 0.2 and self.step_id > 1:
            homotopy_rew -= 1000*self.ratio
        if state[2] < 0.3 or state[2] > 1:
            homotopy_rew -= 2000
        reward += homotopy_rew
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            reward_homotopy=homotopy_rew)

class HomotopyObstacleUpAntEnv(AntEnv):
    def __init__(self):
        self.ratio = 1.
        self.step_id = 0
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/homotopy_ant.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def set_param(self, ratio):
        self.ratio = ratio

    def step(self, a):
        self.step_id += 1
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.3 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()

        joint_angle = ob[5:13]
        homotopy_rew = 0
        if joint_angle[0] > 0:
            homotopy_rew += 1
        else:
            homotopy_rew -= 1
        if -0.2*self.ratio < joint_angle[0] < 0.2*self.ratio and self.step_id > 1:
            homotopy_rew -= 1000
        if state[2] < 0.3 or state[2] > 1:
            homotopy_rew -= 2000
        reward += homotopy_rew
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            reward_homotopy=homotopy_rew)

class HomotopyObstacleDownAntEnv(AntEnv):
    def __init__(self):
        self.ratio = 1.
        self.step_id = 0
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/homotopy_ant.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def set_param(self, ratio):
        self.ratio = ratio

    def step(self, a):
        self.step_id += 1
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.3 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()

        joint_angle = ob[5:13]
        homotopy_rew = 0
        if joint_angle[0] < 0:
            homotopy_rew += 1
        else:
            homotopy_rew -= 1
        if -0.2*self.ratio < joint_angle[0] < 0.2*self.ratio and self.step_id > 1:
            homotopy_rew -= 1000
        if state[2] < 0.3 or state[2] > 1:
            homotopy_rew -= 2000
        reward += homotopy_rew
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            reward_homotopy=homotopy_rew)

