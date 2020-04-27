import os
from gym import utils
from fetch.fetch_envs import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')

JOINT_NAMES = ['robot0:slide0', 'robot0:slide1', 'robot0:slide2', 'robot0:torso_lift_joint', 'robot0:head_pan_joint',
               'robot0:head_tilt_joint', 'robot0:shoulder_pan_joint', 'robot0:shoulder_lift_joint',
               'robot0:upperarm_roll_joint', 'robot0:elbow_flex_joint', 'robot0:forearm_roll_joint',
               'robot0:wrist_flex_joint', 'robot0:wrist_roll_joint', 'robot0:r_gripper_finger_joint',
               'robot0:l_gripper_finger_joint']
INITIAL_QPOS = [4.04899887e-01,  4.80000000e-01,  2.79906896e-07, -2.10804408e-05,
        1.80448057e-10,  6.00288106e-02,  9.67580396e-03, -8.28231087e-01,
       -3.05625957e-03,  1.44397975e+00,  2.53423937e-03,  9.55099996e-01,
        5.96093593e-03,  1.97805133e-04,  7.15193042e-05]

class FetchEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        #initial_qpos = {
        #    'robot0:slide0': 0.4049,
        #    'robot0:slide1': 0.48,
        #    'robot0:slide2': 0.0,
        #}
        assert len(JOINT_NAMES) == len(INITIAL_QPOS)
        initial_qpos = {}
        for i,name in enumerate(JOINT_NAMES):
            initial_qpos[name] = INITIAL_QPOS[i]

        fetch_env.FetchEnv.__init__(
            #self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.01,
            #initial_qpos=initial_qpos, reward_type=reward_type)
            initial_qpos=initial_qpos, reward_type=None)
        utils.EzPickle.__init__(self)
