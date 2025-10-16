import copy
import time
from typing import OrderedDict

import gymnasium as gym
import numpy as np
import requests
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture
from franka_env.envs.franka_env import FrankaEnv
from franka_env.utils.rotations import euler_2_quat
from rb_env.envs.rb3_env import Rb3Env
from scipy.spatial.transform import Rotation


class Rb3UsbPickupInsertionEnv(Rb3Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = Rotation.from_quat(pose[3:], scalar_first=True).as_euler("xyz")

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = 1 * (
            np.clip(
                np.abs(euler[0]),
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )

        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat(scalar_first=True)

        return pose

    def check_reward(self):
        if self.curr_gripper_pos < 1 and self.curr_gripper_pos > 0.2:
            self.is_reward_ok = True
        else:
            self.is_reward_ok = False

class DualRb3ExampleEnv(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resetpos = np.concatenate(
            [
                self.config.RESET_POSE[:3],
                R.from_euler("xyz", self.config.RESET_POSE[3:]).as_quat(),
            ]
        )



class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]


        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info
