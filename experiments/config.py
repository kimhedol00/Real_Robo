# config.py

# ==========================================================
# 1. 공통 Import 섹션 (양팔/단일 팔 모두 사용)
# ==========================================================
import os
import re
import yaml
import importlib
from collections import OrderedDict

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import List, Any, Union
from typing_extensions import Unpack

import time

import cv2
# from pynput import keyboard


import numpy as np
import jax
import jax.numpy as jnp
NetworkMaskTree = dict[str, Union["NetworkMaskTree", bool]]

# 모든 실제 로봇 환경에서 공통으로 사용
try:
    import rclpy
except ImportError:
    print("Warning: Some ROS/Robot specific libraries not found. Fake mode might be required.")
    pass
from franka_env.camera.realsense_camera import RSCapture
from franka_env.camera.usb_capture import USBCapture
from rb_env.envs.rb3_env import DefaultEnvConfig
from dual_rb3_mujoco_ros2.dual_rb3_mujoco_node import DualRb3MujocoNode

# 공통 SERL 래퍼
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

from rb_env.envs.wrappers import (
    DualQuat2EulerWrapper,
    MultiCameraBinaryRewardClassifierWrapper,
    DualGripperPenaltyWrapper,
    SpacemouseIntervention,
    DualSpacemouseIntervention,
    DualSpacemouseIntervention2,
    Quat2EulerWrapper
)

from rb_env.envs.dual_rb3_env import DualRb3Env
from rb_env.envs.relative_env import RelativeFrame
from rb_env.envs.relative_env import DualRelativeFrame

from experiments.wrapper import Rb3UsbPickupInsertionEnv, DualRb3ExampleEnv

from serl_launcher.networks.reward_classifier import load_classifier_func


from pynput import keyboard
success_triggered = False

def trigger_success():
    global success_triggered
    success_triggered = True

def on_press(key):
    try:
        if key == keyboard.Key.space:
            trigger_success()
        elif key.char == 's':
            trigger_success()
            # print("Skip triggered!")


    except AttributeError:
        pass


def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print('start listener')

start_keyboard_listener()




class DefaultTrainingConfig:
    """Default training configuration."""

    agent: str = "drq"
    max_traj_length: int = 100
    batch_size: int = 256
    cta_ratio: int = 2
    discount: float = 0.97
    use_proprio: bool = False  # change classifier type
    
    # target entropy = -action_dim * target_entropy_scale  
    # 0.5 was default (-act_dim/2) in HIL-SERL.
    # Original SAC used 1.0, but for smaller std output, use larger value.
    # too large entropy scale can lead to collapsing into deterministic policy,
    # but is it that relavant in HIL setting?
    target_entropy_scale: float = 0.5  
    reward_scale: float = 1.0
    # done_override: the ratio to override the done flag of each transition.
    # It does not bias the value function at all.
    # However, it obviously discourages value bootstrapping (= slower learning)
    # so I recommend using a small value. (~0.1?)
    done_override_ratio: float = 0.0
    reset_strategies: tuple["ResetStrategy"] = tuple()

    max_steps: int = 1000000
    replay_buffer_capacity: int = 200000
    n_step_return: int = 1

    random_steps: int = 0
    training_starts: int = 100
    steps_per_update: int = 50

    log_period: int = 10
    eval_period: int = 2000

    # "resnet" for ResNet10 from scratch and "resnet-pretrained" for frozen ResNet10 with pretrained weights
    encoder_type: str = "resnet-pretrained"
    mlp_cls: str = "MLP"
    demo_path: str = None
    checkpoint_period: int = 0
    buffer_period: int = 0

    eval_checkpoint_step: int = 0
    eval_n_trajs: int = 5

    image_keys: List[str] = None
    classifier_keys: List[str] = None
    proprio_keys: List[str] = None

    # "single-arm-learned-gripper", "dual-arm-learned-gripper" for with learned gripper,
    # "single-arm-fixed-gripper", "dual-arm-fixed-gripper" for without learned gripper (i.e. pregrasped)
    setup_mode: str = "single-arm-fixed-gripper"
    
    def __init__(self):
        if self.n_step_return != 1:
            raise NotImplementedError("Not yet supported")

    @abstractmethod
    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        raise NotImplementedError

    @abstractmethod
    def process_demos(self, demo):
        raise NotImplementedError
    
    
    use_curriculum_reward = False
    
    def get_curriculum_reward_fn(self):
        return partial(self._calculate_curriculum_reward)
    
    @staticmethod
    def _calculate_curriculum_reward(
        state: dict[str, np.ndarray | jax.Array],
        action: np.ndarray | jax.Array,
        next_state: dict[str, np.ndarray | jax.Array],
        train_steps: int,
        *args, **kwargs
    ) -> float | None:  # actually 0D array?
        """
        Calculates curriculum reward given a single transition.
        
        states are decomposed into proprio key dict.
        shapes: {key: value[#Frame Stacks, Dim]}  
        e.g. {'tcp_pose': Array[1, 6], 'tcp_force': Array[1, 3], 'gripper_pose': Array[1, 1], ...}
        
        action has 1D shape of HIL-SERL default action space.  e.g. Array[6]
        
        This function will be wrapped afterward for batched processing,
        so just write for a single transition.
        
        returns:
            curriculum_rewards: reward calculated from a transition
            info: info.
        """
        if isinstance(action, jax.Array):
            # Use jnp only. this if-else is not traced after JIT.
            return None  # * jnp.clip((train_steps - curriculum_starts) / curriculum_steps, 0.0, 1.0)
        else:
            # Use np only.
            return None  # * np.clip((train_steps - curriculum_starts) / curriculum_steps, 0.0, 1.0)

    
    @dataclass(frozen=True)
    class ResetStrategy:
        """
        ... ask jihun if you need help
        """
        name: str = "reset_except_encoder"
        steps_every: int = None
        pretrained_encoder: bool = False  # resetting pretrained encoder can be done toward pretrained weights.
        shrink_factor: float = 0.0
        perturb_factor: float = 1.0
        subkeys: NetworkMaskTree = field(default_factory=lambda: {
            'modules_actor': {
                'network': True,  # MLP
                'Dense_0': True,  # mean
                'Dense_1': True,  # std
                # 'encoder': False
            },
            'modules_critic': True,  # The whole thing
            # 'modules_grasp_critic': False,
            # 'modules_temperature': False
        })  # be careful NOT to reset pretrained encoder if frozen
        
        def _construct_subkey_tree(self, agent) -> NetworkMaskTree:
            def tree_wildcard_fill(orig_tree, wildcard_tree, key_cond=lambda key: True):
                if isinstance(wildcard_tree, dict):
                    for k in wildcard_tree:
                        tree_wildcard_fill(orig_tree[k], wildcard_tree[k])
                elif wildcard_tree == True:
                    for k, v in orig_tree.items():
                        if isinstance(v, dict):
                            tree_wildcard_fill(orig_tree[k], wildcard_tree)
                        elif key_cond(k):
                            orig_tree[k] = True
            return tree_wildcard_fill(jax.tree.map(lambda _: False, (agent.state.params)), self.subkeys)
        
        Agent = Any  # for your type hint ;)
        def do_your_reset(self, agent, batch, train_step: int) -> Agent:
            if train_step > 0 and train_step % self.steps_every == 0:
                if self.pretrained_encoder:
                    from serl_launcher.utils.train_utils import load_resnet10_params

                    return load_resnet10_params(
                        agent, agent.config["image_keys"], shrink_factor=self.shrink_factor, perturb_factor=self.perturb_factor
                    )
                else:
                    return agent.shrink_and_perturb(
                        jax.tree.map(lambda x: x[:1], batch['observations']),  # [:1] for compressing batch dimension
                        batch['actions'][:1], 
                        shrink_factor=self.shrink_factor, 
                        perturb_factor=self.perturb_factor, 
                        subkeys=self._construct_subkey_tree()  # dont worry. super fast.
                    )
            else:
                return agent  # no-op

class LeftEnvConfig(DefaultEnvConfig):

    # From ENV file
    XYZ_CLIP_LOW = (-0.08, -0.165, -0.013, 0, 0, 0)
    XYZ_CLIP_HIGH = (0.18, 0.308, 0.08, 0, 0, 0)

    GET_POSE = np.array([0.37098, -0.017392, 0.13, -np.pi, 0, np.pi / 2])
    RESET_POSE = np.array([0.346, -0.32681, 0.13225, np.pi, 0, np.pi / 2])
    RANDOM_RESET = False
    DISPLAY_IMAGE = True
    RANDOM_XY_RANGE = 0.2
    RANDOM_RZ_RANGE = 0.0
    ABS_POSE_LIMIT_LOW = RESET_POSE + XYZ_CLIP_LOW + (-0.015, -0.020, -0.115, -0.15, -0.15, -0.15)
    ABS_POSE_LIMIT_HIGH = RESET_POSE + XYZ_CLIP_HIGH + (0.1, 0.1, 0.000, 0.15, 0.15, 0.15)
    MAX_EPISODE_LENGTH = 1500
    
    # Can be Changed by demo
    ACTION_SCALE = np.array([0.01, 0.01, 1])
    

    # Fixed
    ROBOT_PREFIX = "left"

class RightEnvConfig(DefaultEnvConfig):
    
    # From base.yaml or dual_base.yaml
    CAMERAS = {
        "head_cam": {
            "serial_number": "213522254901",
            "dim": (1280, 720),
            "exposure": 40000,
        },
        "wrist_cam": {
            "cam_id": "usb-0:5.1.3:1.0",
            "dim": (640, 480),
            "strict_device_filter": True,
        },
        "front_cam": {
            "cam_id": "usb-0:5.1.4:1.0",
            "dim": (640, 480),
            "strict_device_filter": True,
        }
    }
    IMAGE_CROP = {
        "head_cam": lambda img: np.rot90(img[:, :], 2),
        "wrist_cam": lambda img: np.rot90(img, 2),
        "front_cam": lambda img: np.rot90(img, 2),
    }

    
    # From Env file
    XYZ_CLIP_LOW = (-0.08, -0.165, -0.013, 0, 0, 0)
    XYZ_CLIP_HIGH = (0.18, 0.308, 0.08, 0, 0, 0)
    GET_POSE = np.array([0.37098, -0.017392, 0.13, -np.pi, 0, np.pi / 2])
    RESET_POSE = np.array([0.346, -0.32681, 0.13225, np.pi, 0, np.pi / 2])
    RANDOM_RESET = False
    DISPLAY_IMAGE = True
    RANDOM_XY_RANGE = 0.2
    RANDOM_RZ_RANGE = 0.0
    ABS_POSE_LIMIT_LOW = RESET_POSE + XYZ_CLIP_LOW + (-0.015, -0.020, -0.115, -0.15, -0.15, -0.15)
    ABS_POSE_LIMIT_HIGH = RESET_POSE + XYZ_CLIP_HIGH + (0.1, 0.1, 0.000, 0.15, 0.15, 0.15)
    MAX_EPISODE_LENGTH = 1500

    # Can be Changed by demo
    ACTION_SCALE = np.array([0.01, 0.01, 1])

    # Fixed 
    ROBOT_PREFIX = "right"

class TrainConfig(DefaultTrainingConfig):
    # from default.yaml or dual_default.yaml
    image_keys = ["head_cam", "wrist_cam", "front_cam"]
    classifier_keys = ["head_cam", "wrist_cam", "front_cam"]
    proprio_keys = [
        'gripper_pose',   # 19
        'tcp_force',      # 20 ~ 22
        'tcp_pose',       # 23 ~ 28
        'tcp_torque',     # 29 ~ 31
        'tcp_vel',        # 32 ~ 37 
    ]
    setup_mode = "single-arm-learned-gripper"
    arm_type = "right"
    
    # Fixed
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    encoder_type = "resnet-pretrained"
    use_proprio = False
    hz = 10

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        
        # Fixed 
        rclpy.init()
        head_cam = RSCapture(
            "head_cam", 
            **RightEnvConfig.CAMERAS["head_cam"]
        )
        right_wrist_cam = USBCapture(
            "right_wrist_cam", 
            **RightEnvConfig.CAMERAS["wrist_cam"]
        )
        front_cam = USBCapture(
            "front_cam", 
            **RightEnvConfig.CAMERAS["front_cam"]
        )
        cameras = OrderedDict(head_cam=head_cam, wrist_cam=right_wrist_cam, front_cam=front_cam)

        # left_wrist_cam = USBCapture(
        #     "left_wrist_cam", 
        #     **LeftEnvConfig.CAMERAS["wrist_cam"]
        # )
        # left_cameras = OrderedDict(wrist_cam=left_wrist_cam)

        # left_wrist_cam.start()
        head_cam.start()
        right_wrist_cam.start()     
        front_cam.start()     
        
        dual_rb3_mujoco_node = DualRb3MujocoNode("", is_sim=False, is_usb_gripper=[False, True])

        # Can be Changed by armtype
        if self.arm_type == "dual":
            right_env = Rb3UsbPickupInsertionEnv(
                cameras=cameras,
                robot_node=dual_rb3_mujoco_node,
                fake_env=fake_env,
                save_video=save_video,
                config=RightEnvConfig,
                hz=self.hz
            )
            left_env = Rb3UsbPickupInsertionEnv(
                cameras=cameras,
                robot_node=dual_rb3_mujoco_node,
                fake_env=fake_env,
                save_video=save_video,
                config=LeftEnvConfig,
                hz=self.hz
            )
            env = DualRb3Env(left_env, right_env, display_images=True)
            env = DualSpacemouseIntervention2(env)
            env = DualRelativeFrame(env)
            env = DualQuat2EulerWrapper(env)
            env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
            env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)


        if self.arm_type == "right":
            env = Rb3UsbPickupInsertionEnv(
                cameras=cameras,
                robot_node=dual_rb3_mujoco_node,
                fake_env=fake_env,
                save_video=save_video,
                config=RightEnvConfig,
                hz=self.hz
            )
        elif self.arm_type == "left":
            env = Rb3UsbPickupInsertionEnv(
                cameras=cameras,
                robot_node=dual_rb3_mujoco_node,
                fake_env=fake_env,
                save_video=save_video,
                config=LeftEnvConfig,
                hz=self.hz
            )

        if self.arm_type != "dual":
            env = SpacemouseIntervention(env)
            env = RelativeFrame(env)
            env = Quat2EulerWrapper(env)
            env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
            env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        def reward_func(obs):
            global success_triggered
            if success_triggered:
                success_triggered=False
                return int(1)
            else:
                return 0

        env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)


        return env

def _merge_configs(base, override):
    """딕셔너리를 재귀적으로 병합합니다."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = _merge_configs(base[key], value)
        else:
            base[key] = value
    return base

def load_and_apply_config(env_name, arm_type, overrides=None):
    config_dir = os.path.join(os.path.dirname(__file__), 'configs')
    if overrides is None: overrides = {}
    
    # 1. 기본 설정 파일 로드
    base_config_file = 'dual_base.yaml' if arm_type == 'dual' else 'base.yaml'
    with open(os.path.join(config_dir, base_config_file), 'r') as f:
        config = yaml.safe_load(f)

    # 2. 실험별 설정 덮어쓰기
    try:
        with open(os.path.join(config_dir, f'{env_name}.yaml'), 'r') as f:
            exp_config = yaml.safe_load(f)
            config = _merge_configs(config, exp_config)
    except FileNotFoundError: pass

    # ✨ 3. CLI 인자(overrides)로 최종 덮어쓰기 ✨
    # action_scale이 들어오면 양팔 모두에 적용되도록 구조화
    if 'action_scale' in overrides:
        scale = overrides.pop('action_scale')
        if 'env' not in config: config['env'] = {}
        if 'action_scales' not in config['env']: config['env']['action_scales'] = {}
        config['env']['action_scales']['right'] = scale
        config['env']['action_scales']['left'] = scale

    # 나머지 overrides를 config에 병합
    config = _merge_configs(config, overrides)

    # 3. TrainConfig 클래스 속성 덮어쓰기
    for key, value in config.get('train', {}).items(): setattr(TrainConfig, key, value)
    for key in ['env_class']:
        if key in config: setattr(TrainConfig, key, config[key])
    setattr(TrainConfig, 'arm_type', arm_type)

    # 4. EnvConfig 클래스 속성 덮어쓰기
    env_data = config.get('env', {})
    for side, ConfigClass in [('right', RightEnvConfig), ('left', LeftEnvConfig)]:
        # 공통 속성
        setattr(ConfigClass, 'MAX_EPISODE_LENGTH', env_data.get('max_episode_length'))
        setattr(ConfigClass, 'RANDOM_RESET', env_data.get('random_reset'))
        setattr(ConfigClass, 'DISPLAY_IMAGE', env_data.get('display_image'))

        # IMAGE_CROP (lambda 함수 처리)
        image_crop_dict = {}
        for key, val_str in env_data.get('image_crop', {}).items():
            image_crop_dict[key] = eval(val_str, {"np": np})
        setattr(ConfigClass, 'IMAGE_CROP', image_crop_dict)

        # side별 특정 속성
        if 'poses' in env_data and side in env_data['poses']:
            setattr(ConfigClass, 'RESET_POSE', np.array(env_data['poses'][side]['reset']))
            setattr(ConfigClass, 'GET_POSE', np.array(env_data['poses'][side]['get']))
        if 'action_scales' in env_data and side in env_data['action_scales']:
            setattr(ConfigClass, 'ACTION_SCALE', np.array(env_data['action_scales'][side]))
        
        # 안전 영역 동적 계산
        if getattr(ConfigClass, 'RESET_POSE') is not None and 'xyz_clips' in env_data and side in env_data['xyz_clips']:
            setattr(ConfigClass, 'ABS_POSE_LIMIT_LOW', getattr(ConfigClass, 'RESET_POSE') + np.array(env_data['xyz_clips'][side]['low']) + np.array(env_data['safety_offsets'][side]['low']))
            setattr(ConfigClass, 'ABS_POSE_LIMIT_HIGH', getattr(ConfigClass, 'RESET_POSE') + np.array(env_data['xyz_clips'][side]['high']) + np.array(env_data['safety_offsets'][side]['high']))

    return TrainConfig()
