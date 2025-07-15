# File: tmrl/custom/tm/tm_gym_interfaces.py
# rtgym interfaces for Trackmania with enhanced reward support

# standard library imports
import logging
import time
from collections import deque

# third-party imports
import cv2
import gymnasium.spaces as spaces
import numpy as np

# third-party imports
from rtgym import RealTimeGymInterface

# local imports
import tmrl.config.config_constants as cfg
from tmrl.custom.tm.utils.compute_reward import RewardFunction, EnhancedRewardFunction, SpeedFocusedRewardFunction, CheckpointRewardFunction
from tmrl.custom.tm.utils.control_gamepad import control_gamepad, gamepad_reset, gamepad_close_finish_pop_up_tm20
from tmrl.custom.tm.utils.control_mouse import mouse_close_finish_pop_up_tm20
from tmrl.custom.tm.utils.control_keyboard import apply_control, keyres
from tmrl.custom.tm.utils.window import WindowInterface
from tmrl.custom.tm.utils.tools import Lidar, TM2020OpenPlanetClient, save_ghost

# Globals ==============================================================================================================

CHECK_FORWARD = 500  # this allows (and rewards) 50m cuts


# Interface for Trackmania 2020 ========================================================================================

class TM2020Interface(RealTimeGymInterface):
    """
    This is the API needed for the algorithm to control TrackMania 2020
    """
    def __init__(self,
                 img_hist_len: int = 4,
                 gamepad: bool = True,
                 save_replays: bool = False,
                 grayscale: bool = True,
                 resize_to=(64, 64),
                 reward_type: str = "original",
                 reward_params: dict = None):
        """
        Base rtgym interface for TrackMania 2020 (Full environment)

        Args:
            img_hist_len: int: history of images that are part of observations
            gamepad: bool: whether to use a virtual gamepad for control
            save_replays: bool: whether to save TrackMania replays on successful episodes
            grayscale: bool: whether to output grayscale images or color images
            resize_to: Tuple[int, int]: resize output images to this (width, height)
            reward_type: str: type of reward function to use
                - "original": Original trajectory following (default)
                - "enhanced": Multi-component balanced reward
                - "speed": Speed-focused aggressive reward
                - "checkpoint": Checkpoint-based progression reward
            reward_params: dict: optional parameters to override reward function defaults
        """
        self.last_time = None
        self.img_hist_len = img_hist_len
        self.img_hist = None
        self.img = None
        self.reward_function = None
        self.client = None
        self.gamepad = gamepad
        self.j = None
        self.window_interface = None
        self.small_window = None
        self.save_replays = save_replays
        self.grayscale = grayscale
        self.resize_to = resize_to
        self.finish_reward = cfg.REWARD_CONFIG['END_OF_TRACK']
        self.constant_penalty = cfg.REWARD_CONFIG['CONSTANT_PENALTY']
        self.reward_type = reward_type
        self.reward_params = reward_params if reward_params is not None else {}
        self.last_action = None  # Track actions for enhanced rewards

        self.initialized = False

    def initialize_common(self):
        if self.gamepad:
            import vgamepad as vg
            self.j = vg.VX360Gamepad()
            logging.debug(" virtual joystick in use")
        self.window_interface = WindowInterface("Trackmania")
        self.window_interface.move_and_resize()
        self.last_time = time.time()
        self.img_hist = deque(maxlen=self.img_hist_len)
        self.img = None

        print(self.reward_type)
        
        # Initialize the chosen reward function
        if self.reward_type == "enhanced":
            # Enhanced multi-component reward
            self.reward_function = EnhancedRewardFunction(
                reward_data_path=cfg.REWARD_PATH,
                nb_obs_forward=self.reward_params.get('nb_obs_forward', 15),
                nb_obs_backward=self.reward_params.get('nb_obs_backward', 10),
                nb_zero_rew_before_failure=self.reward_params.get('nb_zero_rew_before_failure', 15),
                min_nb_steps_before_failure=self.reward_params.get('min_nb_steps_before_failure', int(3.5 * 20)),
                max_dist_from_traj=self.reward_params.get('max_dist_from_traj', 80.0),
                speed_reward_weight=self.reward_params.get('speed_reward_weight', 0.3),
                progress_reward_weight=self.reward_params.get('progress_reward_weight', 0.4),
                smooth_control_weight=self.reward_params.get('smooth_control_weight', 0.2),
                checkpoint_reward_weight=self.reward_params.get('checkpoint_reward_weight', 0.1)
            )
        elif self.reward_type == "speed":
            # Speed-focused reward
            self.reward_function = SpeedFocusedRewardFunction(
                reward_data_path=cfg.REWARD_PATH,
                nb_obs_forward=self.reward_params.get('nb_obs_forward', 20),
                nb_obs_backward=self.reward_params.get('nb_obs_backward', 5),
                nb_zero_rew_before_failure=self.reward_params.get('nb_zero_rew_before_failure', 30),
                min_nb_steps_before_failure=self.reward_params.get('min_nb_steps_before_failure', int(3 * 20)),
                max_dist_from_traj=self.reward_params.get('max_dist_from_traj', 100.0),
                target_speed=self.reward_params.get('target_speed', 180.0),
                min_speed=self.reward_params.get('min_speed', 30.0),
                speed_exp=self.reward_params.get('speed_exp', 2.0)
            )
        elif self.reward_type == "checkpoint":
            # Checkpoint-based reward
            self.reward_function = CheckpointRewardFunction(
                reward_data_path=cfg.REWARD_PATH,
                num_checkpoints=self.reward_params.get('num_checkpoints', 20),
                checkpoint_reward=self.reward_params.get('checkpoint_reward', 10.0),
                time_bonus_per_checkpoint=self.reward_params.get('time_bonus_per_checkpoint', 5.0),
                max_time_per_checkpoint=self.reward_params.get('max_time_per_checkpoint', 200),
                min_speed_bonus=self.reward_params.get('min_speed_bonus', 50.0),
                exploration_bonus=self.reward_params.get('exploration_bonus', 0.1)
            )
        else:
            # Default to original reward function
            self.reward_function = RewardFunction(
                reward_data_path=cfg.REWARD_PATH,
                nb_obs_forward=self.reward_params.get('nb_obs_forward', cfg.REWARD_CONFIG['CHECK_FORWARD']),
                nb_obs_backward=self.reward_params.get('nb_obs_backward', cfg.REWARD_CONFIG['CHECK_BACKWARD']),
                nb_zero_rew_before_failure=self.reward_params.get('nb_zero_rew_before_failure', cfg.REWARD_CONFIG['FAILURE_COUNTDOWN']),
                min_nb_steps_before_failure=self.reward_params.get('min_nb_steps_before_failure', cfg.REWARD_CONFIG['MIN_STEPS']),
                max_dist_from_traj=self.reward_params.get('max_dist_from_traj', cfg.REWARD_CONFIG['MAX_STRAY'])
            )
            
        self.client = TM2020OpenPlanetClient()

    def initialize(self):
        self.initialize_common()
        self.small_window = True
        self.initialized = True

    def send_control(self, control):
        """
        Non-blocking function
        Applies the action given by the RL policy
        If control is None, does nothing (e.g. to record)
        Args:
            control: np.array: [forward,backward,right,left]
        """
        # Track action for enhanced rewards
        if control is not None:
            self.last_action = control.copy()
            
        if self.gamepad:
            if control is not None:
                control_gamepad(self.j, control)
        else:
            if control is not None:
                actions = []
                if control[0] > 0:
                    actions.append('f')
                if control[1] > 0:
                    actions.append('b')
                if control[2] > 0.5:
                    actions.append('r')
                elif control[2] < -0.5:
                    actions.append('l')
                apply_control(actions)

    def grab_data_and_img(self):
        img = self.window_interface.screenshot()[:, :, :3]  # BGR ordering
        if self.resize_to is not None:  # cv2.resize takes dim as (width, height)
            img = cv2.resize(img, self.resize_to)
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = img[:, :, ::-1]  # reversed view for numpy RGB convention
        data = self.client.retrieve_data()
        self.img = img  # for render()
        return data, img

    def reset_race(self):
        if self.gamepad:
            gamepad_reset(self.j)
        else:
            keyres()

    def reset_common(self):
        if not self.initialized:
            self.initialize()
        self.send_control(self.get_default_action())
        self.reset_race()
        time_sleep = max(0, cfg.SLEEP_TIME_AT_RESET - 0.1) if self.gamepad else cfg.SLEEP_TIME_AT_RESET
        time.sleep(time_sleep)  # must be long enough for image to be refreshed
        self.last_action = None  # Reset action tracking

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        data, img = self.grab_data_and_img()
        speed = np.array([
            data[0],
        ], dtype='float32')
        gear = np.array([
            data[9],
        ], dtype='float32')
        rpm = np.array([
            data[10],
        ], dtype='float32')
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [speed, gear, rpm, imgs]
        self.reward_function.reset()
        return obs, {}

    def close_finish_pop_up_tm20(self):
        if self.gamepad:
            gamepad_close_finish_pop_up_tm20(self.j)
        else:
            mouse_close_finish_pop_up_tm20(small_window=self.small_window)

    def wait(self):
        """
        Non-blocking function
        The agent stays 'paused', waiting in position
        """
        self.send_control(self.get_default_action())
        if self.save_replays:
            save_ghost()
            time.sleep(1.0)
        self.reset_race()
        time.sleep(0.5)
        self.close_finish_pop_up_tm20()

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        data, img = self.grab_data_and_img()
        speed_value = data[0]  # Speed in km/h
        speed = np.array([speed_value], dtype='float32')
        gear = np.array([data[9]], dtype='float32')
        rpm = np.array([data[10]], dtype='float32')
        
        # Position for reward computation
        pos = np.array([data[2], data[3], data[4]])
        
        # Check if reward function supports enhanced parameters
        if self.reward_type in ["enhanced", "speed", "checkpoint"]:
            # New reward functions with speed and action support
            rew, terminated = self.reward_function.compute_reward(
                pos=pos,
                speed=speed_value,
                action=self.last_action if self.last_action is not None else self.get_default_action()
            )
        else:
            # Original reward function
            rew, terminated = self.reward_function.compute_reward(pos=pos)
        
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [speed, gear, rpm, imgs]
        end_of_track = bool(data[8])
        info = {}
        
        if end_of_track:
            terminated = True
            rew += self.finish_reward
        rew += self.constant_penalty
        rew = np.float32(rew)
        
        return obs, rew, terminated, info

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        gear = spaces.Box(low=0.0, high=6, shape=(1, ))
        rpm = spaces.Box(low=0.0, high=np.inf, shape=(1, ))
        if self.resize_to is not None:
            w, h = self.resize_to
        else:
            w, h = cfg.WINDOW_HEIGHT, cfg.WINDOW_WIDTH
        if self.grayscale:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w))  # cv2 grayscale images are (h, w)
        else:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w, 3))  # cv2 images are (h, w, c)
        return spaces.Tuple((speed, gear, rpm, img))

    def get_action_space(self):
        """
        must return a Box
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(3, ))

    def get_default_action(self):
        """
        initial action at episode start
        """
        return np.array([0.0, 0.0, 0.0], dtype='float32')


class TM2020InterfaceLidar(TM2020Interface):
    def __init__(self, img_hist_len=1, gamepad=False, save_replays: bool = False, 
                 reward_type: str = "speed", reward_params: dict = None):
        super().__init__(img_hist_len, gamepad, save_replays, reward_type=reward_type, reward_params=reward_params)
        self.window_interface = None
        self.lidar = None

    def grab_lidar_speed_and_data(self):
        img = self.window_interface.screenshot()[:, :, :3]
        data = self.client.retrieve_data()
        speed = np.array([
            data[0],
        ], dtype='float32')
        lidar = self.lidar.lidar_20(img=img, show=False)
        return lidar, speed, data

    def initialize(self):
        super().initialize_common()
        self.small_window = False
        self.lidar = Lidar(self.window_interface.screenshot())
        self.initialized = True

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        img, speed, data = self.grab_lidar_speed_and_data()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        img, speed, data = self.grab_lidar_speed_and_data()
        pos = np.array([data[2], data[3], data[4]])
        
        # Check if reward function supports enhanced parameters
        if self.reward_type in ["enhanced", "speed", "checkpoint"]:
            rew, terminated = self.reward_function.compute_reward(
                pos=pos,
                speed=data[0],
                action=self.last_action if self.last_action is not None else self.get_default_action()
            )
        else:
            rew, terminated = self.reward_function.compute_reward(pos=pos)
            
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        end_of_track = bool(data[8])
        info = {}
        if end_of_track:
            rew += self.finish_reward
            terminated = True
        rew += self.constant_penalty
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        imgs = spaces.Box(low=0.0, high=np.inf, shape=(
            self.img_hist_len,
            19,
        ))  # lidars
        return spaces.Tuple((speed, imgs))


class TM2020InterfaceLidarProgress(TM2020InterfaceLidar):

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        img, speed, data = self.grab_lidar_speed_and_data()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        progress = np.array([0], dtype='float32')
        obs = [speed, progress, imgs]
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        img, speed, data = self.grab_lidar_speed_and_data()
        pos = np.array([data[2], data[3], data[4]])
        
        # Check if reward function supports enhanced parameters
        if self.reward_type in ["enhanced", "speed", "checkpoint"]:
            rew, terminated = self.reward_function.compute_reward(
                pos=pos,
                speed=data[0],
                action=self.last_action if self.last_action is not None else self.get_default_action()
            )
        else:
            rew, terminated = self.reward_function.compute_reward(pos=pos)
            
        progress = np.array([self.reward_function.cur_idx / self.reward_function.datalen], dtype='float32')
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, progress, imgs]
        end_of_track = bool(data[8])
        info = {}
        if end_of_track:
            rew += self.finish_reward
            terminated = True
        rew += self.constant_penalty
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        progress = spaces.Box(low=0.0, high=1.0, shape=(1,))
        imgs = spaces.Box(low=0.0, high=np.inf, shape=(
            self.img_hist_len,
            19,
        ))  # lidars
        return spaces.Tuple((speed, progress, imgs))


if __name__ == "__main__":
    # Example usage with different reward functions
    
    # Default: Original reward function
    env_original = TM2020Interface(
        img_hist_len=4,
        gamepad=True,
        save_replays=False
    )
    
    # Enhanced multi-component reward
    env_enhanced = TM2020Interface(
        img_hist_len=4,
        gamepad=True,
        save_replays=False,
        reward_type="enhanced",
        reward_params={
            'speed_reward_weight': 0.3,
            'progress_reward_weight': 0.4,
            'smooth_control_weight': 0.2,
            'checkpoint_reward_weight': 0.1
        }
    )
    
    # Speed-focused reward for aggressive driving
    env_speed = TM2020Interface(
        img_hist_len=4,
        gamepad=True,
        save_replays=False,
        reward_type="speed",
        reward_params={
            'target_speed': 200.0,
            'speed_exp': 2.5
        }
    )
    
    # Checkpoint-based reward for early training
    env_checkpoint = TM2020Interface(
        img_hist_len=4,
        gamepad=True,
        save_replays=False,
        reward_type="checkpoint",
        reward_params={
            'num_checkpoints': 15,
            'checkpoint_reward': 15.0
        }
    )
