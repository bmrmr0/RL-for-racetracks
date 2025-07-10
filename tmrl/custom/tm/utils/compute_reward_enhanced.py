# File: tmrl/custom/tm/utils/compute_reward.py
# Enhanced multi-component reward function for Trackmania 2020

# standard library imports
import os
import pickle
import numpy as np
import logging
from collections import deque


class EnhancedRewardFunction:
    """
    An enhanced reward function for Trackmania 2020 that combines multiple reward components
    to encourage fast, smooth, and efficient racing.
    """
    def __init__(self,
                 reward_data_path,
                 nb_obs_forward=15,
                 nb_obs_backward=10,
                 nb_zero_rew_before_failure=15,
                 min_nb_steps_before_failure=int(3.5 * 20),
                 max_dist_from_traj=80.0,
                 speed_reward_weight=0.3,
                 progress_reward_weight=0.4,
                 smooth_control_weight=0.2,
                 checkpoint_reward_weight=0.1):
        """
        Instantiates an enhanced reward function for TM2020.

        Args:
            reward_data_path: path where the trajectory file is stored
            nb_obs_forward: max distance of allowed cuts (as a number of positions in the trajectory)
            nb_obs_backward: same thing but for when rewinding the reward to a previously visited position
            nb_zero_rew_before_failure: after this number of steps with no reward, episode is terminated
            min_nb_steps_before_failure: the episode must have at least this number of steps before failure
            max_dist_from_traj: the reward is 0 if the car is further than this distance from the demo trajectory
            speed_reward_weight: weight for speed maintenance component
            progress_reward_weight: weight for trajectory progress component
            smooth_control_weight: weight for smooth control component
            checkpoint_reward_weight: weight for checkpoint bonuses
        """
        # Load trajectory data
        if not os.path.exists(reward_data_path):
            logging.debug(f" reward not found at path:{reward_data_path}")
            self.data = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # dummy reward
            self.has_trajectory = False
        else:
            with open(reward_data_path, 'rb') as f:
                self.data = pickle.load(f)
            self.has_trajectory = True

        # Trajectory following parameters
        self.cur_idx = 0
        self.nb_obs_forward = nb_obs_forward
        self.nb_obs_backward = nb_obs_backward
        self.nb_zero_rew_before_failure = nb_zero_rew_before_failure
        self.min_nb_steps_before_failure = min_nb_steps_before_failure
        self.max_dist_from_traj = max_dist_from_traj
        self.step_counter = 0
        self.failure_counter = 0
        self.datalen = len(self.data)
        
        # Reward component weights
        self.speed_reward_weight = speed_reward_weight
        self.progress_reward_weight = progress_reward_weight
        self.smooth_control_weight = smooth_control_weight
        self.checkpoint_reward_weight = checkpoint_reward_weight
        
        # Additional tracking variables
        self.prev_speed = 0.0
        self.speed_history = deque(maxlen=10)
        self.prev_action = np.zeros(3)
        self.checkpoint_indices = []
        self.checkpoints_passed = set()
        self.max_progress = 0
        self.stuck_counter = 0
        
        # Calculate checkpoint indices (every 10% of the track)
        if self.has_trajectory:
            for i in range(1, 10):
                checkpoint_idx = int(self.datalen * i / 10)
                self.checkpoint_indices.append(checkpoint_idx)
        
        # Speed thresholds (adjust based on track characteristics)
        self.min_desired_speed = 50.0  # km/h
        self.optimal_speed = 150.0     # km/h
        self.max_speed = 300.0         # km/h

    def compute_reward(self, pos, speed=None, action=None):
        """
        Computes the current reward given the position, speed, and action
        Args:
            pos: the current position (numpy array)
            speed: current speed (float, in km/h)
            action: current action [forward, backward, steer] (numpy array)
        Returns:
            float, bool: the reward and the terminated signal
        """
        terminated = False
        self.step_counter += 1
        
        # Initialize reward components
        progress_reward = 0.0
        speed_reward = 0.0
        smooth_control_reward = 0.0
        checkpoint_bonus = 0.0
        
        # 1. Progress Reward (similar to original but enhanced)
        if self.has_trajectory:
            progress_reward, terminated = self._compute_progress_reward(pos)
        
        # 2. Speed Reward
        if speed is not None:
            speed_reward = self._compute_speed_reward(speed)
            self.speed_history.append(speed)
            self.prev_speed = speed
        
        # 3. Smooth Control Reward
        if action is not None:
            smooth_control_reward = self._compute_smooth_control_reward(action)
            self.prev_action = action.copy()
        
        # 4. Checkpoint Bonus
        if self.has_trajectory:
            checkpoint_bonus = self._compute_checkpoint_bonus()
        
        # Combine all reward components
        total_reward = (
            self.progress_reward_weight * progress_reward +
            self.speed_reward_weight * speed_reward +
            self.smooth_control_weight * smooth_control_reward +
            self.checkpoint_reward_weight * checkpoint_bonus
        )
        
        # Additional termination conditions
        if self._check_stuck():
            terminated = True
            total_reward -= 10.0  # Penalty for getting stuck
        
        return total_reward, terminated

    def _compute_progress_reward(self, pos):
        """Compute reward based on progress along the trajectory"""
        min_dist = np.inf
        index = self.cur_idx
        temp = self.nb_obs_forward
        best_index = 0
        
        # Find best matching position in trajectory (forward search)
        while True:
            dist = np.linalg.norm(pos - self.data[index])
            if dist <= min_dist:
                min_dist = dist
                best_index = index
                temp = self.nb_obs_forward
            index += 1
            temp -= 1
            
            if index >= self.datalen or temp <= 0:
                break
        
        # Check if too far from trajectory
        if min_dist > self.max_dist_from_traj:
            best_index = self.cur_idx
        
        # If no progress, search backward
        if best_index == self.cur_idx:
            min_dist = np.inf
            index = self.cur_idx
            
            while True:
                dist = np.linalg.norm(pos - self.data[index])
                if dist <= min_dist:
                    min_dist = dist
                    best_index = index
                    temp = self.nb_obs_backward
                index -= 1
                temp -= 1
                
                if index <= 0 or temp <= 0:
                    break
            
            # Check failure conditions
            if self.step_counter > self.min_nb_steps_before_failure:
                self.failure_counter += 1
                if self.failure_counter > self.nb_zero_rew_before_failure:
                    return -1.0, True  # Negative reward for failure
        else:
            self.failure_counter = 0
        
        # Calculate progress reward with distance penalty
        progress = best_index - self.cur_idx
        distance_penalty = min(min_dist / self.max_dist_from_traj, 1.0)
        progress_reward = (progress / 100.0) * (1.0 - 0.3 * distance_penalty)
        
        # Update tracking variables
        self.cur_idx = best_index
        self.max_progress = max(self.max_progress, best_index)
        
        # Bonus for new progress
        if best_index > self.max_progress:
            progress_reward += 0.5
        
        return progress_reward, False

    def _compute_speed_reward(self, speed):
        """Compute reward based on speed maintenance"""
        # Normalize speed to [0, 1] range
        if speed < self.min_desired_speed:
            # Penalty for going too slow
            speed_reward = -0.5 * (1.0 - speed / self.min_desired_speed)
        elif speed < self.optimal_speed:
            # Reward for maintaining good speed
            speed_reward = (speed - self.min_desired_speed) / (self.optimal_speed - self.min_desired_speed)
        elif speed < self.max_speed:
            # Slight reward for high speed, but diminishing returns
            speed_reward = 1.0 + 0.2 * (speed - self.optimal_speed) / (self.max_speed - self.optimal_speed)
        else:
            # Small penalty for excessive speed (might lose control)
            speed_reward = 1.1 - 0.1 * min((speed - self.max_speed) / 50.0, 1.0)
        
        # Reward for smooth speed changes (avoid sudden braking/acceleration)
        if len(self.speed_history) > 1:
            speed_change = abs(speed - self.prev_speed)
            if speed_change < 20.0:  # Smooth change
                speed_reward += 0.1
            elif speed_change > 50.0:  # Sudden change
                speed_reward -= 0.1
        
        return speed_reward

    def _compute_smooth_control_reward(self, action):
        """Compute reward for smooth control inputs"""
        # action = [forward, backward, steer]
        smooth_reward = 0.0
        
        # Penalize simultaneous forward and backward
        if action[0] > 0.5 and action[1] > 0.5:
            smooth_reward -= 0.5
        
        # Reward smooth steering changes
        steer_change = abs(action[2] - self.prev_action[2])
        if steer_change < 0.3:
            smooth_reward += 0.2
        elif steer_change > 0.7:
            smooth_reward -= 0.1
        
        # Reward consistent throttle when at good speed
        if len(self.speed_history) > 0:
            avg_speed = np.mean(self.speed_history)
            if avg_speed > self.optimal_speed * 0.8:
                throttle_consistency = 1.0 - abs(action[0] - self.prev_action[0])
                smooth_reward += 0.1 * throttle_consistency
        
        return smooth_reward

    def _compute_checkpoint_bonus(self):
        """Compute bonus for passing checkpoints"""
        checkpoint_bonus = 0.0
        
        for i, checkpoint_idx in enumerate(self.checkpoint_indices):
            if self.cur_idx >= checkpoint_idx and i not in self.checkpoints_passed:
                self.checkpoints_passed.add(i)
                checkpoint_bonus = 5.0  # Big bonus for reaching checkpoint
                self.stuck_counter = 0  # Reset stuck counter
                break
        
        return checkpoint_bonus

    def _check_stuck(self):
        """Check if the car is stuck"""
        if len(self.speed_history) >= 10:
            avg_recent_speed = np.mean(list(self.speed_history)[-5:])
            if avg_recent_speed < 10.0:  # Nearly stopped
                self.stuck_counter += 1
                if self.stuck_counter > 40:  # Stuck for 2 seconds at 20Hz
                    return True
            else:
                self.stuck_counter = max(0, self.stuck_counter - 1)
        
        return False

    def reset(self):
        """
        Resets the reward function for a new episode.
        """
        self.cur_idx = 0
        self.step_counter = 0
        self.failure_counter = 0
        self.prev_speed = 0.0
        self.speed_history.clear()
        self.prev_action = np.zeros(3)
        self.checkpoints_passed.clear()
        self.max_progress = 0
        self.stuck_counter = 0
