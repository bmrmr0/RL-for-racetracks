# File: tmrl/custom/tm/utils/compute_reward.py
# Consolidated reward functions for Trackmania 2020

# standard library imports
import os
import pickle
import numpy as np
import logging
from collections import deque


class RewardFunction:
    """
    Original reward function - Computes a reward from the Openplanet API for Trackmania 2020.
    This is the default reward function that follows a trajectory.
    """
    def __init__(self,
                 reward_data_path,
                 nb_obs_forward=10,
                 nb_obs_backward=10,
                 nb_zero_rew_before_failure=10,
                 min_nb_steps_before_failure=int(3.5 * 20),
                 max_dist_from_traj=60.0):
        """
        Instantiates a reward function for TM2020.

        Args:
            reward_data_path: path where the trajectory file is stored
            nb_obs_forward: max distance of allowed cuts (as a number of positions in the trajectory)
            nb_obs_backward: same thing but for when rewinding the reward to a previously visited position
            nb_zero_rew_before_failure: after this number of steps with no reward, episode is terminated
            min_nb_steps_before_failure: the episode must have at least this number of steps before failure
            max_dist_from_traj: the reward is 0 if the car is further than this distance from the demo trajectory
        """
        if not os.path.exists(reward_data_path):
            logging.debug(f" reward not found at path:{reward_data_path}")
            self.data = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # dummy reward
        else:
            with open(reward_data_path, 'rb') as f:
                self.data = pickle.load(f)

        self.cur_idx = 0
        self.nb_obs_forward = nb_obs_forward
        self.nb_obs_backward = nb_obs_backward
        self.nb_zero_rew_before_failure = nb_zero_rew_before_failure
        self.min_nb_steps_before_failure = min_nb_steps_before_failure
        self.max_dist_from_traj = max_dist_from_traj
        self.step_counter = 0
        self.failure_counter = 0
        self.datalen = len(self.data)

        # self.traj = []

    def compute_reward(self, pos):
        """
        Computes the current reward given the position pos
        Args:
            pos: the current position
        Returns:
            float, bool: the reward and the terminated signal
        """
        # self.traj.append(pos)

        terminated = False
        self.step_counter += 1  # step counter to enable failure counter
        min_dist = np.inf  # smallest distance found so far in the trajectory to the target pos
        index = self.cur_idx  # cur_idx is where we were last step in the trajectory
        temp = self.nb_obs_forward  # counter used to find cuts
        best_index = 0  # index best matching the target pos

        while True:
            dist = np.linalg.norm(pos - self.data[index])  # distance of the current index to target pos
            if dist <= min_dist:  # if dist is smaller than our minimum found distance so far,
                min_dist = dist  # then we found a new best distance,
                best_index = index  # and a new best index
                temp = self.nb_obs_forward  # we will have to check this number of positions to find a possible cut
            index += 1  # now we will evaluate the next index in the trajectory
            temp -= 1  # so we can decrease the counter for cuts
            # stop condition
            if index >= self.datalen or temp <= 0:  # if trajectory complete or cuts counter depleted
                # We check that we are not too far from the demo trajectory:
                if min_dist > self.max_dist_from_traj:
                    best_index = self.cur_idx  # if so, consider we didn't move

                break  # we found the best index and can break the while loop

        # The reward is then proportional to the number of passed indexes (i.e., track distance):
        reward = (best_index - self.cur_idx) / 100.0

        if best_index == self.cur_idx:  # if the best index didn't change, we rewind (more Markovian reward)
            min_dist = np.inf
            index = self.cur_idx

            # Find the best matching index in rewind:
            while True:
                dist = np.linalg.norm(pos - self.data[index])
                if dist <= min_dist:
                    min_dist = dist
                    best_index = index
                    temp = self.nb_obs_backward
                index -= 1
                temp -= 1
                # stop condition
                if index <= 0 or temp <= 0:
                    break

            # If failure happens for too many steps, the episode terminates
            if self.step_counter > self.min_nb_steps_before_failure:
                self.failure_counter += 1
                if self.failure_counter > self.nb_zero_rew_before_failure:
                    terminated = True

        else:  # if we did progress on the track
            self.failure_counter = 0  # we reset the counter triggering episode termination

        self.cur_idx = best_index  # finally, we save our new best matching index

        print("original:", reward)

        return reward, terminated

    def reset(self):
        """
        Resets the reward function for a new episode.
        """
        # from pathlib import Path
        # import pickle as pkl
        # path_traj = Path.home() / 'TmrlData' / 'reward' / 'traj.pkl'
        # with open(path_traj, 'wb') as file_traj:
        #     pkl.dump(self.traj, file_traj)

        self.cur_idx = 0
        self.step_counter = 0
        self.failure_counter = 0

        # self.traj = []


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

        print("enhanced:", reward)
        
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


class SpeedFocusedRewardFunction:
    """
    A reward function that primarily rewards maintaining high speeds while making progress.
    Good for: Learning aggressive driving, speed tracks, late-stage optimization
    """
    def __init__(self,
                 reward_data_path,
                 nb_obs_forward=20,
                 nb_obs_backward=5,
                 nb_zero_rew_before_failure=20,
                 min_nb_steps_before_failure=int(3 * 20),
                 max_dist_from_traj=100.0,
                 target_speed=180.0,
                 min_speed=30.0,
                 speed_exp=2.0):
        """
        Speed-focused reward function for TM2020.
        
        Args:
            reward_data_path: path where the trajectory file is stored
            nb_obs_forward: max distance of allowed cuts
            nb_obs_backward: backward search distance
            nb_zero_rew_before_failure: steps with no reward before termination
            min_nb_steps_before_failure: minimum steps before failure
            max_dist_from_traj: maximum distance from trajectory
            target_speed: target speed for maximum reward (km/h)
            min_speed: minimum acceptable speed (km/h)
            speed_exp: exponent for speed reward curve (higher = more aggressive)
        """
        # Load trajectory
        if not os.path.exists(reward_data_path):
            logging.debug(f" reward not found at path:{reward_data_path}")
            self.data = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
            self.has_trajectory = False
        else:
            with open(reward_data_path, 'rb') as f:
                self.data = pickle.load(f)
            self.has_trajectory = True
        
        # Trajectory parameters
        self.cur_idx = 0
        self.nb_obs_forward = nb_obs_forward
        self.nb_obs_backward = nb_obs_backward
        self.nb_zero_rew_before_failure = nb_zero_rew_before_failure
        self.min_nb_steps_before_failure = min_nb_steps_before_failure
        self.max_dist_from_traj = max_dist_from_traj
        self.step_counter = 0
        self.failure_counter = 0
        self.datalen = len(self.data)
        
        # Speed parameters
        self.target_speed = target_speed
        self.min_speed = min_speed
        self.speed_exp = speed_exp
        
        # Tracking
        self.speed_history = deque(maxlen=20)
        self.position_history = deque(maxlen=10)
        self.max_sustained_speed = 0.0
        self.current_segment_speeds = []
        
    def compute_reward(self, pos, speed=None, action=None):
        """
        Computes reward heavily weighted towards speed maintenance
        """
        terminated = False
        self.step_counter += 1
        
        # Track position for stuck detection
        self.position_history.append(pos)
        
        # Default values
        progress_mult = 1.0
        speed_reward = 0.0
        
        # Progress calculation (simplified)
        if self.has_trajectory:
            progress_mult, new_idx, terminated = self._calculate_progress(pos)
            self.cur_idx = new_idx
        
        # Speed reward (primary component)
        if speed is not None:
            self.speed_history.append(speed)
            
            # Exponential speed reward
            if speed < self.min_speed:
                speed_reward = -1.0  # Heavy penalty for going too slow
            else:
                # Normalized speed (0 to 1 based on target)
                norm_speed = min(speed / self.target_speed, 1.5)
                speed_reward = np.power(norm_speed, self.speed_exp)
                
                # Bonus for sustaining high speed
                if len(self.speed_history) >= 10:
                    avg_speed = np.mean(list(self.speed_history)[-10:])
                    if avg_speed > self.target_speed * 0.8:
                        speed_reward += 0.5
                        
                # Track segment speeds for bonus
                if progress_mult > 0:
                    self.current_segment_speeds.append(speed)
                    if len(self.current_segment_speeds) > 5:
                        segment_avg = np.mean(self.current_segment_speeds[-5:])
                        if segment_avg > self.max_sustained_speed:
                            self.max_sustained_speed = segment_avg
                            speed_reward += 1.0  # Bonus for new speed records
        
        # Combine rewards: speed is primary, progress is multiplier
        total_reward = speed_reward * max(progress_mult, 0.1)
        
        # Termination for being stuck
        if self._is_stuck():
            terminated = True
            total_reward = -5.0
        
        print("speed tr:", total_reward, "sr:", speed_reward)
        
        return total_reward, terminated
    
    def _calculate_progress(self, pos):
        """Simplified progress calculation"""
        min_dist = np.inf
        best_index = self.cur_idx
        
        # Forward search
        for i in range(self.cur_idx, min(self.cur_idx + self.nb_obs_forward, self.datalen)):
            dist = np.linalg.norm(pos - self.data[i])
            if dist < min_dist:
                min_dist = dist
                best_index = i
        
        # Check if too far
        if min_dist > self.max_dist_from_traj:
            return 0.0, self.cur_idx, False
        
        # Progress multiplier based on advancement
        progress = best_index - self.cur_idx
        if progress > 0:
            self.failure_counter = 0
            # Distance penalty for straying
            distance_mult = max(0.5, 1.0 - (min_dist / self.max_dist_from_traj))
            return 1.0 * distance_mult, best_index, False
        else:
            # No progress
            if self.step_counter > self.min_nb_steps_before_failure:
                self.failure_counter += 1
                if self.failure_counter > self.nb_zero_rew_before_failure:
                    return 0.0, self.cur_idx, True
            return 0.5, self.cur_idx, False  # Small multiplier for no progress
    
    def _is_stuck(self):
        """Check if car is stuck based on position history"""
        if len(self.position_history) >= 10:
            # Check if we haven't moved much
            positions = np.array(list(self.position_history))
            movement = np.max(np.std(positions, axis=0))
            if movement < 2.0:  # Less than 2 meters of movement variance
                return True
        return False
    
    def reset(self):
        """Reset for new episode"""
        self.cur_idx = 0
        self.step_counter = 0
        self.failure_counter = 0
        self.speed_history.clear()
        self.position_history.clear()
        self.max_sustained_speed = 0.0
        self.current_segment_speeds = []


class CheckpointRewardFunction:
    """
    A reward function based on reaching discrete checkpoints with time bonuses.
    Good for: Early training, complex tracks, building confidence
    """
    def __init__(self,
                 reward_data_path,
                 num_checkpoints=20,
                 checkpoint_reward=10.0,
                 time_bonus_per_checkpoint=5.0,
                 max_time_per_checkpoint=200,
                 min_speed_bonus=50.0,
                 exploration_bonus=0.1):
        """
        Checkpoint-based reward function for TM2020.
        
        Args:
            reward_data_path: path where the trajectory file is stored
            num_checkpoints: number of checkpoints to create along track
            checkpoint_reward: base reward for reaching a checkpoint
            time_bonus_per_checkpoint: max bonus for reaching checkpoint quickly
            max_time_per_checkpoint: steps before checkpoint time bonus becomes 0
            min_speed_bonus: minimum speed for checkpoint to count (km/h)
            exploration_bonus: small reward for exploring new areas
        """
        # Load trajectory
        if not os.path.exists(reward_data_path):
            logging.debug(f" reward not found at path:{reward_data_path}")
            self.data = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
            self.has_trajectory = False
        else:
            with open(reward_data_path, 'rb') as f:
                self.data = pickle.load(f)
            self.has_trajectory = True
        
        self.datalen = len(self.data)
        
        # Checkpoint configuration
        self.num_checkpoints = min(num_checkpoints, self.datalen // 10)
        self.checkpoint_reward = checkpoint_reward
        self.time_bonus_per_checkpoint = time_bonus_per_checkpoint
        self.max_time_per_checkpoint = max_time_per_checkpoint
        self.min_speed_bonus = min_speed_bonus
        self.exploration_bonus = exploration_bonus
        
        # Create checkpoints
        self.checkpoint_indices = []
        self.checkpoint_positions = []
        self.checkpoint_radii = []
        
        if self.has_trajectory:
            for i in range(self.num_checkpoints):
                idx = int((i + 1) * self.datalen / (self.num_checkpoints + 1))
                self.checkpoint_indices.append(idx)
                self.checkpoint_positions.append(self.data[idx])
                
                # Calculate checkpoint radius based on local track curvature
                radius = self._calculate_checkpoint_radius(idx)
                self.checkpoint_radii.append(radius)
        
        # Tracking
        self.checkpoints_reached = set()
        self.checkpoint_times = {}
        self.current_checkpoint = 0
        self.step_counter = 0
        self.last_checkpoint_time = 0
        self.visited_positions = set()
        self.stuck_counter = 0
        
    def _calculate_checkpoint_radius(self, idx):
        """Calculate checkpoint radius based on track curvature"""
        if idx < 10 or idx >= self.datalen - 10:
            return 30.0  # Default radius
        
        # Look at track curvature around checkpoint
        positions = self.data[idx-10:idx+10]
        
        # Calculate path curvature
        if len(positions) > 2:
            # Simple curvature estimation
            diffs = np.diff(positions, axis=0)
            angles = np.arctan2(diffs[:, 1], diffs[:, 0])
            angle_changes = np.abs(np.diff(angles))
            avg_curvature = np.mean(angle_changes)
            
            # Tighter checkpoints on curves
            if avg_curvature > 0.1:
                return 20.0
            else:
                return 40.0
        
        return 30.0
    
    def compute_reward(self, pos, speed=None, action=None):
        """
        Computes reward based on checkpoint progression
        """
        self.step_counter += 1
        reward = 0.0
        terminated = False
        
        # Small exploration bonus for visiting new areas
        pos_key = tuple(np.round(pos[:2] / 5.0).astype(int))  # 5m grid
        if pos_key not in self.visited_positions:
            self.visited_positions.add(pos_key)
            reward += self.exploration_bonus
        
        # Check if we've reached the next checkpoint
        if self.current_checkpoint < len(self.checkpoint_positions):
            checkpoint_pos = self.checkpoint_positions[self.current_checkpoint]
            checkpoint_radius = self.checkpoint_radii[self.current_checkpoint]
            distance = np.linalg.norm(pos[:2] - checkpoint_pos[:2])  # 2D distance
            
            if distance < checkpoint_radius:
                # Check speed requirement
                speed_valid = speed is None or speed >= self.min_speed_bonus
                
                if self.current_checkpoint not in self.checkpoints_reached and speed_valid:
                    # New checkpoint reached!
                    self.checkpoints_reached.add(self.current_checkpoint)
                    
                    # Base checkpoint reward
                    reward += self.checkpoint_reward
                    
                    # Time bonus
                    time_taken = self.step_counter - self.last_checkpoint_time
                    time_ratio = max(0, 1.0 - (time_taken / self.max_time_per_checkpoint))
                    time_bonus = self.time_bonus_per_checkpoint * time_ratio
                    reward += time_bonus
                    
                    # Sequence bonus for reaching checkpoints in order
                    expected_checkpoints = set(range(self.current_checkpoint + 1))
                    if expected_checkpoints.issubset(self.checkpoints_reached):
                        reward += 2.0  # Bonus for proper sequence
                    
                    # Update tracking
                    self.checkpoint_times[self.current_checkpoint] = self.step_counter
                    self.last_checkpoint_time = self.step_counter
                    self.current_checkpoint += 1
                    self.stuck_counter = 0
                    
                    # Special reward for reaching final checkpoint
                    if self.current_checkpoint >= len(self.checkpoint_positions):
                        reward += 20.0
        
        # Distance-based guidance to next checkpoint
        if self.current_checkpoint < len(self.checkpoint_positions):
            next_checkpoint = self.checkpoint_positions[self.current_checkpoint]
            distance_to_next = np.linalg.norm(pos[:2] - next_checkpoint[:2])
            
            # Small negative reward that increases with distance
            distance_penalty = min(distance_to_next / 200.0, 1.0) * 0.1
            reward -= distance_penalty
        
        # Check for getting stuck
        if speed is not None and speed < 10.0:
            self.stuck_counter += 1
            if self.stuck_counter > 60:  # 3 seconds at 20Hz
                terminated = True
                reward = -10.0
        else:
            self.stuck_counter = max(0, self.stuck_counter - 2)
        
        # Timeout termination
        if self.step_counter > self.num_checkpoints * self.max_time_per_checkpoint:
            terminated = True
            reward = -5.0
        
        print("checkpf:", reward)

        return reward, terminated
    
    def get_progress_info(self):
        """Get information about checkpoint progress"""
        return {
            'checkpoints_reached': len(self.checkpoints_reached),
            'total_checkpoints': self.num_checkpoints,
            'current_target': self.current_checkpoint,
            'completion_percentage': len(self.checkpoints_reached) / self.num_checkpoints * 100
        }
    
    def reset(self):
        """Reset for new episode"""
        self.checkpoints_reached.clear()
        self.checkpoint_times.clear()
        self.current_checkpoint = 0
        self.step_counter = 0
        self.last_checkpoint_time = 0
        self.visited_positions.clear()
        self.stuck_counter = 0
