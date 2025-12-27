# standard library imports
import os
import pickle

# third-party imports
import numpy as np
import logging

# asyncio for graph server connection
import asyncio

# math for some mathematical operations
import math

class RewardFunction:

    def resetvars(self):

        self.cur_idx = 0
        self.step_counter = 0
        self.failure_counter = 0

        self.prev_data = []
        self.minor_collision_counter = 0
        self.allowminor1 = True
        self.allowminor3 = True
        self.last_gear_increase = 0
        self.last_rpm_increase = 0
        self.last_collision = 0
        self.prev_path_reward = 0
        
    
    """
    Computes a reward from the Openplanet API for Trackmania 2020.
    """
    def __init__(self,
                 reward_data_path,
                 ws_client,
                 nb_obs_forward=10,
                 nb_obs_backward=10,
                 nb_zero_rew_before_failure=10,
                 min_nb_steps_before_failure=int(3.5 * 20),
                 max_dist_from_traj=60.0,
                 nb_steps_before_speed_penalty=20,
                 max_speed_for_penalty=10,
                 min_speed_for_reward=40,
                 terminate_after_minors=15):
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
            self.pathdata = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # dummy reward
        else:
            with open(reward_data_path, 'rb') as f:
                self.pathdata = pickle.load(f)

        self.nb_obs_forward = nb_obs_forward
        self.nb_obs_backward = nb_obs_backward
        self.nb_zero_rew_before_failure = nb_zero_rew_before_failure
        self.min_nb_steps_before_failure = min_nb_steps_before_failure
        self.max_dist_from_traj = max_dist_from_traj

        self.pathdatalen = len(self.pathdata)

        self.ws_client = ws_client
        self.nb_steps_before_speed_penalty = nb_steps_before_speed_penalty
        self.max_speed_for_penalty = max_speed_for_penalty
        self.min_speed_for_reward = min_speed_for_reward
        self.terminate_after_minors = terminate_after_minors

        self.resetvars()
        
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
            dist = np.linalg.norm(pos - self.pathdata[index])  # distance of the current index to target pos
            if dist <= min_dist:  # if dist is smaller than our minimum found distance so far,
                min_dist = dist  # then we found a new best distance,
                best_index = index  # and a new best index
                temp = self.nb_obs_forward  # we will have to check this number of positions to find a possible cut
            index += 1  # now we will evaluate the next index in the trajectory
            temp -= 1  # so we can decrease the counter for cuts
            # stop condition
            if index >= self.pathdatalen or temp <= 0:  # if trajectory complete or cuts counter depleted
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
                dist = np.linalg.norm(pos - self.pathdata[index])
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

        return reward, terminated
    
    def compute_reward(self, pos, data):
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

        collided = False
        collision_type = 0
        
        path_reward = 0
        path_reward_multiplier = 1
        speed_reward = 0
        speed_reward_multiplier = 1
        collision_reward = 0
        collision_reward_multiplier = 1

        distance = data[1]
        speed = data[0]
        steer = data[5]
        gas = data[6]
        accelerating = gas > 0.02
        braking = data[7] == 1
        braking_raw = data[7]
        gear = data[9]
        rpm = data[10]

        if len(self.prev_data) == 0:
            self.prev_data = data
            displacement = distance
        else:
            displacement = distance - self.prev_data[1]
        
        # Track if going backward for immediate penalty
        going_backward = False
        if displacement < -0.1:  # Going backward (negative displacement)
            going_backward = True

        prev_speed = self.prev_data[0]
        prev_steer = self.prev_data[5]
        prev_accelerating = True if self.prev_data[6] > 0.02 else False
        prev_braking = self.prev_data[7]
        prev_gear = self.prev_data[9]
        prev_rpm = self.prev_data[10]

        gear_increase = gear > prev_gear

        if speed > prev_speed:
            self.allowminor1 = True
        
        if gear > prev_gear:
            self.last_gear_increase = self.step_counter
        
        if rpm > prev_rpm:
            self.last_rpm_increase = self.step_counter
            self.allowminor3 = True


        ################# PATH REWARD FUNCTION #################

        while True:
            dist = np.linalg.norm(pos - self.pathdata[index])  # distance of the current index to target pos
            if dist <= min_dist:  # if dist is smaller than our minimum found distance so far,
                min_dist = dist  # then we found a new best distance,
                best_index = index  # and a new best index
                temp = self.nb_obs_forward  # we will have to check this number of positions to find a possible cut
            index += 1  # now we will evaluate the next index in the trajectory
            temp -= 1  # so we can decrease the counter for cuts
            # stop condition
            if index >= self.pathdatalen or temp <= 0:  # if trajectory complete or cuts counter depleted
                # We check that we are not too far from the demo trajectory:
                if min_dist > self.max_dist_from_traj:
                    best_index = self.cur_idx  # if so, consider we didn't move

                break  # we found the best index and can break the while loop

        # The reward is then proportional to the number of passed indexes (i.e., track distance):
        path_reward = (best_index - self.cur_idx)

        if best_index == self.cur_idx:  # if the best index didn't change, we rewind (more Markovian reward)
            min_dist = np.inf
            index = self.cur_idx

            # Find the best matching index in rewind:
            while True:
                dist = np.linalg.norm(pos - self.pathdata[index])
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
        
        # CRITICAL: Immediately penalize backward movement from the very start
        # ROOT CAUSE: At race start (cur_idx=0), when agent goes backward:
        # - path_reward = 0 (neutral, because best_index = cur_idx = 0)
        # - speed_reward = 0 (no penalty for first nb_steps_before_speed_penalty steps)
        # - Total reward = 0 (neutral!), so backward actions seem "safe" during exploration
        # This causes the policy to learn that backward actions are acceptable
        """if going_backward:
            # Override path_reward with heavy penalty regardless of trajectory matching
            path_reward = -100.0  # Heavy penalty for backward movement
            collision_reward = -50.0  # Additional penalty
            print(f"IMMEDIATE BACKWARD PENALTY: displacement={displacement:.2f}, step={self.step_counter}, path_reward overridden to -100")
        """

        ################# SPEED REWARD FUNCTION #################

        # if not going fast enough after some initial steps, apply penalty to encourage going fast
        if self.step_counter > self.nb_steps_before_speed_penalty:
            if speed < self.max_speed_for_penalty:
                speed_reward = -10
                speed_reward_multiplier -= 0.6
            elif speed > self.min_speed_for_reward:
                speed_reward = math.log(max(speed, math.e)) * 5
                #speed_reward_multiplier += 0.5
       

        ################# COLLISION REWARD FUNCTION #################

        # for loop to skip using continue
        for _ in "_":

            # enigne gear 0 is R. this gives a big penalty because it should never go backwards and should be discouraged
            if (gear == 0):
                print("GOING BACKWARDS - RUN TERMINATED")
                collided = True
                speed_reward = -1000
                # terminated = True
                continue
            
            if not collided and self.step_counter - self.last_collision >= 5:
            
                # sudden loss of speed without braking or turning indicates a collision and should be discouraged
                if speed > 0 and prev_speed > 5:
                    if (speed / prev_speed) < 0.7 and prev_speed > 20:
                        print("MAJOR COLLISION TYPE 1 - RUN TERMINATED")
                        collided = True
                        collision_reward = -30
                        terminated = True
                        collision_type = 11
                        continue
                    elif (speed / prev_speed) < 0.75 and accelerating and not braking and not prev_braking:
                        print("MAJOR COLLISION TYPE 2 - RUN TERMINATED")
                        collided = True
                        collision_reward = -30
                        terminated = True
                        collision_type = 12
                        continue
                    elif self.allowminor1 and (speed / prev_speed) < 0.98 and accelerating and not braking and not prev_braking and not (steer < -0.9 or steer > 0.9) and not (prev_steer < -0.9 or prev_steer > 0.9):
                        print("MINOR COLLISION TYPE 1")
                        collided = True
                        collision_reward_multiplier -= (0.7 + self.minor_collision_counter * 0.01)
                        self.minor_collision_counter += 1
                        self.allowminor1 = False
                        collision_type = 1
                        continue
                    elif (speed / prev_speed) < 0.85 and not braking:
                        print("MINOR COLLISION TYPE 2")
                        collided = True
                        collision_reward_multiplier -= (0.7 + self.minor_collision_counter * 0.01)
                        self.minor_collision_counter += 1
                        collision_type = 2
                        continue
                    elif self.allowminor3 and (speed / prev_speed) < 0.98 and (rpm / prev_rpm) < 0.98 and not gear_increase and (self.last_rpm_increase - self.last_gear_increase) >= 4 and accelerating and prev_accelerating and not braking and not prev_braking:
                        print("MINOR COLLISION TYPE 3")
                        collided = True
                        collision_reward_multiplier -= (0.7 + self.minor_collision_counter * 0.01)
                        self.minor_collision_counter += 1
                        self.allowminor3 = False
                        collision_type = 3
                        continue
                    elif path_reward > 0 and self.prev_path_reward > 0 and 0.2 < (path_reward / self.prev_path_reward) < 0.4 and accelerating and (speed < prev_speed) and (rpm / prev_rpm) < 0.9 and not gear_increase and self.last_gear_increase < self.last_rpm_increase:
                        print("MINOR COLLISION TYPE 5")
                        collided = True
                        collision_reward_multiplier -= (0.7 + self.minor_collision_counter * 0.01)
                        self.minor_collision_counter += 1
                        self.allowminor3 = False
                        collision_type = 5
                        continue
                
                """
                if not collided and speed > 0 and prev_speed > 30:
                    if ((speed / prev_speed) < 0.87) or ((speed / prev_speed) < 0.98 and not braking and not self.prev_data[7] and not (steer < -0.9 or steer > 0.9)):
                        print("MINOR COLLISION TYPE 1")
                        collided = True
                        reward_multiplier -= 0.8
                        self.minor_collision_counter += 1
                
                if not collided and speed > 0 and prev_speed > 5:
                    if (speed / prev_speed) < 0.63:
                        print("MAJOR COLLISION TYPE 1 - RUN TERMINATED")
                        collided = True
                        reward = -30
                        terminated = True
                        continue
                    elif (speed / prev_speed) < 0.85 and not braking:
                        print("MINOR COLLISION TYPE 2")
                        collided = True
                        reward_multiplier -= 0.7
                        self.minor_collision_counter += 1
                """

                if rpm > 9000 and displacement < 0.8 and not braking:
                    print("MINOR COLLISION TYPE 4")
                    collided = True
                    collision_reward_multiplier -= (0.7 + self.minor_collision_counter * 0.01)
                    self.minor_collision_counter += 1
                    collision_type = 4
                    continue

        if not terminated and self.minor_collision_counter >= self.terminate_after_minors:
            print(f"MINOR COLLISION LIMIT {self.terminate_after_minors} REACHED - RUN TERMINATED")
            collision_reward = -20
            terminated = True
        
        if collided:
            self.last_collision = self.step_counter
            self.allowminor3 = False

        self.prev_path_reward = path_reward

        """
        if reward == 0:
            if 0 < reward_multiplier < 1:
                reward = 10
                reward_multiplier = 1 - reward_multiplier
                reward_multiplier *= -1
            elif reward_multiplier < 0:
                reward = 10
                reward_multiplier -= 1
        elif reward < 0:
            if 0 < reward_multiplier < 1:
                reward_multiplier += 1
            elif reward_multiplier < 0:
                reward_multiplier -= 1
                reward_multiplier *= -1
        """

        if path_reward < 0:
            path_reward_multiplier = 1.0 / max(path_reward_multiplier, 1e-9)

        if speed_reward < 0:
            speed_reward_multiplier = 1.0 / max(speed_reward_multiplier, 1e-9)

        if collision_reward < 0:
            collision_reward_multiplier = 1.0 / max(collision_reward_multiplier, 1e-9)

        ################# OVERALL REWARD FUNCTION #################

        mode = 2

        if mode == 1: # all rewards are considered
            reward = path_reward * path_reward_multiplier + speed_reward * speed_reward_multiplier + collision_reward * collision_reward_multiplier
        
        elif mode == 2: # only path reward is considered, vanilla reward function
            reward = path_reward * path_reward_multiplier

        elif mode == 3: # path and speed rewards are considered
            reward = path_reward * path_reward_multiplier + speed_reward * speed_reward_multiplier
        
        elif mode == 4: # path and collision rewards are considered
            reward = path_reward * path_reward_multiplier + collision_reward * collision_reward_multiplier

        
        #print(data[5], data[6], data[7], data[9], data[10])
        datatosend = {
            "speed": speed,
            "distance": distance,
            "displacement": displacement,
            "gas": gas,
            "braking": braking,
            "input steer": steer,
            "gear": gear,
            "rpm": rpm,
            "reward": reward,
            "collision": collision_type
        }
        self.ws_client.send_async(datatosend)

        print("step:", self.step_counter, " "*(3-len(str(self.step_counter))), 
        "Frew:", "{:.2f}".format(reward), " "*(3-len(str(reward))), 
        "Prew:", "{:.2f}".format(path_reward * path_reward_multiplier), " "*(3-len(str(path_reward * path_reward_multiplier))), 
        "Srew:", "{:.2f}".format(speed_reward * speed_reward_multiplier), " "*(3-len(str(speed_reward * speed_reward_multiplier))), 
        "Crew:", "{:.2f}".format(collision_reward * collision_reward_multiplier), " "*(3-len(str(collision_reward * collision_reward_multiplier))), 
        "  speed:", "{:.3f}".format(speed), 
        "dist:", "{:.2f}".format(distance), 
        "displ:", "{:.2f}".format(displacement), 
        "  extra:  ", "{:.2f}".format(steer), "{:.2f}".format(gas), braking_raw, gear, "{:.2f}".format(rpm), 
        "se:", self.last_gear_increase , self.last_rpm_increase)

        self.prev_data = data
        
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

        self.resetvars()


# ============================================================================
# EXPERIMENT FRAMEWORK REWARD VARIANTS
# These subclasses extend the base RewardFunction with different strategies
# for optimizing lap times.
# ============================================================================


class TimeOptimalReward(RewardFunction):
    """
    Time-optimal reward function.
    
    Adds time pressure to the base reward by penalizing slow segment completion
    and rewarding fast progress through the track.
    
    Key features:
    - Base waypoint progress reward (inherited)
    - Time bonus: Rewards faster-than-baseline progress
    - Time penalty: Penalizes slower-than-baseline progress
    - Speed maintenance rewards
    """
    
    def __init__(self,
                 reward_data_path,
                 ws_client,
                 time_pressure_weight=0.5,
                 baseline_steps_per_waypoint=1.0,
                 speed_bonus_weight=0.3,
                 **kwargs):
        """
        Initialize TimeOptimalReward.
        
        Args:
            time_pressure_weight: Weight for time pressure bonus/penalty
            baseline_steps_per_waypoint: Expected steps per waypoint (baseline)
            speed_bonus_weight: Weight for speed maintenance bonus
            **kwargs: Additional args passed to parent
        """
        super().__init__(reward_data_path, ws_client, **kwargs)
        
        self.time_pressure_weight = time_pressure_weight
        self.baseline_steps_per_waypoint = baseline_steps_per_waypoint
        self.speed_bonus_weight = speed_bonus_weight
        
        logging.info(f"TimeOptimalReward initialized with time_pressure={time_pressure_weight}")
    
    def compute_reward(self, pos, data):
        """
        Compute time-optimal reward.
        
        Extends base reward with time pressure component.
        """
        # Get base reward
        base_reward, terminated = super().compute_reward(pos, data)
        
        # Extract speed from data
        speed = data[0] if len(data) > 0 else 0
        
        # Time pressure: compare actual progress to expected baseline
        expected_progress = self.step_counter * self.baseline_steps_per_waypoint
        actual_progress = self.cur_idx
        
        if actual_progress > expected_progress:
            # Going faster than baseline - bonus
            time_reward = (actual_progress - expected_progress) * self.time_pressure_weight
        else:
            # Going slower than baseline - penalty (smaller magnitude)
            time_reward = (actual_progress - expected_progress) * self.time_pressure_weight * 0.5
        
        # Speed bonus for maintaining high speed
        speed_reward = 0.0
        if speed > 100:
            speed_reward = (speed - 100) / 200.0 * self.speed_bonus_weight
        elif speed < 30 and self.step_counter > 20:
            # Penalty for going too slow after initial acceleration
            speed_reward = -0.5 * self.speed_bonus_weight
        
        total_reward = base_reward + time_reward + speed_reward
        
        return total_reward, terminated


class RacingLineReward(RewardFunction):
    """
    Racing line optimization reward function.
    
    Adds apex detection and curvature-aware speed targets to encourage
    optimal racing trajectories.
    
    Key features:
    - Base waypoint progress reward (inherited)
    - Apex bonus: Extra reward for being close to track apexes (turn entry points)
    - Curvature-aware speed: Rewards appropriate speed for track curvature
    - Momentum preservation: Penalizes unnecessary speed loss
    """
    
    def __init__(self,
                 reward_data_path,
                 ws_client,
                 apex_bonus_weight=1.0,
                 curvature_lookahead=20,
                 speed_reward_weight=0.8,
                 apex_distance_threshold=5.0,
                 **kwargs):
        """
        Initialize RacingLineReward.
        
        Args:
            apex_bonus_weight: Weight for apex proximity bonus
            curvature_lookahead: Waypoints to look ahead for curvature calculation
            speed_reward_weight: Weight for speed optimization
            apex_distance_threshold: Distance threshold for apex bonus
            **kwargs: Additional args passed to parent
        """
        super().__init__(reward_data_path, ws_client, **kwargs)
        
        self.apex_bonus_weight = apex_bonus_weight
        self.curvature_lookahead = curvature_lookahead
        self.speed_reward_weight = speed_reward_weight
        self.apex_distance_threshold = apex_distance_threshold
        
        # Pre-compute apex locations
        self.apex_indices = self._detect_apex_locations()
        
        logging.info(f"RacingLineReward initialized. Found {len(self.apex_indices)} apex points.")
    
    def _detect_apex_locations(self):
        """
        Pre-compute apex locations based on track curvature.
        
        Returns:
            List of waypoint indices where apexes are located
        """
        apex_indices = []
        
        if len(self.pathdata) < 5:
            return apex_indices
        
        for i in range(2, len(self.pathdata) - 2):
            # Calculate curvature using vectors before and after point
            v1 = self.pathdata[i] - self.pathdata[i-1]
            v2 = self.pathdata[i+1] - self.pathdata[i]
            
            # Normalize vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm < 1e-6 or v2_norm < 1e-6:
                continue
            
            v1 = v1 / v1_norm
            v2 = v2 / v2_norm
            
            # Calculate angle between vectors
            dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.arccos(dot)
            
            # Apex is a point of significant curvature change
            if angle > 0.3:  # ~17 degrees - significant turn
                apex_indices.append(i)
        
        return apex_indices
    
    def _compute_apex_bonus(self, pos):
        """
        Compute bonus for proximity to apex points.
        
        Args:
            pos: Current position
            
        Returns:
            Apex bonus reward
        """
        bonus = 0.0
        
        for apex_idx in self.apex_indices:
            # Only consider apexes near current position
            if abs(self.cur_idx - apex_idx) > 10:
                continue
            
            apex_pos = self.pathdata[apex_idx]
            dist_to_apex = np.linalg.norm(pos - apex_pos)
            
            if dist_to_apex < self.apex_distance_threshold:
                # Closer to apex = more reward (linear decay)
                bonus += self.apex_bonus_weight * (1.0 - dist_to_apex / self.apex_distance_threshold)
        
        return bonus
    
    def _compute_optimal_speed(self, cur_idx):
        """
        Compute optimal speed for current track segment based on upcoming curvature.
        
        Args:
            cur_idx: Current waypoint index
            
        Returns:
            Optimal speed target (km/h)
        """
        if cur_idx + self.curvature_lookahead >= len(self.pathdata):
            lookahead = len(self.pathdata) - cur_idx - 1
        else:
            lookahead = self.curvature_lookahead
        
        if lookahead < 2:
            return 150.0  # Default speed
        
        # Calculate maximum curvature in lookahead window
        max_curvature = 0.0
        
        for i in range(cur_idx + 1, min(cur_idx + lookahead, len(self.pathdata) - 1)):
            if i < 1:
                continue
            
            v1 = self.pathdata[i] - self.pathdata[i-1]
            v2 = self.pathdata[i+1] - self.pathdata[i] if i+1 < len(self.pathdata) else v1
            
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm < 1e-6 or v2_norm < 1e-6:
                continue
            
            dot = np.clip(np.dot(v1/v1_norm, v2/v2_norm), -1.0, 1.0)
            angle = np.arccos(dot)
            max_curvature = max(max_curvature, angle)
        
        # Map curvature to target speed
        if max_curvature < 0.1:  # Straight
            target_speed = 300.0
        elif max_curvature < 0.3:  # Gentle curve
            target_speed = 200.0
        elif max_curvature < 0.6:  # Medium turn
            target_speed = 120.0
        else:  # Sharp turn
            target_speed = 80.0
        
        return target_speed
    
    def _compute_speed_reward(self, speed):
        """
        Compute reward for appropriate speed given track curvature.
        
        Args:
            speed: Current speed
            
        Returns:
            Speed optimization reward
        """
        target_speed = self._compute_optimal_speed(self.cur_idx)
        
        speed_error = abs(speed - target_speed)
        
        # Gaussian reward centered at target speed
        reward = self.speed_reward_weight * np.exp(-speed_error**2 / (2 * 50**2))
        
        # Extra penalty for being much slower than target
        if speed < target_speed * 0.7:
            reward -= 0.3 * self.speed_reward_weight
        
        return reward
    
    def compute_reward(self, pos, data):
        """
        Compute racing line optimized reward.
        
        Extends base reward with apex and speed components.
        """
        # Get base reward
        base_reward, terminated = super().compute_reward(pos, data)
        
        # Extract speed from data
        speed = data[0] if len(data) > 0 else 0
        
        # Apex bonus
        apex_bonus = self._compute_apex_bonus(pos)
        
        # Speed reward (curvature-aware)
        speed_reward = self._compute_speed_reward(speed)
        
        total_reward = base_reward + apex_bonus + speed_reward
        
        return total_reward, terminated


class HybridReward(RewardFunction):
    """
    Hybrid reward combining multiple optimization strategies.
    
    Allows configurable weights for different reward components.
    """
    
    def __init__(self,
                 reward_data_path,
                 ws_client,
                 path_weight=1.0,
                 speed_weight=0.5,
                 time_weight=0.3,
                 apex_weight=0.5,
                 efficiency_weight=0.2,
                 **kwargs):
        """
        Initialize HybridReward.
        
        Args:
            path_weight: Weight for path following reward
            speed_weight: Weight for speed rewards
            time_weight: Weight for time pressure
            apex_weight: Weight for apex bonuses
            efficiency_weight: Weight for efficiency (smooth driving)
            **kwargs: Additional args passed to parent
        """
        super().__init__(reward_data_path, ws_client, **kwargs)
        
        self.path_weight = path_weight
        self.speed_weight = speed_weight
        self.time_weight = time_weight
        self.apex_weight = apex_weight
        self.efficiency_weight = efficiency_weight
        
        # For tracking previous state
        self.prev_speed = 0
        self.prev_steer = 0
        
        logging.info(f"HybridReward initialized with weights: path={path_weight}, "
                     f"speed={speed_weight}, time={time_weight}, apex={apex_weight}, "
                     f"efficiency={efficiency_weight}")
    
    def compute_reward(self, pos, data):
        """
        Compute hybrid reward.
        """
        # Get base path reward
        terminated = False
        self.step_counter += 1
        
        # === Path reward (from base class logic) ===
        min_dist = np.inf
        index = self.cur_idx
        temp = self.nb_obs_forward
        best_index = 0
        
        while True:
            dist = np.linalg.norm(pos - self.pathdata[index])
            if dist <= min_dist:
                min_dist = dist
                best_index = index
                temp = self.nb_obs_forward
            index += 1
            temp -= 1
            if index >= self.pathdatalen or temp <= 0:
                if min_dist > self.max_dist_from_traj:
                    best_index = self.cur_idx
                break
        
        path_reward = (best_index - self.cur_idx) * self.path_weight
        
        if best_index == self.cur_idx:
            min_dist = np.inf
            index = self.cur_idx
            while True:
                dist = np.linalg.norm(pos - self.pathdata[index])
                if dist <= min_dist:
                    min_dist = dist
                    best_index = index
                    temp = self.nb_obs_backward
                index -= 1
                temp -= 1
                if index <= 0 or temp <= 0:
                    break
            
            if self.step_counter > self.min_nb_steps_before_failure:
                self.failure_counter += 1
                if self.failure_counter > self.nb_zero_rew_before_failure:
                    terminated = True
        else:
            self.failure_counter = 0
        
        self.cur_idx = best_index
        
        # Extract data
        speed = data[0] if len(data) > 0 else 0
        steer = data[5] if len(data) > 5 else 0
        
        # === Speed reward ===
        speed_reward = 0.0
        if speed > 100:
            speed_reward = (speed - 100) / 200.0 * self.speed_weight
        elif speed < 30 and self.step_counter > 20:
            speed_reward = -0.3 * self.speed_weight
        
        # === Time pressure reward ===
        expected_progress = self.step_counter * 1.0
        actual_progress = self.cur_idx
        time_reward = (actual_progress - expected_progress) * 0.1 * self.time_weight
        
        # === Efficiency reward (smooth steering) ===
        steer_change = abs(steer - self.prev_steer)
        efficiency_reward = -steer_change * self.efficiency_weight if steer_change > 0.3 else 0
        
        # Update previous state
        self.prev_speed = speed
        self.prev_steer = steer
        
        # Combine all rewards
        total_reward = path_reward + speed_reward + time_reward + efficiency_reward
        
        return total_reward, terminated
    
    def reset(self):
        """Reset for new episode."""
        super().reset()
        self.prev_speed = 0
        self.prev_steer = 0