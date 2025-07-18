# standard library imports
import os
import pickle

# third-party imports
import numpy as np
import logging

# asyncio for graph server connection
import asyncio

class RewardFunction:
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
                 min_speed_for_reward=40):
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

        self.nb_steps_before_speed_penalty = nb_steps_before_speed_penalty
        self.max_speed_for_penalty = max_speed_for_penalty
        self.min_speed_for_reward = min_speed_for_reward
        self.prev_data = []
        self.ws_client = ws_client
        self.minor_collision_counter = 0

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
        reward = 0
        reward_multiplier = 1

        distance = data[1]
        speed = data[0]
        accelerating = True if data[6] > 0.02 else False
        braking = True if data[7] == 1 else False
        gear = data[9]
        rpm = data[10]

        if len(self.prev_data) == 0:
            self.prev_data = data
            displacement = distance
        else:
            displacement = distance - self.prev_data[1]

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
        reward += (best_index - self.cur_idx)

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

        # if not going fast enough after some initial steps, apply penalty to encourage going fast
        if self.step_counter > self.nb_steps_before_speed_penalty:
            if speed < self.max_speed_for_penalty:
                reward_multiplier -= 0.6
            elif speed > self.min_speed_for_reward:
                reward_multiplier += 0.5
        
        # for loop so i can skip stuff using continue
        for _ in "_":

            # enigne gear 0 is R. this gives big penalty because it should never go backwards, and should be discouraged
            if (gear == 0):
                print("GOING BACKWARDS - RUN TERMINATED")
                collided = True
                reward = -20
                terminated = True
                continue

            # sudden loss of speed without braking indicates a collision and should be discouraged
            if not collided and speed > 0 and self.prev_data[0] > 30:
                if (speed / self.prev_data[0]) < 0.63: # major collision, even with braking
                    print("MAJOR COLLISION TYPE 1 - RUN TERMINATED")
                    collided = True
                    reward = -20
                    terminated = True
                    continue
                elif ((speed / self.prev_data[0]) < 0.87) or ((speed / self.prev_data[0]) < 0.98 and not braking): # minor collision
                    collided = True
                    print("MINOR COLLISION TYPE 1")
                    reward_multiplier -= 0.7
                    self.minor_collision_counter += 1
            
            if not collided and speed > 1 and self.prev_data[0] > 5:
                if (speed / self.prev_data[0]) < 0.6:
                    print("MAJOR COLLISION TYPE 2 - RUN TERMINATED")
                    collided = True
                    reward = -20
                    terminated = True
                    continue
                elif (speed / self.prev_data[0]) < 0.8 and not braking:
                    print("MINOR COLLISION TYPE 2")
                    collided = True
                    reward_multiplier -= 0.7
                    self.minor_collision_counter += 1

            if not collided and rpm > 9000 and displacement < 0.8 and not braking:
                collided = True
                print("MINOR COLLISION TYPE 3")
                reward_multiplier -= 0.7
                self.minor_collision_counter += 1

            if self.minor_collision_counter > 10:
                reward = -20
                terminated = True
                continue

            if reward == 0:
                reward = 10
                if 0 < reward_multiplier < 1:
                    reward_multiplier = 1 - reward_multiplier
                    reward_multiplier *= -1
                elif reward_multiplier < 0:
                    reward_multiplier -= 1
        
        #print(data[5], data[6], data[7], data[9], data[10])
        datatosend = {
            "speed": speed,
            "distance": distance,
            "displacement": displacement,
            "gas": data[6],
            "braking": braking,
            "input steer": data[5],
            "gear": gear,
            "rpm": rpm,
            "reward": reward * reward_multiplier
        }
        self.ws_client.send_sync(datatosend)

        print("step:", self.step_counter, " "*(4-len(str(self.step_counter))), "raw rew:", reward, " "*(3-len(str(reward))), "mult:", "{:.2f}".format(reward_multiplier), " "*(5-len(str("{:.2f}".format(reward_multiplier)))), " final rew:", "{:.2f}".format(reward * reward_multiplier), " "*(4-len(str("{:.2f}".format(reward * reward_multiplier)))), "   speed:", "{:.3f}".format(speed), "dist:", "{:.2f}".format(distance), "displ:", "{:.2f}".format(displacement), "  extra:  ", "{:.2f}".format(data[5]), "{:.2f}".format(data[6]), data[7], data[9], "{:.2f}".format(data[10]))

        self.prev_data = data
        
        return reward * reward_multiplier, terminated

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

        self.minor_collision_counter = 0
        self.prev_data = []