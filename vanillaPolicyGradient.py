#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 22:53:27 2023

@author: admin
"""

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch

from torch import nn
from torch import optim
from torch import tensor

import math
import time
from zmqRemoteApi import RemoteAPIClient
import random
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,"/home/admin/shahil_ws/src/ME8930")

class PolicyEstimator():
    def __init__(self, observation_space, action_space):
        self.num_observations = observation_space.shape[0]
        self.num_actions = action_space.n

        self.network = nn.Sequential(
            nn.Linear(self.num_observations, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_actions),
            nn.Softmax(dim=-1)
        )
        #self.network.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4, weight_decay=1e-5)

    def predict(self, observation):
        return self.network(torch.FloatTensor(observation))
    
    def saveModel(self):
        torch.save(self.network.state_dict(), 'pgModel_7.pth')
    
    def loadModel(self):
        self.network.load_state_dict(torch.load('pgModel_7.pth'))
        
def reset():
    #robot_x_array = np.array([1.622, -1.80, -0.75])
    #robot_y_array = np.array([-0.60, -1.37, 2.65])
    
    robot_x_array = -1.80
    robot_y_array = -1.37
    
    #goal_x_array = np.array([-1.68, -0.01, 1.56])
    #goal_y_array = np.array([0.22, -0.88, -2.08])
    
    goal_x_array = -1.68
    goal_y_array = 0.22
    
    min_distance = 1
    # generate random locations until the minimum distance condition is satisfied
    while True:
        '''rand = np.random.randint(0,3)
        robot_x = robot_x_array[rand]
        robot_y = robot_y_array[rand]
        
        rand = np.random.randint(0,3)
        goal_x = goal_x_array[rand]
        goal_y = goal_y_array[rand]'''
        
        '''robot_x = random.uniform(-3.5, 3.5)
        robot_y = random.uniform(-3.5, 3.5)
        goal_x = random.uniform(-2.5, 2.5)
        goal_y = random.uniform(-2.5, 2.5)'''
        
        robot_x = robot_x_array
        robot_y = robot_y_array
        
        goal_x = goal_x_array
        goal_y = goal_y_array
        
        obs_x_1 = -2.1
        obs_y_1 = 2.025
        
        obs_x_2 = 2.5
        obs_y_2 = 0.30
        
        obs_x_3 = -1.15
        obs_y_3 = -2.70
        
        # calculate distances between objects
        dist_robot_goal = math.sqrt((robot_x - goal_x)**2 + (robot_y - goal_y)**2)
        
        dist_robot_obs_1 = math.sqrt((robot_x - obs_x_1)**2 + (robot_y - obs_y_1)**2)
        dist_robot_obs_2 = math.sqrt((robot_x - obs_x_2)**2 + (robot_y - obs_y_2)**2)
        dist_robot_obs_3 = math.sqrt((robot_x - obs_x_3)**2 + (robot_y - obs_y_3)**2)
        
        dist_goal_obs_1 = math.sqrt((goal_x - obs_x_1)**2 + (goal_y - obs_y_1)**2)
        dist_goal_obs_2 = math.sqrt((goal_x - obs_x_2)**2 + (goal_y - obs_y_2)**2)
        dist_goal_obs_3 = math.sqrt((goal_x - obs_x_3)**2 + (goal_y - obs_y_3)**2)
        
        # check if minimum distance condition is satisfied
        if (dist_robot_goal >= min_distance and dist_robot_obs_1 >= min_distance and dist_robot_obs_2 >= min_distance and dist_robot_obs_3 >= min_distance and dist_goal_obs_1 >= min_distance and dist_goal_obs_2 >= min_distance and dist_goal_obs_3 >= min_distance):
            break
    # set object positions
    sim.setObjectPosition(robot_handle, -1, [robot_x, robot_y, 0.42])
    sim.setObjectOrientation(robot_handle, -1, [0, 0, 0])
    sim.setObjectPosition(goal_handle, -1, [goal_x, goal_y, 0.32])
    #sim.setObjectPosition(obstacle_handle, -1, [obs_x, obs_y, 0.82])
    
    #obstacle_positions = 
    
def get_observation():
    # Get the position and orientation of the robot
    robot_pos = sim.getObjectPosition(robot_handle, -1)
    robot_orient = sim.getObjectOrientation(robot_handle, -1)
    # Get the position of the goal
    goal_pos = sim.getObjectPosition(goal_handle, -1)
    
    # Get the position and size of the obstacle
    obstacle_pos_1 = sim.getObjectPosition(obstacle_handle_1, -1)
    obstacle_pos_2 = sim.getObjectPosition(obstacle_handle_2, -1)
    obstacle_pos_3 = sim.getObjectPosition(obstacle_handle_3, -1)
    #obstacle_pos = sim.getObjectPosition(obstacle_handle_4, -1)
    #obstacle_size = sim.getObjectFloatParameter(obstacle_handle, 9)
    disToWall = sim.checkDistance(robot_handle, obstacle_handle_4)
    
    # Combine the observations into a single state vector
    observation = np.concatenate([robot_pos[0:2], robot_orient[2:], goal_pos[0:2], obstacle_pos_1[0:2], obstacle_pos_2[0:2], obstacle_pos_3[0:2]])
    
    return observation

def detectCollision(observation):
    collision_1 = sim.checkCollision(robot_handle, obstacle_handle_1)
    collision_2 = sim.checkCollision(robot_handle, obstacle_handle_2)
    collision_3 = sim.checkCollision(robot_handle, obstacle_handle_3)
    collision_4 = sim.checkCollision(robot_handle, obstacle_handle_4)
    
    if collision_1[0] == 0 and collision_2[0] == 0 and collision_3[0] == 0 and collision_4[0] == 0:
        collision = 0
    else:
        collision = 1
        
    if observation[0] > 3.5 or observation[0] < -3.5 or observation[1] > 3.5 or observation[1] < -3.5:
        collision = 1
        
    return collision

def calculate_reward(observation, prevObservation):
    
    # Calculate the distance between the robot and the goal
    #disToGoal = sim.checkDistance(robot_handle, goal_handle) 
    disToGoal = np.linalg.norm(observation[0:2] - observation[3:5])
    prevDisToGoal = np.linalg.norm(prevObservation[0:2] - prevObservation[3:5])
    
    distance_difference =  disToGoal -  prevDisToGoal
    
    # Calculate the heading error between the robot's current heading and the heading to the goal
    goal_heading = math.atan2(observation[4] - observation[1], observation[3] - observation[0])
    heading_error = abs(goal_heading - observation[2])
    if heading_error > math.pi:
        heading_error = 2*math.pi - heading_error
    
    # Calculate the distance to the nearest obstacle
    obstacle_distances = []
    for i in range(3):
        obstacle_distance = math.sqrt((observation[5+(i*2)] - observation[0])**2 + (observation[6+(i*2)] - observation[1])**2)
        obstacle_distances.append(obstacle_distance)
        
    nearest_obstacle_distance = min(obstacle_distances)
    
    #print('distance difference = ', distance_difference)
    #print('Heading error = ', heading_error)
    
    #check for collision with obstacle
    collision = detectCollision(observation)
    
    # Calculate the reward based on distance to the goal and proximity to obstacles
    if collision == 0:
        done = False
        if distance_difference < 0.0:
            #rospy.logwarn("DECREASE IN DISTANCE GOOD")
            reward = -100 * abs(np.pi - heading_error) * distance_difference / disToGoal #-30*(1-numpy.tanh(4.5*(min_distance-0.5)))
        else:
            reward = -20
            
        if nearest_obstacle_distance < 1:
            reward -= 10/nearest_obstacle_distance
            
        if disToGoal < 0.6:
            reward += 1000
            done = True
        
    else:
        reward = -1000
        done = True
                
    #print('reward = ', reward)
    return reward, done


    
def step(action, prevObservation):
    # Define the possible actions
    #actions = {0: np.linspace(-1, 1, 21),  # Linear
    #           1: np.linspace(-1, 1, 21)}  # Angular
    
    if action == 0:  # Forward
        left_torque = 1
        right_torque = 1
    elif action == 1:  # Left
        left_torque = -0.5
        right_torque = 0.5
    elif action == 2:  # Right
        left_torque = 0.5
        right_torque = -0.5
    elif action == 3:  # backward
        left_torque = -1
        right_torque = -1
    
    # Set the robot's velocity based on the action
    sim.setJointTargetVelocity(left_motor, left_torque)
    sim.setJointTargetVelocity(right_motor, right_torque)
    
    # Wait for a short time to allow the robot to move
    time.sleep(0.1)
    
    # Get the next observation and reward
    observation = get_observation()
    reward, done = calculate_reward(observation, prevObservation)
    #done = check_if_done()
    
    info = None
    
    return observation, reward, done, info

def vanilla_policy_gradient(observation_space, action_space, estimator, num_episodes=1500, batch_size=10, discount_factor=0.99, render=False,
                            early_exit_reward_amount=None):
    total_rewards, batch_rewards, batch_observations, batch_actions = [], [], [], []
    batch_counter = 1

    optimizer = optim.Adam(estimator.network.parameters(), lr=0.01)
    action_space = np.arange(action_space.n) # [0, 1] for cartpole (either left or right)

    for current_episode in range(num_episodes):
        reset()
        observation = get_observation()
        rewards, actions, observations = [], [], []

        while True:
            #if render:
            #    render()

            # use policy to make predictions and run an action
            action_probs = estimator.predict(observation).detach().numpy()
            action = np.random.choice(action_space, p=action_probs) # randomly select an action weighted by its probability

            # push all episodic data, move to next observation
            observations.append(observation)
            prevObservation = observation
            observation, reward, done, _ = step(action, prevObservation)
            rewards.append(reward)
            actions.append(action)
            
            estimator.loadModel()

            if done:
                # apply discount to rewards
                r = np.full(len(rewards), discount_factor) ** np.arange(len(rewards)) * np.array(rewards)
                r = r[::-1].cumsum()[::-1]
                discounted_rewards = r - r.mean()

                # collect the per-batch rewards, observations, actions
                batch_rewards.extend(discounted_rewards)
                batch_observations.extend(observations)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))
                
                print('episode number: ', current_episode,'reward = ', sum(rewards))

                if batch_counter >= batch_size:
                    # reset gradient
                    optimizer.zero_grad()

                    # tensorify things
                    batch_rewards = torch.FloatTensor(batch_rewards)
                    batch_observationss = torch.FloatTensor(batch_observations)
                    batch_actions = torch.LongTensor(batch_actions)

                    # calculate loss
                    logprob = torch.log(estimator.predict(batch_observations))
                    batch_actions = batch_actions.reshape(len(batch_actions), 1)
                    selected_logprobs = batch_rewards * torch.gather(logprob, 1, batch_actions).squeeze()
                    loss = -selected_logprobs.mean()

                    # backprop/optimize
                    loss.backward()
                    optimizer.step()

                    # reset the batch
                    batch_rewards, batch_observations, batch_actions = [], [], []
                    batch_counter = 1
                    
                #if current_episode % batch_size == 0:
                estimator.saveModel()
                
                # get running average of last 100 rewards, print every 100 episodes
                average_reward = np.mean(total_rewards[-100:])
                if current_episode % 100 == 0:
                    print(f"average of last 100 rewards as of episode {current_episode}: {average_reward:.2f}")

                # quit early if average_reward is high enough
                if early_exit_reward_amount and average_reward > early_exit_reward_amount:
                    return total_rewards

                break

    return total_rewards

if __name__ == '__main__':
    
    client = RemoteAPIClient()
    sim = client.getObject('sim')

    # Get handles to the robot and obstacle objects
    left_motor = sim.getObject('/leftMotor')
    right_motor = sim.getObject('/rightMotor')
    robot_handle = sim.getObject('/PioneerP3DX')
    goal_handle = sim.getObject('/Disc')
    obstacle_handle_1 = sim.getObject('/Cuboid[2]')
    obstacle_handle_2 = sim.getObject('/Cuboid[0]')
    obstacle_handle_3 = sim.getObject('/Cuboid[1]')
    obstacle_handle_4 = sim.getObject('/ExternalWall')

    action_space = gym.spaces.Discrete(3)  # Three possible actions: forward, left, right, backward
    observation_space = gym.spaces.Box(low=-3.5, high=3.5, shape=(11,), dtype=np.float32)
    
    # create environment
    #env_name = 'CartPole-v0'
    #env = gym.make(env_name)

    # actually run the algorithm
    rewards = vanilla_policy_gradient(observation_space, action_space, PolicyEstimator(observation_space, action_space), num_episodes=1500)

    # moving average
    moving_average_num = 100
    def moving_average(x, n=moving_average_num):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[n:] - cumsum[:-n]) / float(n)

    # plotting
    plt.scatter(np.arange(len(rewards)), rewards, label='individual episodes')
    plt.plot(moving_average(rewards), label=f'moving average of last {moving_average_num} episodes')
    plt.title(f'Vanilla Policy Gradient')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()