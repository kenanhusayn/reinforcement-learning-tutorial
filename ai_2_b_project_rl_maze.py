####################                Artificial Intelligence_2_B_Project
####################                Reinforcement Learning
####################                Prof. Luca Iocchi
####################                Student Kenan Husayn
####################                SAPIENZA University, Roma, Italy, 2017

import numpy as np
import random

# size of the matrix, and also the number of the states of the environment
size = 25

# walls
everything_else = -1

# the available paths to move, feel free to add gates through the walls i.e (6,7)
allowed_paths = ((0,5), (5,6), (6,11), (10,11), (10,15), (15,20), (20,21),
                 (16,21), (16,17), (17,22), (22,23), (18,23), (13,18), (12,13),
                 (7,12), (7,8), (1,2), (2,3), (3,8), (3,4), (4,9), (9,14), (14,19),
                 (19,24))

# initial position of the Agent, bad_guy and the princess
initial_state = (11)
bad_guy = (1, 6)
princess = (24)

# Gamma (learning parameter).
gamma = 0.8

# we will prepare the rewards table below
# create a sizeXsize matrix that consists of ones
R = np.matrix(np.ones([size,size]))
# multiply by whatever value it should carry by default, in this case I use -1
R *= everything_else
# returns are allowed
for i in range(size):
    R[i,i] = 0
# if (x,y) is an allowed path, so is (y,x)
# tuple to list, append reversed paths and add them together
allowed_paths = list(allowed_paths)
new_paths = []
for i in allowed_paths:
    a = i[::-1]
    new_paths.append(a)
allowed_paths = allowed_paths + new_paths

# rewards for allowed paths
for i in allowed_paths:
    R[i] = 0
# rewards for bumping into the bad_guy and kissing the princess
# they should be in the available paths
for i in allowed_paths:
    if i[0] == bad_guy or i[1] == bad_guy:
        R[i] = -100
    if i[0] == princess or i[1] == princess:
        R[i] = 100

# we will prepare the memory of the agent (Q table) below
# create a sizeXsize zero matrix, we don't know anything yet
Q = np.matrix(np.zeros([size,size]))

# return the available actions for the state
def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row != -1)[1]
    return av_act
available_act = available_actions(initial_state)

# given the range of all the available actions for the state, pick one at random, exploration
def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_actions_range,1))
    return next_action
action = sample_next_action(available_act)

# update the Q table based on the learning algorithm
# Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
def update(current_state, action, gamma):
    
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size = 1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]
    
    # Q learning formula
    Q[current_state, action] = R[current_state, action] + gamma * max_value
update(initial_state,action,gamma)

# train the agent many times
for i in range(size*1000):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    update(current_state,action,gamma)
    
# Q matrix is divided by the largest reward value, which will return 1
# then multiply by 100 to get a normalized table
print("Q table after the training:")
print(Q/np.max(Q)*100)

# test run
# starting froim the initial state
current_state = initial_state
steps = [current_state]

# don't give up without finding the princess
while current_state != princess:
    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size = 1))
    else:
        next_step_index = int(next_step_index)
# note down each next step towards the goal
    steps.append(next_step_index)
    current_state = next_step_index

# results
print("The best path to save the princess is:")
print(steps)
