import numpy as np
import random
from keras.utils import to_categorical
from collections import deque
from itertools import islice

import os
import cv2

data_root = 'C:/Mrugank_pc/Online competitions/MineRL/Raw Data/'

def observation_wrapper(state, batch_size = 1):
    state['pov'] = np.reshape(state['pov']/255.0, [batch_size,64,64,3])
    state['compassAngle'] = np.reshape(state['compassAngle'], [batch_size,1])
    
    return state


def no_op_action(mission):

    action = {
        'attack' : 0,
        'back': 0,
        'camera': [0., 0.],
        'forward': 0,
        'jump': 0,
        'left': 0,
        'place': 'none',
        'right': 0,
        'sneak': 0,
        'sprint': 0
    }
    
    return action


def map_to_actionset(mission, action_vec, max_camera_rot = 10):
    
    action = no_op_action(mission)
        
    if mission in ['MineRLNavigateDense-v0','MineRLNavigate-v0']:
        
        #action_vec = [0:Back-Forward, 1:Left-Right, 2:Sneak-Sprint, 3:attack, 4:jump, 
        #              5:place_none, 6:place_dirt, 7:camera_hor, 8:camera_ver]      
        
        if action_vec[0] < -0.3:
            action['back'] = 1
        elif action_vec[0] > 0.3:
            action['forward'] = 1
        
        if action_vec[1] < -0.3:
            action['left'] = 1
        elif action_vec[1] > 0.3:
            action['right'] = 1
        
        if action_vec[2] < -0.3:
            action['sneak'] = 1
        elif action_vec[2] > 0.3:
            action['sprint'] = 1
        
        if action_vec[3] > 0.5:
            action['attack'] = 1
        
        if action_vec[4] > 0.5:
            action['jump'] = 1
        
          
        place = np.argmax(action_vec[5:7])
        #place=[none,dirt]
        
        if place == 0:
            action['place'] = 'none'
        elif place == 1:
            action['place'] = 'dirt'
        
        
        action['camera'] = [action_vec[7]*max_camera_rot, action_vec[8]*max_camera_rot]
        
    return action

def map_from_actionset(actionset, batch_size):
    
    
    action_dim = 9
    action_vec = np.zeros((batch_size, action_dim))
    
    for i in range(batch_size):
        if actionset['forward'][i] == 1:
            action_vec[i][0] = 0.95
        elif actionset['back'][i] == 1:
            action_vec[i][0] = -0.95

        if actionset['right'][i] == 1:
            action_vec[i][1] = 0.95
        elif actionset['left'][i] == 1:
            action_vec[i][1] = -0.95

        if actionset['sneak'][i] == 1:
            action_vec[i][2] = -0.95
        elif actionset['sprint'][i] == 1:
            action_vec[i][2] = 0.95

        if actionset['attack'][i] == 1:
            action_vec[i][3] = 0.95

        if actionset['jump'][i] == 1:
            action_vec[i][4] = 0.95

        
        place_items = 2
        place_vec = np.zeros(place_items)
        place_vec[:] = 0.05/(place_items-1)

        #0:none, 1:dirt 
        if actionset['place'][i] == 0:
            place_vec[0] = 0.95
        elif actionset['place'][i] == 1:
            place_vec[1] = 0.95
        
        #print(place_vec,action_vec[i])
        action_vec[i][5] = place_vec[0]
        action_vec[i][6] = place_vec[1]
        

        action_vec[i][7], action_vec[i][8] = np.clip(actionset['camera'][i], -10, 10)

    return action_vec

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size, sequence = False):
        # Randomly sample batch_size examples
        if not sequence :
            if self.num_experiences < batch_size:
                return random.sample(self.buffer, self.num_experiences)
            else:
                return random.sample(self.buffer, batch_size)
        else:
            if self.num_experiences < batch_size:
                return deque(islice(self.buffer,1,self.num_experiences+1))
            else:
                return deque(islice(self.buffer,1,batch_size))
            
    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

def add_noise(action, epsilon):
    
    action[0] = np.clip(action[0] + (1-epsilon)*np.random.uniform(-0.3, 0.3), -0.95, 0.95)
    action[1] = np.clip(action[1] + (1-epsilon)*np.random.uniform(-0.3, 0.3), -0.95, 0.95)
    action[2] = np.clip(action[2] + (1-epsilon)*np.random.uniform(-0.3, 0.3), -0.95, 0.95)
    action[3] = np.clip(action[3] + (1-epsilon)*np.random.uniform(-0.3, 0.3), 0, 0.95)
    action[4] = np.clip(action[4] + (1-epsilon)*np.random.uniform(-0.3, 0.3), 0, 0.95)
    
    action[7] = np.clip(action[7] + (1-epsilon)*np.random.uniform(-0.3, 0.3), -0.95, 0.95)
    action[8] = np.clip(action[8] + (1-epsilon)*np.random.uniform(-0.3, 0.3), -0.95, 0.95)
    
    return action

class OU(object):

    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)
    
