# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:48:03 2020

@author: Abdelhamid Bouzid
"""

'''

Buffer is a class that stores all transitions and also used to sample batches:

'''
import numpy as np

class Buffer(object):
    def __init__(self, max_size, input_shape):
        super(Buffer, self).__init__
        
        self.mem_size    = max_size
        self.current_mem = 0
        
        self.state_memory     = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.New_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory    = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory    = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory  = np.zeros(self.mem_size, dtype=np.int64)
        
    def store_transitions(self, state, action, reward, n_state, done):
        
        index = self.current_mem % self.mem_size
        
        self.state_memory[index]     = state
        self.action_memory[index]    = action
        self.reward_memory[index]    = reward
        self.New_state_memory[index] = n_state
        self.terminal_memory[index]  = done
        
        self.current_mem += 1
        
    def sample_buffer(self, batch_size):
        
        buffer_size = min(self.mem_size,self.current_mem)
        
        batch = np.random.choice(buffer_size, batch_size, replace=False)
        
        states   = self.state_memory[batch]
        actions  = self.action_memory[batch]
        rewards  = self.reward_memory[batch]
        n_states = self.New_state_memory[batch]
        terminal = self.terminal_memory[batch]
        
        return states, actions, rewards, n_states, terminal
