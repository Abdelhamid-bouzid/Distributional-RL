# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 23:23:53 2020

@author: Abdelhamid Bouzid
"""
import gym
import numpy as np
from Agent import Agent
#from utils import make_env
from plot import plot_epi_step

if __name__ == '__main__':
    
    env             = gym.make('PongNoFrameskip-v4')
    
    best_score      = -np.inf
    n_games         = 1000000
    
    input_dims1     = (env.observation_space.shape)
    n_actions1      = env.action_space.n
    
    agent = Agent(gamma=0.99,lamda=0.9, epsilon=1, lr=10**-4, input_dims=input_dims1, n_actions=n_actions1, max_size=50000, replace=10000, batch_size=32,eps_dec=1e-5,eps_min=0.1, hiddens = [128,128], N=100,Vmin=-200,Vmax=200)
    
    figure_file = 'plot.png'
    
    #n_steps = 0
    game_scores, eps_history, game_steps = [], [], []
    
    for i in range(n_games):
        
        '''################################### start the env from begining ########################'''
        done = False
        state = env.reset()
        
        '''################################### for one game do 'online' ########################'''
        score   = 0
        n_steps = 0
        while not done:
            action = agent.choose_action(state)
            
            n_state, reward, done, info = env.step(action)

            agent.store_transitions(state, action, reward , n_state, int(done))
            agent.learn()
            
            state    = n_state
            n_steps +=1
            score   += reward
            
        game_scores.append(score)
        game_steps.append(n_steps)
        
        print("############## episode number = {} ######### number of steps = {} ############ score {}".format(i, n_steps,score))
    plot_epi_step(game_scores,game_steps)   
