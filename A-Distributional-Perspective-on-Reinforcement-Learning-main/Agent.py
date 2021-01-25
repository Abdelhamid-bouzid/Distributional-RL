# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 21:34:24 2020

@author: Abdelhamid Bouzid
"""
import numpy as np
import torch as T
from models.DeepQnetwork import DeepQnetwork
from Buffer import Buffer

class Agent(object):
    def __init__(self, gamma, lamda, epsilon, lr, input_dims, n_actions, max_size, replace, batch_size,eps_dec,eps_min,hiddens, N=100,Vmin=-200,Vmax=200):
        
        self.gamma      = gamma
        self.lamda      = lamda
        self.epsilon    = epsilon
        self.lr         = lr
        self.input_dims = input_dims
        self.n_actions  = n_actions
        self.max_size   = max_size
        self.replace    = replace
        self.batch_size = batch_size
        self.eps_min    = eps_min
        self.eps_dec    = eps_dec
        self.N          = N
        self.Vmin       = Vmin
        self.Vmax       = Vmax
        self.delta_z    = (Vmax-Vmin)/(N-1)
        self.hiddens    = hiddens
        
        self.learn_step_cntr = 0
        
        self.memory = Buffer(self.max_size, self.input_dims)
        
        self.eval_model = DeepQnetwork(self.lr, self.input_dims, self.n_actions, self.N, self.hiddens)
        self.next_model = DeepQnetwork(self.lr, self.input_dims, self.n_actions, self.N, self.hiddens)
        
        self.atoms      = T.unsqueeze(T.from_numpy(np.array([[self.Vmin + i * self.delta_z for i in range(self.N)]])), 2).float().to(self.eval_model.device)
        
    def choose_action(self, state):
        self.eval_model.eval()
        if np.random.random()> self.epsilon:
            state   = T.tensor([state],dtype=T.float).to(self.eval_model.device)
            
            probs, _  = self.eval_model.forward(state)
            probs     = probs.detach()
            Q_target  = T.matmul(probs, self.atoms).squeeze(1)
            
            action    = T.argmax(T.squeeze(Q_target)).cpu().numpy()
            
        else:
            possible_actions = [i for i in range(self.n_actions)]
            action           = np.random.choice(possible_actions)
            
        self.eval_model.train()   
        return action
    
    def learn(self):
        if self.memory.current_mem < self.batch_size:
            return
        
        self.eval_model.optimizer.zero_grad()
        self.update_model_weights()
        
        indices = np.arange(self.batch_size)
        
        states, actions, rewards, n_states, terminal = self.sample_memory()
        
        '''######################################### the prediction  output###########################'''
        _,log_Q_network = self.eval_model(states)
        log_Q_network   = log_Q_network[indices,actions,:]
        
        '''######################################### determine best action t###########################'''
        n_probs,_     = self.next_model(n_states)
        n_Q_target    = T.matmul(n_probs.detach(), self.atoms)
        a_star        = T.squeeze(T.argmax(n_Q_target, dim=1))
        n_probs       = n_probs[indices,a_star,:]
        
        '''############################## Compute the probs of R+ gamma*Z t###########################'''
        m = T.zeros(self.batch_size,self.N).to(self.eval_model.device)
        for i in range(self.N):
            terminal = (1-terminal)
            T_zj = T.clamp(rewards.long() + self.gamma*terminal*i*self.delta_z, min = self.Vmin, max = self.Vmax).float()
            bj   = (T_zj-self.Vmin)/self.delta_z
            l    = bj.floor().long()
            u    = bj.ceil().long()
            
            mask_Q_l = T.zeros(m.size()).to(self.eval_model.device)
            mask_Q_u = T.zeros(m.size()).to(self.eval_model.device)
            
            mask_Q_l.scatter_(1, l.unsqueeze(1), n_probs[:,i].unsqueeze(1))
            mask_Q_u.scatter_(1, u.unsqueeze(1), n_probs[:,i].unsqueeze(1))
            
            m += mask_Q_l*(u.float() + (l == u).float()-bj.float()).unsqueeze(1)
            m += mask_Q_u*(-l.float()+bj.float()).unsqueeze(1)
        
        loss = - T.sum(T.sum(T.mul(n_probs, m),-1),-1) / self.batch_size
        loss.backward()
        
        self.eval_model.optimizer.step()
        self.learn_step_cntr += 1
        
        self.decrement_epsilon()
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min
                               
    def update_model_weights(self):
        if (self.replace is not None) and (self.learn_step_cntr % self.replace ==0):
            self.next_model.load_state_dict(self.eval_model.state_dict())
            
    def store_transitions(self, state, action, reward, n_state, done):
        self.memory.store_transitions(state, action, reward, n_state, done)
        
    def sample_memory(self):
        
        states, actions, rewards, n_states, terminal = self.memory.sample_buffer(self.batch_size)
         
        states    = T.tensor(states).to(self.eval_model.device)
        actions   = T.tensor(actions).to(self.eval_model.device)
        rewards   = T.tensor(rewards).to(self.eval_model.device)
        n_states  = T.tensor(n_states).to(self.eval_model.device)
        terminal  = T.tensor(terminal).to(self.eval_model.device)
        
        return states, actions, rewards, n_states, terminal
    
    def save_models(self):
        self.eval_model.save_checkpoint()
        self.next_model.save_checkpoint()

    def load_models(self):
        self.eval_model.load_checkpoint()
        self.next_model.load_checkpoint()
        
