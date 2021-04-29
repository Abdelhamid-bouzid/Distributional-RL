# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 21:34:24 2020

@author: Abdelhamid Bouzid
"""
import numpy as np
import torch as T
from DeepQnetwork import DeepQnetwork
from Buffer import Buffer
import numpy as np

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
        # quantiles
        self.QUANTS          = np.linspace(0.0, 1.0, self.N + 1)[1:]
        self.QUANTS_TARGET   = torch.FloatTensor((np.linspace(0.0, 1.0, self.N + 1)[:-1] + QUANTS)/2).view(1, -1,-1) # (1, N_QUANT,1)
        self.hiddens    = hiddens
        
        self.learn_step_cntr = 0
        
        self.memory = Buffer(self.max_size, self.input_dims)
        
        self.eval_model = DeepQnetwork(self.lr, self.input_dims, self.n_actions, self.N)
        self.next_model = DeepQnetwork(self.lr, self.input_dims, self.n_actions, self.N)
        
        self.loss_fn    = T.nn.SmoothL1Loss(reduction='none')
        
    def choose_action(self, state):
        self.eval_model.eval()
        if np.random.random()> self.epsilon:
            state   = T.tensor([state],dtype=T.float).to(self.eval_model.device)
            
            Q         = self.eval_model.forward(state)   #(batch_size,n_actions,n_quants)
            Q         = Q.detach()                       # to free history
            Q         = Q.mean(dim=2)                    #(batch_size,n_actions)

            action    = T.argmax(Q,dim=1).cpu().numpy()
            
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
        Q_pred = self.eval_model(states)    #(batch_size,n_actions,n_quants)
        Q_pred = Q_pred[indices,actions,:]  #(batch_size,n_quants)
        Q_pred = Q_pred.unsqueeze(2)        #(batch_size,n_quants, 1)
        
        '''######################################### determine best action t###########################'''
        Q_next        = self.next_model(n_states)        #(batch_size,n_actions, n_quants)
        best_actions  = Q_next.mean(dim=2).argmax(dim=1) # (batch_size)
        Q_next        = Q_next[indices,best_actions,:]   #(batch_size, n_quants)
        Q_target      = rewards + self.gamma*Q_next      #(batch_size, n_quants)
        Q_target      = Q_target.unsqueeze(1)            # (batch_size , 1, n_quants)
        
        '''######################################### compute quantile Huber loss ###########################'''
        error  = Q_target.detach() - Q_pred                           # (batch_size , n_quants, n_quants)
        weight = torch.abs(self.QUANTS_TARGET - error.le(0.).float()) # (batch_size, n_quants, n_quants)
        loss   = (self.loss_fn (Q_target, Q_pred)*weight).mean()
        
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
        
