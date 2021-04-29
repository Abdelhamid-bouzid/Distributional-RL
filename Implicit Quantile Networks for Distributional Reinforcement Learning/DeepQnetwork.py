import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQnetwork(nn.Module):
    def __init__(self, lr, input_dim, n_actions, N):
        super(DeepQnetwork, self).__init__()
        
        self.check_dir = check_dir
        self.path      = os.path.join(check_dir,model_name)
        self.N         = N
        
        self.conv1     = nn.Conv2d(input_dim[0], 32, 8, stride=4)
        self.conv2     = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3     = nn.Conv2d(64, 64, 3, stride=1)
        
        fc_input_dims  = self.calculate_conv_output_dims(input_dim)
        
        self.fc1       = nn.Linear(fc_input_dims, 512)
        self.fc2       = nn.Linear(512, n_actions*self.N)
        
        self.phi       = nn.Linear(1, fc_input_dims, bias=False)
        self.phi_bias  = nn.Parameter(torch.zeros(fc_input_dims))
        
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss      = nn.MSELoss()
        
        self.device    = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size()[0], -1).unsqueeze(1) 
        
        rand_feat,tau = self.quantile()
        
        x = x*rand_feat
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.reshape(-1, self.action_size, self.N)
 
        return x,tau
        
    def quantile(self):
        # Rand Initlialization
        tau = T.rand(self.N, 1) # (self.N, 1)
        # Quants=[1,2,3,...,N_QUANT]
        quants = T.arange(0, N_QUANT, 1.0) # (N_QUANT,1)
        if USE_GPU:
            tau = tau.cuda()
            quants = quants.cuda()
        
        # phi_j(tau) = RELU(sum(cos(π*i*τ)*w_ij + b_j))
        cos_trans = T.cos(quants * tau * 3.141592).unsqueeze(2) # (N_QUANT, N_QUANT, 1)
        rand_feat = F.relu(self.phi(cos_trans).mean(dim=1) + self.phi_bias.unsqueeze(0)).unsqueeze(0) 
        
        return rand_feat,tau
        
    def calculate_conv_output_dims(self, input_dim):
        state = T.zeros(1, *input_dim)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))
    
    def save_model(self):
        print('################## saving model ########################')
        T.save(self.state_dict(), self.path)
    
    def load_model(self):
        print('################## loading model ########################')
        self.load_state_dict(T.load(self.path))

