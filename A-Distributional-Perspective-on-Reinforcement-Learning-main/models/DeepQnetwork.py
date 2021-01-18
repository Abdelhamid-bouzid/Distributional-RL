import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from .q_network import QNetwork

class DeepQnetwork(nn.Module):
    def __init__(self, lr, state_size, action_size, N, hiddens=[64], seed=0):
        super(DeepQnetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hiddens = hiddens
        self.action_size = action_size
        self.N = N
        self.q_network = QNetwork(state_size, hiddens[-1], hiddens[:-1], seed = seed)
        self.last_layer = nn.Linear(hiddens[-1], action_size * N)
        
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.device    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.q_network(x))
        x = self.last_layer(x)
        x = x.reshape(-1, self.action_size, self.N)
        output = F.softmax(x, dim = -1)
        log_output = F.log_softmax(x, dim = -1)

        return output, log_output

