import numpy as np
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, episode_len=[1,2,3], seed=42):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            episode_len (int list): each int i of the list correspond to a memory deque for episodes of len i
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.episode_len = episode_len
        self.memory = [deque(maxlen=buffer_size) for i in range(len(episode_len))] # internal memory (deque)
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.seed = random.seed(seed)
    
    def add(self, states, actions, rewards, next_states, dones, memory_spot):
        """Add a new experience to memory in the specified spot."""
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory[memory_spot].append(e)
    
    def sample(self, memory_spot):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory[memory_spot], k=self.batch_size)

        states = np.array([e.states for e in experiences if e is not None])
        actions = np.array([e.actions for e in experiences if e is not None])
        rewards = np.array([e.rewards for e in experiences if e is not None])
        next_states = np.array([e.next_states for e in experiences if e is not None])
        dones = np.array([e.dones for e in experiences if e is not None]).astype(np.uint8)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return min(len(self.memory[i]) for i in len(self.memory))
    
class actorAgent:
    
    def __init__(self, memory_size = 3):
        self.memory_size = memory_size
        self.individual_memory = deque(maxlen=memory_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.individual_memory.append(e)

        
        
class criticAgent:
    
    def __init__(self, criticModel):
        """
        """
        
        self.local = criticModel
        self.target = criticModel # soft updated copy of local model
        self.optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
    
    def evaluate(self, states, actions, rewards, next_state, next_action, dones):
        """
        inputs:
        states, actions, rewards, next_states, dones: Numpy arrays over a sequence
        output:
        tuple(first state, first action, Q value of the state-action function for the first state-action couple
        with respect to the whole sequence)
        """
        # Get the length of the sequence
        length = len(states)
        # generate the discounted rewards sequence
        gammas = np.array([GAMMA**i for i in range(length)])
        discountedRewards = gammas*rewards
        # evaluate last next_state using target model
        Qlast = self.target(next_state, next_action) * GAMMA**length
        # sum to obtain the discounted reward
        Qest = np.sum(discountedRewards) + Qlast
        return (state[0], action[0], Qest)
    
    def learn(self, states, actions, estimated_rewards, lrFactor = 1.):
        """
        input:
        A batch of states, actions, and rewards from evaluate method
        lrFractor (float): learning rate discounter to take into account the importance of the current batch
        """
        Q_expected = self.local(states, actions)
        critic_loss = F.mse_loss(Q_expected, estimated_rewards)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # optimize with learning rate adapted by lrFactor
        self.optimizer.lr *= lrFactor
        self.optimizer.step()
        self.optimizer.lr /= lrFactor
        # Soft update the target model
        self.soft_update(self.local, self.target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    
class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=256, fc3_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)