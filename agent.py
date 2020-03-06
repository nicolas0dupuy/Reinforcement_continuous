import numpy as np
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, episode_len=[1,2,3], seed=42):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            episode_len (int list): each int i of the list correspond to a memory deque for episodes of len i
        """
        buffer_size = int(buffer_size)
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
        return min(len(self.memory[i]) for i in range(len(self.memory)))
    
class ActorAgent:
    
    def __init__(self, actorModel, criticAgent, lr = LR_ACTOR , memory_size = 3):
        self.memory_size = memory_size
        self.individual_memory = deque(maxlen=memory_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
        self.critic = criticAgent.local.to(device)
        self.local = actorModel.to(device)
        self.target = actorModel.to(device) # soft updated copy of local model
        self.lr = lr
        self.optimizer = optim.Adam(self.local.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        self.noise = OUNoise(4, 42)
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.individual_memory.append(e)
        
    def act(self, state, add_noise=True):
        """
        select an action from state
        """
        state = torch.from_numpy(state).float().to(device)
        action = self.local(state).cpu().data.numpy()
        if add_noise:
            action += self.noise.sample()
        return action
        
    def learn(self, states):
        # Compute actor loss
        states = torch.from_numpy(states).float().to(device)
        actions_pred = self.local(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        # Minimize the loss
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()
        
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

def mixed_loss(expected, from_eval, threshold = 0.05, weigths = [5, 1]):
    """
    weigthed loss for the critic to give more importance to the high rewards
    """
    if len(from_eval[from_eval >= threshold]) > 0:
        a = F.mse_loss(expected[from_eval >= threshold], from_eval[from_eval >= threshold])
    else:
        a = 0
    if len(from_eval[from_eval < threshold]) > 0:
        b = F.mse_loss(expected[from_eval < threshold], from_eval[from_eval < threshold])
    else:
        b = 0
    return weigths[0] * a + weigths[1] * b        
        
class CriticAgent:
    
    def __init__(self, criticModel, lr, lossthreshold = 0.05, lossweigths = [5, 1]):
        """
        """
        
        self.local = criticModel.to(device)
        self.target = criticModel.to(device) # soft updated copy of local model
        self.lr = lr
        self.optimizer = optim.Adam(self.local.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        self.lossthreshold = lossthreshold
        self.lossweigths = lossweigths
        
    
    def evaluate(self, states, actions, rewards, next_state, next_action, dones):
        """
        inputs:
        states, actions, rewards, next_states, dones: Numpy arrays over a sequence
        output:
        tuple(first state, first action, Q value of the state-action function for the first state-action couple
        with respect to the whole sequence)
        """
        # Get the length of the sequence
        #length = len(states)
        # generate the discounted rewards sequence
        #gammas = np.array([GAMMA**i for i in range(length)])
        #discountedRewards = gammas*rewards
        # evaluate last next_state using target model
        Qlast = self.target(torch.from_numpy(next_state).float().to(device), torch.from_numpy(next_action).float().to(device)).cpu().data.numpy() * GAMMA #**length ADAPT LATER TO LONG EPISODES
        # sum to obtain the discounted reward
        Qest = rewards + Qlast  # Qest = np.sum(discountedRewards) + Qlast
        return (states, actions, Qest)
    
    def learn(self, states, actions, estimated_rewards, lrFactor = 1.):
        """
        input:
        A batch of states, actions, and rewards from evaluate method
        lrFractor (float): learning rate discounter to take into account the importance of the current batch
        """
        states, actions = torch.from_numpy(states).float().to(device), torch.from_numpy(actions).float().to(device)
        # print("states: {}".format(states))
        # print("actions: {}".format(actions))
        Q_expected = self.local(states, actions) # .cpu().data.numpy()
        estimated_rewards = torch.from_numpy(estimated_rewards).float().to(device)
        # print("Q_expected: {}".format(Q_expected))
        # print("estimated_rewards: {}".format(estimated_rewards))
        critic_loss = mixed_loss(Q_expected, estimated_rewards, self.lossthreshold, self.lossweigths)  # F.mse_loss(Q_expected, estimated_rewards)
        # Minimize the loss
        self.optimizer.zero_grad()
        critic_loss.backward()
        # optimize with learning rate adapted by lrFactor
        # self.optimizer.lr *= lrFactor
        self.optimizer.step()
        # self.optimizer.lr /= lrFactor
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
    
    
class CriticModel(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed = 42, fcs1_units=256, fc2_units=256, fc3_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(CriticModel, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.bn3 = nn.BatchNorm1d(fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # print("state: {}".format(state))
        # print("action: {}".format(action))
        xs = F.tanh(self.fcs1(state))
        # print("xs: {}".format(xs))
        x = torch.cat((xs, action), dim=1)
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return self.fc4(x)
    
class ActorModel(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=42, fc_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(ActorModel, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.tanh(self.fc1(state))
        return F.tanh(self.fc2(x))
    
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state