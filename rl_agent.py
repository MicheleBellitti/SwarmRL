import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque, namedtuple
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
import copy

# Experience replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Experience(*args))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Advanced Neural Network Architectures
class AttentionNetwork(nn.Module):
    """Attention mechanism for focusing on relevant environmental features"""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, V)
        
        return attended_values, attention_weights

class GRUMemory(nn.Module):
    """Memory system for agents with GRU cells"""
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
        output, hidden = self.gru(x, hidden)
        return output, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

class ActorCriticNetwork(nn.Module):
    """Advanced Actor-Critic with attention and memory"""
    def __init__(self, state_dim, action_dim, hidden_dim=128, use_attention=True, use_memory=True):
        super().__init__()
        self.use_attention = use_attention
        self.use_memory = use_memory
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = AttentionNetwork(hidden_dim, hidden_dim//2)
        
        # Memory system
        if use_memory:
            self.memory = GRUMemory(hidden_dim, hidden_dim)
            
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, hidden=None):
        features = self.feature_extractor(state)
        
        if self.use_attention:
            features, _ = self.attention(features.unsqueeze(1))
            features = features.squeeze(1)
            
        if self.use_memory and hidden is not None:
            features, hidden = self.memory(features.unsqueeze(1), hidden)
            features = features.squeeze(1)
            
        action_probs = self.actor(features)
        state_value = self.critic(features)
        
        return action_probs, state_value, hidden

# PPO Agent (Fixed Implementation)
class PPOAgent:
    def __init__(self, n_states, n_actions, lr=3e-4, gamma=0.99, eps_clip=0.2, 
                 k_epochs=4, use_attention=True, use_memory=True):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.policy = ActorCriticNetwork(n_states, n_actions, use_attention=use_attention, 
                                        use_memory=use_memory)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCriticNetwork(n_states, n_actions, use_attention=use_attention,
                                           use_memory=use_memory)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.memory = ReplayBuffer(capacity=10000)
        self.hidden = None
        
    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, _, self.hidden = self.policy_old(state_tensor, self.hidden)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def update(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
            
        # Sample batch
        batch = self.memory.sample(batch_size)
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.FloatTensor([e.done for e in batch])
        
        # Calculate returns
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # PPO update
        for _ in range(self.k_epochs):
            action_probs, state_values, _ = self.policy(states)
            dist = Categorical(action_probs)
            action_log_probs = dist.log_prob(actions)
            
            # Calculate advantages
            advantages = returns - state_values.squeeze()
            
            # Calculate ratios
            old_action_probs, _, _ = self.policy_old(states)
            old_dist = Categorical(old_action_probs)
            old_action_log_probs = old_dist.log_prob(actions)
            
            ratios = torch.exp(action_log_probs - old_action_log_probs)
            
            # Clipped objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Calculate losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(state_values.squeeze(), returns)
            entropy_loss = -dist.entropy().mean()
            
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())

# A2C (Advantage Actor-Critic) Agent
class A2CAgent:
    def __init__(self, n_states, n_actions, lr=1e-3, gamma=0.99, n_steps=5):
        self.gamma = gamma
        self.n_steps = n_steps
        
        self.network = ActorCriticNetwork(n_states, n_actions)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        
    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, value, _ = self.network(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        self.states.append(state)
        self.actions.append(action)
        self.values.append(value)
        self.log_probs.append(dist.log_prob(action))
        
        return action.item()
    
    def store_reward(self, reward):
        self.rewards.append(reward)
        
    def update(self):
        if len(self.rewards) < self.n_steps:
            return
            
        # Calculate returns and advantages
        returns = []
        advantages = []
        R = 0
        
        for i in reversed(range(len(self.rewards))):
            R = self.rewards[i] + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.FloatTensor(returns)
        values = torch.cat(self.values)
        advantages = returns - values.squeeze()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate losses
        log_probs = torch.cat(self.log_probs)
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values.squeeze(), returns)
        
        loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        # Clear memory
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()

# MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
class MADDPGAgent:
    def __init__(self, n_agents, state_dim, action_dim, lr=1e-3):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Create actors and critics for each agent
        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        for i in range(n_agents):
            # Actor network
            actor = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )
            self.actors.append(actor)
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=lr))
            
            # Critic network (takes all states and actions)
            critic = nn.Sequential(
                nn.Linear(state_dim * n_agents + action_dim * n_agents, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            self.critics.append(critic)
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=lr))
            
            # Target networks
            target_actor = copy.deepcopy(actor)
            target_critic = copy.deepcopy(critic)
            self.target_actors.append(target_actor)
            self.target_critics.append(target_critic)
            
        self.memory = ReplayBuffer(capacity=100000)
        self.tau = 0.01  # Soft update parameter
        
    def get_actions(self, states):
        actions = []
        for i, state in enumerate(states):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = self.actors[i](state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            actions.append(action.item())
        return actions
    
    def update(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
            
        # This is a simplified version - full MADDPG requires more complex update logic
        # Including centralized training with decentralized execution
        pass
    
    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

# Communication Protocol for Multi-Agent Systems
class CommunicationProtocol:
    def __init__(self, n_agents, message_dim=16):
        self.n_agents = n_agents
        self.message_dim = message_dim
        self.message_encoder = nn.Linear(64, message_dim)
        self.message_decoder = nn.Linear(message_dim * n_agents, 64)
        
    def encode_message(self, hidden_state):
        """Encode agent's hidden state into a message"""
        return self.message_encoder(hidden_state)
    
    def aggregate_messages(self, messages):
        """Aggregate messages from all agents"""
        # Simple mean aggregation - can be replaced with attention mechanism
        return torch.mean(torch.stack(messages), dim=0)
    
    def decode_messages(self, aggregated_messages):
        """Decode aggregated messages back to hidden state"""
        return self.message_decoder(aggregated_messages)

# Hierarchical Multi-Agent System
class HierarchicalAgent:
    def __init__(self, role='worker', n_states=3, n_actions=4):
        self.role = role
        self.n_states = n_states
        self.n_actions = n_actions
        
        if role == 'leader':
            # Leader has more complex network
            self.policy = ActorCriticNetwork(n_states + 16, n_actions + 4)  # Extra actions for commands
            self.influence_radius = 10.0
        elif role == 'scout':
            # Scout has enhanced perception
            self.policy = ActorCriticNetwork(n_states + 8, n_actions)  # Extra state info
            self.exploration_bonus = 2.0
        elif role == 'guard':
            # Guard protects nest area
            self.policy = ActorCriticNetwork(n_states, n_actions)
            self.defense_radius = 5.0
        else:  # worker
            self.policy = ActorCriticNetwork(n_states, n_actions)
            
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        
    def get_action(self, state, leader_command=None):
        if self.role == 'worker' and leader_command is not None:
            # Workers influenced by leader commands
            state = np.concatenate([state, leader_command])
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, value, _ = self.policy(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), value

# Graph Neural Network for Ant Relationships
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features
        
        self.W = nn.Linear(in_features, out_features * n_heads, bias=False)
        self.a = nn.Parameter(torch.randn(n_heads, 2 * out_features))
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x, adj):
        B, N, F = x.shape  # Batch, Nodes, Features
        
        # Linear transformation
        h = self.W(x).view(B, N, self.n_heads, self.out_features)
        
        # Attention mechanism
        h_i = h.unsqueeze(2).repeat(1, 1, N, 1, 1)
        h_j = h.unsqueeze(1).repeat(1, N, 1, 1, 1)
        concat = torch.cat([h_i, h_j], dim=-1)
        
        e = torch.einsum('bnmhf,hf->bnmh', concat, self.a)
        e = self.leaky_relu(e)
        
        # Masked attention
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.unsqueeze(-1) > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        
        # Aggregate
        h_prime = torch.einsum('bnmh,bmhf->bnhf', attention, h)
        return h_prime.mean(dim=2)  # Average over heads

class GNNAgent:
    """Agent using Graph Neural Networks for multi-agent coordination"""
    def __init__(self, n_states, n_actions, n_agents):
        self.n_agents = n_agents
        self.gat = GraphAttentionLayer(n_states, 64)
        self.policy = ActorCriticNetwork(64, n_actions)
        self.optimizer = optim.Adam(list(self.gat.parameters()) + 
                                   list(self.policy.parameters()), lr=1e-3)
        
    def get_actions(self, states, adjacency_matrix):
        """Get actions for all agents considering their relationships"""
        states_tensor = torch.FloatTensor(states).unsqueeze(0)
        adj_tensor = torch.FloatTensor(adjacency_matrix).unsqueeze(0)
        
        # Process through GNN
        gnn_features = self.gat(states_tensor, adj_tensor)
        
        actions = []
        for i in range(self.n_agents):
            action_probs, _, _ = self.policy(gnn_features[0, i].unsqueeze(0))
            dist = Categorical(action_probs)
            action = dist.sample()
            actions.append(action.item())
            
        return actions

# Multi-Objective Optimization Wrapper
class MultiObjectiveAgent:
    def __init__(self, base_agent, objectives=['food', 'exploration', 'energy']):
        self.base_agent = base_agent
        self.objectives = objectives
        self.objective_weights = {obj: 1.0 for obj in objectives}
        
    def calculate_multi_objective_reward(self, state, action, env_info):
        """Calculate weighted sum of multiple objectives"""
        rewards = {}
        
        if 'food' in self.objectives:
            rewards['food'] = env_info.get('food_collected', 0) * 10
            
        if 'exploration' in self.objectives:
            rewards['exploration'] = env_info.get('new_area_explored', 0) * 5
            
        if 'energy' in self.objectives:
            # Penalize excessive movement
            rewards['energy'] = -env_info.get('distance_traveled', 0) * 0.1
            
        # Weighted sum
        total_reward = sum(rewards[obj] * self.objective_weights[obj] 
                          for obj in self.objectives)
        
        return total_reward, rewards
    
    def update_objective_weights(self, performance_metrics):
        """Dynamically adjust objective weights based on performance"""
        for obj in self.objectives:
            if performance_metrics[obj] < performance_metrics['target_' + obj]:
                self.objective_weights[obj] *= 1.1  # Increase weight
            else:
                self.objective_weights[obj] *= 0.9  # Decrease weight
                
        # Normalize weights
        total = sum(self.objective_weights.values())
        for obj in self.objectives:
            self.objective_weights[obj] /= total