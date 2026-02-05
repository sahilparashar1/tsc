"""
QMIX Network Implementation
Monotonic Value Function Factorization for Multi-Agent Reinforcement Learning

Architecture:
- Agent Networks: DRQN with GRU cells
- Mixing Network: Hypernetwork for Q_tot computation
- Episode-based Replay Buffer for sequential data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict
from collections import deque
import random
import pickle


class DRQNAgent(nn.Module):
    """
    Deep Recurrent Q-Network Agent with GRU cells.
    Each agent processes local observations with temporal memory.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        rnn_hidden_dim: int = 64
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        
        # Feature extraction
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        
        # Recurrent layer (GRU for temporal dependencies)
        self.gru = nn.GRU(hidden_dim, rnn_hidden_dim, batch_first=True)
        
        # Q-value output
        self.fc2 = nn.Linear(rnn_hidden_dim, action_dim)
        
    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the agent network.
        
        Args:
            obs: Observation tensor [batch, seq_len, obs_dim]
            hidden_state: GRU hidden state [1, batch, rnn_hidden_dim]
            
        Returns:
            q_values: Q-values for each action [batch, seq_len, action_dim]
            new_hidden: Updated hidden state
        """
        batch_size = obs.shape[0]
        seq_len = obs.shape[1] if len(obs.shape) > 2 else 1
        
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(1)  # Add sequence dimension
            
        # Feature extraction
        x = F.relu(self.fc1(obs))  # [batch, seq_len, hidden_dim]
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = torch.zeros(1, batch_size, self.rnn_hidden_dim, device=obs.device)
            
        # Recurrent processing
        x, new_hidden = self.gru(x, hidden_state)  # [batch, seq_len, rnn_hidden_dim]
        
        # Q-value computation
        q_values = self.fc2(x)  # [batch, seq_len, action_dim]
        
        return q_values, new_hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state for the GRU."""
        return torch.zeros(1, batch_size, self.rnn_hidden_dim, device=device)


class QMIXMixingNetwork(nn.Module):
    """
    QMIX Mixing Network using Hypernetworks.
    Combines individual agent Q-values into a global Q_tot with monotonicity constraints.
    """
    
    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        mixing_embed_dim: int = 32,
        hypernet_embed_dim: int = 64
    ):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim
        
        # Hypernetwork for first mixing layer weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, n_agents * mixing_embed_dim)
        )
        
        # Hypernetwork for first mixing layer bias
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)
        
        # Hypernetwork for second mixing layer weights
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, mixing_embed_dim)
        )
        
        # Hypernetwork for final bias (V(s) term)
        self.hyper_v = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )
        
    def forward(
        self,
        agent_q_values: torch.Tensor,
        global_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Mix agent Q-values into Q_tot.
        
        Args:
            agent_q_values: Individual Q-values [batch, seq_len, n_agents]
            global_state: Global state [batch, seq_len, state_dim]
            
        Returns:
            q_tot: Global Q-value [batch, seq_len, 1]
        """
        batch_size = agent_q_values.shape[0]
        seq_len = agent_q_values.shape[1]
        
        # Reshape for processing
        agent_q = agent_q_values.view(batch_size * seq_len, 1, self.n_agents)
        state = global_state.view(batch_size * seq_len, self.state_dim)
        
        # First layer
        w1 = torch.abs(self.hyper_w1(state))  # Monotonicity constraint
        w1 = w1.view(-1, self.n_agents, self.mixing_embed_dim)
        b1 = self.hyper_b1(state).view(-1, 1, self.mixing_embed_dim)
        
        hidden = F.elu(torch.bmm(agent_q, w1) + b1)  # [batch*seq, 1, embed_dim]
        
        # Second layer
        w2 = torch.abs(self.hyper_w2(state))  # Monotonicity constraint
        w2 = w2.view(-1, self.mixing_embed_dim, 1)
        
        # State value term (allows non-monotonic offset)
        v = self.hyper_v(state).view(-1, 1, 1)
        
        # Final output
        q_tot = torch.bmm(hidden, w2) + v
        q_tot = q_tot.view(batch_size, seq_len, 1)
        
        return q_tot


class QMIXNetwork(nn.Module):
    """
    Complete QMIX architecture combining agents and mixing network.
    """
    
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int = None,
        hidden_dim: int = 64,
        rnn_hidden_dim: int = 64,
        mixing_embed_dim: int = 32
    ):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim or (obs_dim * n_agents)  # Default: concat of all obs
        self.rnn_hidden_dim = rnn_hidden_dim
        
        # Create agent networks (shared or individual)
        self.agents = nn.ModuleList([
            DRQNAgent(obs_dim, action_dim, hidden_dim, rnn_hidden_dim)
            for _ in range(n_agents)
        ])
        
        # Mixing network
        self.mixer = QMIXMixingNetwork(
            n_agents, self.state_dim, mixing_embed_dim
        )
        
    def forward(
        self,
        obs: torch.Tensor,
        global_state: torch.Tensor,
        actions: torch.Tensor,
        hidden_states: List[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through QMIX.
        
        Args:
            obs: Agent observations [batch, seq_len, n_agents, obs_dim]
            global_state: Global state [batch, seq_len, state_dim]
            actions: Actions taken [batch, seq_len, n_agents]
            hidden_states: List of hidden states for each agent
            
        Returns:
            q_tot: Global Q-value
            new_hidden_states: Updated hidden states
        """
        batch_size = obs.shape[0]
        seq_len = obs.shape[1]
        
        if hidden_states is None:
            hidden_states = [None] * self.n_agents
            
        # Get Q-values for each agent
        agent_q_values = []
        new_hidden_states = []
        
        for i, agent in enumerate(self.agents):
            agent_obs = obs[:, :, i, :]  # [batch, seq_len, obs_dim]
            q_vals, new_hidden = agent(agent_obs, hidden_states[i])
            
            # Select Q-value for taken action
            agent_actions = actions[:, :, i].unsqueeze(-1).long()
            chosen_q = torch.gather(q_vals, dim=-1, index=agent_actions).squeeze(-1)
            
            agent_q_values.append(chosen_q)
            new_hidden_states.append(new_hidden)
            
        # Stack agent Q-values
        agent_q_values = torch.stack(agent_q_values, dim=-1)  # [batch, seq_len, n_agents]
        
        # Mix into Q_tot
        q_tot = self.mixer(agent_q_values, global_state)
        
        return q_tot, new_hidden_states
    
    def get_agent_q_values(
        self,
        obs: torch.Tensor,
        hidden_states: List[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Get Q-values from all agents for action selection."""
        if hidden_states is None:
            hidden_states = [None] * self.n_agents
            
        all_q_values = []
        new_hidden_states = []
        
        for i, agent in enumerate(self.agents):
            agent_obs = obs[:, :, i, :]
            q_vals, new_hidden = agent(agent_obs, hidden_states[i])
            all_q_values.append(q_vals)
            new_hidden_states.append(new_hidden)
            
        return all_q_values, new_hidden_states


class EpisodeReplayBuffer:
    """
    Episode-based replay buffer for DRQN.
    Stores complete episodes to maintain GRU hidden state consistency.
    """
    
    def __init__(self, capacity: int, max_episode_length: int = 200):
        self.capacity = capacity
        self.max_episode_length = max_episode_length
        self.buffer = deque(maxlen=capacity)
        
    def push(self, episode: Dict):
        """
        Store an episode.
        
        Episode format:
        {
            'obs': [T, n_agents, obs_dim],
            'state': [T, state_dim],
            'actions': [T, n_agents],
            'rewards': [T],
            'next_obs': [T, n_agents, obs_dim],
            'next_state': [T, state_dim],
            'dones': [T]
        }
        """
        self.buffer.append(episode)
        
    def sample(self, batch_size: int) -> Dict:
        """Sample a batch of episodes."""
        episodes = random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
        
        # Find max length in batch
        max_len = max(ep['obs'].shape[0] for ep in episodes)
        
        # Pad and stack
        batch = {
            'obs': [],
            'state': [],
            'actions': [],
            'rewards': [],
            'next_obs': [],
            'next_state': [],
            'dones': [],
            'mask': []
        }
        
        for ep in episodes:
            ep_len = ep['obs'].shape[0]
            pad_len = max_len - ep_len
            
            # Create mask (1 for valid, 0 for padded)
            mask = np.concatenate([np.ones(ep_len), np.zeros(pad_len)])
            
            # Pad each array
            for key in ['obs', 'next_obs']:
                padded = np.pad(ep[key], ((0, pad_len), (0, 0), (0, 0)), mode='constant')
                batch[key].append(padded)
                
            for key in ['state', 'next_state']:
                padded = np.pad(ep[key], ((0, pad_len), (0, 0)), mode='constant')
                batch[key].append(padded)
                
            for key in ['actions']:
                padded = np.pad(ep[key], ((0, pad_len), (0, 0)), mode='constant')
                batch[key].append(padded)
                
            for key in ['rewards', 'dones']:
                padded = np.pad(ep[key], (0, pad_len), mode='constant')
                batch[key].append(padded)
                
            batch['mask'].append(mask)
            
        # Convert to tensors
        for key in batch:
            batch[key] = torch.FloatTensor(np.array(batch[key]))
            
        batch['actions'] = batch['actions'].long()
        
        return batch
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, path: str):
        """Save buffer to disk."""
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)
            
    def load(self, path: str):
        """Load buffer from disk."""
        with open(path, 'rb') as f:
            episodes = pickle.load(f)
            self.buffer = deque(episodes, maxlen=self.capacity)


class MaxPressureReward:
    """
    Max Pressure reward function for traffic signal control.
    
    Formula: R_i = -sum(Incoming Queues) + sum(Outgoing Space)
    Penalizes flickering (frequent phase changes).
    """
    
    def __init__(self, flickering_penalty: float = 0.1):
        self.flickering_penalty = flickering_penalty
        self.last_actions = {}
        
    def compute(
        self,
        intersection_id: str,
        incoming_queues: Dict[str, int],
        outgoing_capacity: Dict[str, int],
        current_action: int
    ) -> float:
        """
        Compute Max Pressure reward.
        
        Args:
            intersection_id: ID of the intersection
            incoming_queues: Queue lengths for incoming lanes
            outgoing_capacity: Available space on outgoing lanes
            current_action: Current signal phase action
            
        Returns:
            reward: Computed reward value
        """
        # Max Pressure component
        incoming_pressure = sum(incoming_queues.values())
        outgoing_space = sum(outgoing_capacity.values())
        
        base_reward = -incoming_pressure + outgoing_space
        
        # Flickering penalty
        penalty = 0.0
        if intersection_id in self.last_actions:
            if current_action != self.last_actions[intersection_id]:
                penalty = self.flickering_penalty
                
        self.last_actions[intersection_id] = current_action
        
        return base_reward - penalty
    
    def compute_batch(
        self,
        states: Dict,
        actions: torch.Tensor,
        last_actions: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute rewards for a batch of states.
        
        Args:
            states: Dictionary with 'incoming_queues' and 'outgoing_capacity'
            actions: Current actions [batch, n_agents]
            last_actions: Previous actions [batch, n_agents]
            
        Returns:
            rewards: Reward tensor [batch]
        """
        incoming = states['incoming_queues']  # [batch, n_agents, n_lanes]
        outgoing = states['outgoing_capacity']  # [batch, n_agents, n_lanes]
        
        # Sum across lanes
        incoming_sum = incoming.sum(dim=-1)  # [batch, n_agents]
        outgoing_sum = outgoing.sum(dim=-1)  # [batch, n_agents]
        
        # Max pressure reward per agent
        base_rewards = -incoming_sum + outgoing_sum
        
        # Flickering penalty
        if last_actions is not None:
            flickering = (actions != last_actions).float()
            penalties = self.flickering_penalty * flickering
            base_rewards = base_rewards - penalties
            
        # Sum across agents for global reward
        return base_rewards.sum(dim=-1)


class QMIXTrainer:
    """
    Trainer for QMIX algorithm.
    """
    
    def __init__(
        self,
        qmix_network: QMIXNetwork,
        lr: float = 5e-4,
        gamma: float = 0.99,
        target_update_freq: int = 200,
        grad_clip: float = 10.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip
        
        # Online and target networks
        self.qmix = qmix_network.to(self.device)
        self.target_qmix = QMIXNetwork(
            qmix_network.n_agents,
            qmix_network.obs_dim,
            qmix_network.action_dim,
            qmix_network.state_dim
        ).to(self.device)
        self.target_qmix.load_state_dict(self.qmix.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.qmix.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = EpisodeReplayBuffer(capacity=5000)
        
        # Reward function
        self.reward_fn = MaxPressureReward()
        
        self.update_count = 0
        
    def train_step(self, batch_size: int = 32) -> Optional[float]:
        """Perform one training step."""
        if len(self.replay_buffer) < batch_size:
            return None
            
        # Sample batch
        batch = self.replay_buffer.sample(batch_size)
        
        # Move to device
        obs = batch['obs'].to(self.device)
        state = batch['state'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        next_state = batch['next_state'].to(self.device)
        dones = batch['dones'].to(self.device)
        mask = batch['mask'].to(self.device)
        
        # Compute Q_tot for current state-action
        q_tot, _ = self.qmix(obs, state, actions)
        q_tot = q_tot.squeeze(-1)  # [batch, seq_len]
        
        # Compute target Q_tot
        with torch.no_grad():
            # Get max Q-values from target network for next state
            target_q_values, _ = self.target_qmix.get_agent_q_values(next_obs)
            
            # Greedy action selection
            target_actions = torch.stack([
                q.argmax(dim=-1) for q in target_q_values
            ], dim=-1)
            
            # Compute target Q_tot
            target_q_tot, _ = self.target_qmix(next_obs, next_state, target_actions)
            target_q_tot = target_q_tot.squeeze(-1)
            
            # TD target
            targets = rewards + self.gamma * (1 - dones) * target_q_tot
            
        # Masked loss (ignore padded timesteps)
        td_error = (q_tot - targets) * mask
        loss = (td_error ** 2).sum() / mask.sum()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qmix.parameters(), self.grad_clip)
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_qmix.load_state_dict(self.qmix.state_dict())
            
        return loss.item()
    
    def select_actions(
        self,
        obs: torch.Tensor,
        hidden_states: List[torch.Tensor] = None,
        epsilon: float = 0.0
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Select actions using epsilon-greedy policy."""
        with torch.no_grad():
            obs = obs.to(self.device)
            q_values, new_hidden = self.qmix.get_agent_q_values(obs, hidden_states)
            
            actions = []
            for q in q_values:
                if random.random() < epsilon:
                    action = torch.randint(0, q.shape[-1], (q.shape[0], q.shape[1]))
                else:
                    action = q.argmax(dim=-1)
                actions.append(action)
                
            return torch.stack(actions, dim=-1), new_hidden
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'qmix_state_dict': self.qmix.state_dict(),
            'target_state_dict': self.target_qmix.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count
        }, path)
        
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.qmix.load_state_dict(checkpoint['qmix_state_dict'])
        self.target_qmix.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint['update_count']
