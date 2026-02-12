"""
QMIX Multi-Agent Reinforcement Learning for Traffic Signal Control

QMIX (Qmix Value Decomposition) enables cooperative multi-agent learning by:
- Each agent has an independent DQN with recurrent memory (DRQN)
- A mixing network combines individual Q-values into Q_tot while enforcing monotonicity
- Max Pressure reward function guides agents toward system-level optimization

KEY DESIGN: Agents coordinate through value function decomposition
- Each agent decides independently based on local observations
- Mixing network ensures global reward is monotonically decomposable
- GRU memory allows agents to learn temporal dependencies
"""

import os
import sys
import random
import numpy as np
import torch
import json
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Optional
import traci

# Add workspace root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cloud.src.agents.qmix_network import (
    QMIXNetwork, DRQNAgent, QMIXMixingNetwork, 
    QMIXTrainer, EpisodeReplayBuffer, MaxPressureReward
)

# ============================================================================
# CONFIGURATION - Modify these parameters to experiment with training
# ============================================================================

# SUMO Configuration
SUMO_CONFIG = r"..\Sioux\data\network\exp.sumocfg"
USE_GUI = False  # Set to True to use sumo-gui

# QMIX Network Architecture
AGENT_HIDDEN_DIM = 64           # Dense layer hidden dimension for agent networks
AGENT_RNN_HIDDEN_DIM = 64       # GRU hidden dimension for temporal memory
MIXING_EMBED_DIM = 32           # Embedding dimension for mixing network
HYPERNET_EMBED_DIM = 64         # Hypernetwork embedding dimension

# QMIX Training Hyperparameters
LEARNING_RATE = 5e-4            # Adam learning rate
DISCOUNT_FACTOR = 0.99          # Gamma - future reward discounting
TARGET_UPDATE_FREQ = 200        # Update target network every N steps
GRAD_CLIP = 10.0                # Gradient clipping threshold
EPSILON_START = 1.0             # Initial exploration rate
EPSILON_MIN = 0.05              # Minimum exploration rate
EPSILON_DECAY = 0.995           # Exponential decay per episode

# Replay Buffer & Batch Training
REPLAY_BUFFER_CAPACITY = 5000   # Maximum episodes to store
BATCH_SIZE = 16                  # Batch size for training (reduced from 128 for 4GB GPU)
TRAIN_STEPS_PER_EPISODE = 10    # Number of training steps per episode (reduced to speed up)
MAX_EPISODE_LENGTH = 1800        # Truncate episodes at this length

# Traffic Light Configuration
MIN_GREEN_TIME = 15             # Minimum green time before considering switch
MAX_GREEN_TIME = 120            # Maximum green time before forced switch
DECISION_INTERVAL = 10          # Steps between decisions (increased from 2 to reduce TraCI overhead)
YELLOWTIME = 5                  # Yellow time duration

# Reward Configuration
REWARD_SCALE = 1.0              # Scale factor for rewards
FLICKERING_PENALTY = 0.1        # Penalty for frequent phase changes
QUEUE_WEIGHT = 1.0              # Weight for queue length in reward

# Training Configuration
NUM_EPISODES = 200               # Total training episodes
TEST_INTERVAL = 10              # Run evaluation every N episodes
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Checkpointing
SAVE_CHECKPOINT_INTERVAL = 10   # Save model every N episodes
CHECKPOINT_DIR = "./qmix_checkpoints"
RESULTS_DIR = "./qmix_results"

# Action constants
ACTION_KEEP = 0                 # Keep current phase
ACTION_SWITCH = 1               # Switch to next phase
NUM_ACTIONS = 2


# ============================================================================
# OBSERVATION & STATE COLLECTION
# ============================================================================

class TrafficObservationCollector:
    """Collects agent observations and global state from SUMO simulation."""
    
    def __init__(self, tl_ids: List[str]):
        self.tl_ids = tl_ids
        self.n_agents = len(tl_ids)
        self.tl_index = {tl_id: i for i, tl_id in enumerate(tl_ids)}
        
        # These will be computed lazily on first use
        self.max_lanes = None
        self.max_phases = None
    
    def _compute_dimensions(self):
        """Compute max lanes and phases from current SUMO state."""
        if self.max_lanes is not None:
            return  # Already computed
        
        self.max_lanes = 0
        self.max_phases = 0
        for tl_id in self.tl_ids:
            try:
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                unique_lanes = len(set(controlled_lanes))
                self.max_lanes = max(self.max_lanes, unique_lanes)
                
                program = traci.trafficlight.getAllProgramLogics(tl_id)[0]
                num_phases = len(program.phases)
                self.max_phases = max(self.max_phases, num_phases)
            except Exception as e:
                print(f"Warning: Could not get TL info for {tl_id}: {e}")
        
    def get_agent_observation(self, tl_id: str) -> np.ndarray:
        """
        Get local observation for an agent (traffic light).
        
        Observation includes:
        - Queue lengths on incoming lanes (binned, padded to max_lanes)
        - Current phase (one-hot encoded, padded to max_phases)
        - Waiting time (normalized)
        
        Returns: observation vector of fixed shape (obs_dim,)
        """
        # Compute dimensions on first call
        self._compute_dimensions()
        
        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            unique_lanes = list(set(controlled_lanes))
            
            # Queue levels (binned: 0-3 for empty, light, medium, heavy)
            queue_levels = []
            for lane in unique_lanes:
                queue = traci.lane.getLastStepHaltingNumber(lane)
                if queue == 0:
                    queue_levels.append(0)
                elif queue <= 3:
                    queue_levels.append(1)
                elif queue <= 7:
                    queue_levels.append(2)
                else:
                    queue_levels.append(3)
            
            # Pad queue levels to max_lanes
            queue_levels = queue_levels + [0] * (self.max_lanes - len(queue_levels))
            queue_levels = queue_levels[:self.max_lanes]
            
            # Current phase (one-hot encoded)
            current_phase = traci.trafficlight.getPhase(tl_id)
            program = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            num_phases = len(program.phases)
            phase_one_hot = [1.0 if i == current_phase else 0.0 for i in range(num_phases)]
            
            # Pad phase one-hot to max_phases
            phase_one_hot = phase_one_hot + [0.0] * (self.max_phases - len(phase_one_hot))
            phase_one_hot = phase_one_hot[:self.max_phases]
            
            # Average waiting time (normalized)
            waiting_time = 0.0
            for lane in unique_lanes:
                waiting_time += traci.lane.getWaitingTime(lane)
            waiting_time = min(waiting_time / 100.0, 1.0)  # Normalize
            
            # Concatenate features
            observation = np.array(queue_levels + phase_one_hot + [waiting_time], dtype=np.float32)
            
            return observation
            
        except Exception as e:
            print(f"Error collecting observation for {tl_id}: {e}")
            # Return zero observation with correct size
            return np.zeros(self.max_lanes + self.max_phases + 1, dtype=np.float32)
    
    def get_global_state(self) -> np.ndarray:
        """
        Get global state (concatenation of all agent observations).
        
        Returns: state vector of shape (state_dim,) where state_dim = obs_dim * n_agents
        """
        state_list = []
        for tl_id in self.tl_ids:
            obs = self.get_agent_observation(tl_id)
            state_list.append(obs)
        
        return np.concatenate(state_list, axis=0).astype(np.float32)
    
    def get_incoming_queues(self) -> Dict[str, int]:
        """Get incoming queue lengths for each agent."""
        queues = {}
        for tl_id in self.tl_ids:
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            unique_lanes = list(set(controlled_lanes))
            total_queue = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in unique_lanes)
            queues[tl_id] = total_queue
        return queues


# ============================================================================
# QMIX AGENT CONTROLLER
# ============================================================================

class QMIXController:
    """Controller managing QMIX multi-agent training."""
    
    def __init__(self, tl_ids: List[str], obs_dim: int, state_dim: int):
        self.tl_ids = tl_ids
        self.n_agents = len(tl_ids)
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        
        # Initialize QMIX network
        self.qmix_network = QMIXNetwork(
            n_agents=self.n_agents,
            obs_dim=obs_dim,
            action_dim=NUM_ACTIONS,
            state_dim=state_dim,
            hidden_dim=AGENT_HIDDEN_DIM,
            rnn_hidden_dim=AGENT_RNN_HIDDEN_DIM,
            mixing_embed_dim=MIXING_EMBED_DIM
        )
        
        # Initialize trainer
        self.trainer = QMIXTrainer(
            qmix_network=self.qmix_network,
            lr=LEARNING_RATE,
            gamma=DISCOUNT_FACTOR,
            target_update_freq=TARGET_UPDATE_FREQ,
            grad_clip=GRAD_CLIP,
            device=DEVICE
        )
        
        # Observation collector
        self.obs_collector = TrafficObservationCollector(tl_ids)
        
        # Agent state tracking
        self.current_phase_times = {tl_id: 0 for tl_id in tl_ids}
        self.last_waiting_times = {tl_id: 0.0 for tl_id in tl_ids}
        self.hidden_states = None
        
        # Reward function
        self.reward_fn = MaxPressureReward(flickering_penalty=FLICKERING_PENALTY)
        
        # Epsilon for exploration
        self.epsilon = EPSILON_START
        
        print(f"Initialized QMIX Controller with {self.n_agents} agents")
        print(f"Observation dimension: {obs_dim}")
        print(f"State dimension: {state_dim}")
        print(f"Device: {DEVICE}")
    
    def get_observations_batch(self) -> torch.Tensor:
        """
        Get observations for all agents as a batch tensor.
        
        Returns: tensor of shape (1, 1, n_agents, obs_dim)
        """
        observations = []
        for tl_id in self.tl_ids:
            obs = self.obs_collector.get_agent_observation(tl_id)
            observations.append(obs)
        
        obs_stack = np.stack(observations, axis=0)  # [n_agents, obs_dim]
        obs_batch = torch.FloatTensor(obs_stack).unsqueeze(0).unsqueeze(0)  # [1, 1, n_agents, obs_dim]
        
        return obs_batch
    
    def get_global_state_batch(self) -> torch.Tensor:
        """
        Get global state as a batch tensor.
        
        Returns: tensor of shape (1, 1, state_dim)
        """
        state = self.obs_collector.get_global_state()
        state_batch = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
        
        return state_batch
    
    def select_actions(self) -> Tuple[Dict[str, int], torch.Tensor]:
        """
        Select actions using epsilon-greedy policy with constraints.
        
        Returns:
            action_dict: Dict mapping tl_id to action (0=KEEP, 1=SWITCH)
            action_tensor: Tensor of shape [1, 1, n_agents]
        """
        obs_batch = self.get_observations_batch()
        
        with torch.no_grad():
            actions_tensor, self.hidden_states = self.trainer.select_actions(
                obs_batch, 
                hidden_states=self.hidden_states,
                epsilon=self.epsilon
            )
        
        action_dict = {}
        for i, tl_id in enumerate(self.tl_ids):
            # Apply constraints
            if self.current_phase_times[tl_id] >= MAX_GREEN_TIME:
                action = ACTION_SWITCH
            elif self.current_phase_times[tl_id] < MIN_GREEN_TIME:
                action = ACTION_KEEP
            else:
                action = actions_tensor[0, 0, i].item()
            
            action_dict[tl_id] = action
        
        return action_dict, actions_tensor
    
    def execute_actions(self, action_dict: Dict[str, int]):
        """Execute selected actions in SUMO."""
        for tl_id, action in action_dict.items():
            if action == ACTION_SWITCH:
                try:
                    current_phase = traci.trafficlight.getPhase(tl_id)
                    program = traci.trafficlight.getAllProgramLogics(tl_id)[0]
                    num_phases = len(program.phases)
                    next_phase = (current_phase + 1) % num_phases
                    traci.trafficlight.setPhase(tl_id, next_phase)
                    self.current_phase_times[tl_id] = 0
                except Exception as e:
                    print(f"Error executing action for {tl_id}: {e}")
            
            self.current_phase_times[tl_id] += DECISION_INTERVAL
    
    def compute_rewards(self) -> np.ndarray:
        """
        Compute rewards for all agents using Max Pressure function.
        
        Returns: reward array of shape (n_agents,)
        """
        incoming_queues = self.obs_collector.get_incoming_queues()
        
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        for i, tl_id in enumerate(self.tl_ids):
            queue = incoming_queues.get(tl_id, 0)
            # Simple reward: negative of queue length
            reward = -queue * QUEUE_WEIGHT * REWARD_SCALE
            rewards[i] = reward
        
        return rewards
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        self.trainer.save(path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        self.trainer.load(path)


# ============================================================================
# EPISODE EXECUTION
# ============================================================================

def run_episode(controller: QMIXController, episode: int, learning: bool = True) -> Dict:
    """Run a single QMIX training episode."""
    
    if 'SUMO_HOME' not in os.environ:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
    
    sumo_binary = "sumo-gui" if USE_GUI else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-c", SUMO_CONFIG,
        "--no-step-log",
        "--waiting-time-memory", "1000",
        "--random",
        "--threads", "4",                  # Enable multithreading (adjust to your CPU cores)
        "--step-log.period", "1000",       # Reduce logging frequency
    ]
    
    traci.start(sumo_cmd)
    
    # Reset episode state
    controller.hidden_states = [None] * controller.n_agents
    for tl_id in controller.tl_ids:
        controller.current_phase_times[tl_id] = 0
        controller.last_waiting_times[tl_id] = 0.0
    
    # Episode trajectory
    episode_data = {
        'obs': [],
        'state': [],
        'actions': [],
        'rewards': [],
        'next_obs': [],
        'next_state': [],
        'dones': []
    }
    
    # Metrics
    total_waiting_time = 0.0
    total_speed = 0.0
    total_vehicle_steps = 0
    all_vehicles = set()
    vehicle_waiting_times = {}
    vehicle_speeds = {}
    
    step = 0
    episode_reward = 0.0
    
    try:
        while traci.simulation.getMinExpectedNumber() > 0 and step < MAX_EPISODE_LENGTH * 2:
            traci.simulationStep()
            step += 1
            
            if step % DECISION_INTERVAL == 0:
                # Get current state and obs
                obs = controller.get_observations_batch()
                state = controller.get_global_state_batch()
                
                # Select and execute actions
                action_dict, action_tensor = controller.select_actions()
                controller.execute_actions(action_dict)
                
                # Collect next state
                next_obs = controller.get_observations_batch()
                next_state = controller.get_global_state_batch()
                
                # Compute rewards
                rewards = controller.compute_rewards()
                reward = rewards.sum()  # Global reward
                episode_reward += reward
                
                # Check if done
                done = traci.simulation.getMinExpectedNumber() == 0
                
                # Store trajectory
                episode_data['obs'].append(obs.cpu().numpy())
                episode_data['state'].append(state.cpu().numpy())
                episode_data['actions'].append(action_tensor.cpu().numpy())
                episode_data['rewards'].append(np.array([reward], dtype=np.float32))
                episode_data['next_obs'].append(next_obs.cpu().numpy())
                episode_data['next_state'].append(next_state.cpu().numpy())
                episode_data['dones'].append(np.array([done], dtype=np.float32))
                
                # Collect metrics only at decision intervals to reduce TraCI overhead
                current_vehicles = traci.vehicle.getIDList()
                for veh_id in current_vehicles:
                    all_vehicles.add(veh_id)
                    vehicle_waiting_times[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                    speed = traci.vehicle.getSpeed(veh_id)
                    if veh_id not in vehicle_speeds:
                        vehicle_speeds[veh_id] = []
                    vehicle_speeds[veh_id].append(speed)
    
    except (KeyboardInterrupt, traci.exceptions.FatalTraCIError) as e:
        print(f"Simulation error: {e}")
    finally:
        traci.close()
    
    # Calculate step-based metrics from collected vehicle data
    # (Only collected at decision intervals, not every step)
    if vehicle_speeds:
        total_steps_with_metrics = sum(len(speeds) for speeds in vehicle_speeds.values())
        if total_steps_with_metrics > 0:
            total_speed = sum(sum(speeds) for speeds in vehicle_speeds.values())
            for speeds in vehicle_speeds.values():
                for _ in speeds:
                    total_vehicle_steps += 1
    
    # Process trajectory for storage
    if learning and len(episode_data['obs']) > 0:
        # Stack trajectory
        trajectory = {
            'obs': np.concatenate(episode_data['obs'], axis=1),  # [1, T, n_agents, obs_dim]
            'state': np.concatenate(episode_data['state'], axis=1),  # [1, T, state_dim]
            'actions': np.concatenate(episode_data['actions'], axis=1),  # [1, T, n_agents]
            'rewards': np.concatenate(episode_data['rewards'], axis=0),  # [T]
            'next_obs': np.concatenate(episode_data['next_obs'], axis=1),  # [1, T, n_agents, obs_dim]
            'next_state': np.concatenate(episode_data['next_state'], axis=1),  # [1, T, state_dim]
            'dones': np.concatenate(episode_data['dones'], axis=0),  # [T]
        }
        
        # Remove batch dimension
        for key in ['obs', 'state', 'actions', 'next_obs', 'next_state']:
            trajectory[key] = trajectory[key].squeeze(0)
        
        # Add to replay buffer
        controller.trainer.replay_buffer.push(trajectory)
    
    # Calculate metrics
    results = {
        "steps": step,
        "vehicles": len(all_vehicles),
        "avg_waiting_step": total_waiting_time / total_vehicle_steps if total_vehicle_steps > 0 else 0,
        "avg_speed_step": total_speed / total_vehicle_steps if total_vehicle_steps > 0 else 0,
        "episode_reward": episode_reward,
    }
    
    if len(all_vehicles) > 0:
        total_accumulated_waiting = sum(vehicle_waiting_times.values())
        results["avg_waiting_vehicle"] = total_accumulated_waiting / len(all_vehicles)
        
        vehicle_avg_speeds = [sum(speeds) / len(speeds) for speeds in vehicle_speeds.values() if speeds]
        results["avg_speed_vehicle"] = sum(vehicle_avg_speeds) / len(vehicle_avg_speeds) if vehicle_avg_speeds else 0
    else:
        results["avg_waiting_vehicle"] = 0
        results["avg_speed_vehicle"] = 0
    
    return results


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_qmix():
    """Main QMIX training loop."""
    
    print("=" * 80)
    print("QMIX: MULTI-AGENT REINFORCEMENT LEARNING FOR TRAFFIC SIGNAL CONTROL")
    print("=" * 80)
    print("\nValue Function Decomposition Architecture:")
    print("  - Agent Networks: DRQN with GRU cells for temporal memory")
    print("  - Mixing Network: Hypernetwork with monotonicity constraints")
    print("  - Reward Function: Max Pressure reward with flickering penalty")
    print("\n" + "-" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Discount Factor: {DISCOUNT_FACTOR}")
    print(f"  Target Update Frequency: {TARGET_UPDATE_FREQ}")
    print(f"  Agent Hidden Dimension: {AGENT_HIDDEN_DIM}")
    print(f"  RNN Hidden Dimension: {AGENT_RNN_HIDDEN_DIM}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Training Steps per Episode: {TRAIN_STEPS_PER_EPISODE}")
    print(f"  Device: {DEVICE}")
    print(f"  Total Episodes: {NUM_EPISODES}")
    print("-" * 80 + "\n")
    
    # Create output directories
    Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    
    # First episode to get network dimensions
    if 'SUMO_HOME' not in os.environ:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
    
    sumo_binary = "sumo"
    sumo_cmd = [
        sumo_binary, "-c", SUMO_CONFIG, 
        "--no-step-log",
        "--threads", "4",
    ]
    traci.start(sumo_cmd)
    
    tl_ids = traci.trafficlight.getIDList()
    collector = TrafficObservationCollector(tl_ids)
    
    # Sample observations to determine dimensions
    sample_obs = collector.get_agent_observation(tl_ids[0])
    sample_state = collector.get_global_state()
    obs_dim = len(sample_obs)
    state_dim = len(sample_state)
    
    traci.close()
    
    print(f"Detected {len(tl_ids)} traffic lights")
    print(f"Observation dimension: {obs_dim}")
    print(f"State dimension: {state_dim}\n")
    
    # Initialize controller
    controller = QMIXController(tl_ids, obs_dim, state_dim)
    
    # Training
    episode_results = []
    best_reward = -float('inf')
    
    for episode in range(NUM_EPISODES):
        # Run episode
        results = run_episode(controller, episode, learning=True)
        episode_results.append(results)
        
        # Training steps
        if len(controller.trainer.replay_buffer) > 0:
            total_loss = 0.0
            for _ in range(TRAIN_STEPS_PER_EPISODE):
                loss = controller.trainer.train_step(batch_size=BATCH_SIZE)
                if loss is not None:
                    total_loss += loss
            avg_loss = total_loss / TRAIN_STEPS_PER_EPISODE if total_loss > 0 else 0
        else:
            avg_loss = 0.0
        
        # Decay exploration
        controller.decay_epsilon()
        
        # Clear GPU cache every episode to prevent memory leak
        controller.trainer.clear_cache()
        
        # Logging
        print(f"Episode {episode + 1}/{NUM_EPISODES} | "
              f"Vehicles: {results['vehicles']:3d} | "
              f"Wait: {results['avg_waiting_vehicle']:7.2f}s | "
              f"Speed: {results['avg_speed_vehicle']:6.2f}m/s | "
              f"Reward: {results['episode_reward']:7.3f} | "
              f"Loss: {avg_loss:7.5f} | "
              f"ε: {controller.epsilon:.3f}")
        
        # Save checkpoint
        if (episode + 1) % SAVE_CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"qmix_episode_{episode + 1}.pt")
            controller.save_checkpoint(checkpoint_path)
            print(f"  → Checkpoint saved to {checkpoint_path}")
        
        # Track best model
        if results['episode_reward'] > best_reward:
            best_reward = results['episode_reward']
            best_checkpoint_path = os.path.join(CHECKPOINT_DIR, "qmix_best.pt")
            controller.save_checkpoint(best_checkpoint_path)
    
    # Final evaluation
    print("\n" + "-" * 80)
    print("Running final evaluation (no exploration)...")
    
    controller.epsilon = 0.0
    final_results = run_episode(controller, NUM_EPISODES, learning=False)
    
    # Print results
    print("\n" + "=" * 80)
    print("FINAL RESULTS - QMIX TRAFFIC SIGNAL CONTROL")
    print("=" * 80)
    
    print(f"\nSimulation Duration: {final_results['steps']} steps")
    print(f"Total Vehicles: {final_results['vehicles']}")
    
    print("\n--- Step-Based Metrics ---")
    print(f"Average Waiting Time: {final_results['avg_waiting_step']:.2f} seconds")
    print(f"Average Speed: {final_results['avg_speed_step']:.2f} m/s ({final_results['avg_speed_step'] * 3.6:.2f} km/h)")
    
    print("\n--- Vehicle-Based Metrics ---")
    print(f"Average Accumulated Waiting Time: {final_results['avg_waiting_vehicle']:.2f} seconds")
    print(f"Average Speed per Vehicle: {final_results['avg_speed_vehicle']:.2f} m/s ({final_results['avg_speed_vehicle'] * 3.6:.2f} km/h)")
    
    print("\n--- Episode Reward ---")
    print(f"Final Episode Reward: {final_results['episode_reward']:.3f}")
    print(f"Best Episode Reward: {best_reward:.3f}")
    
    # Compare with first episode
    if episode_results:
        first = episode_results[0]
        print("\n--- Improvement vs Episode 1 ---")
        if first['avg_waiting_vehicle'] > 0:
            wait_change = (first['avg_waiting_vehicle'] - final_results['avg_waiting_vehicle']) / first['avg_waiting_vehicle'] * 100
        else:
            wait_change = 0
        if first['avg_speed_vehicle'] > 0:
            speed_change = (final_results['avg_speed_vehicle'] - first['avg_speed_vehicle']) / first['avg_speed_vehicle'] * 100
        else:
            speed_change = 0
        print(f"Waiting Time Change: {wait_change:+.1f}%")
        print(f"Speed Change: {speed_change:+.1f}%")
    
    print("\n" + "=" * 80)
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80 + "\n")
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Save results to JSON
    results_file = os.path.join(RESULTS_DIR, "qmix_training_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'configuration': {
                'learning_rate': LEARNING_RATE,
                'discount_factor': DISCOUNT_FACTOR,
                'target_update_freq': TARGET_UPDATE_FREQ,
                'agent_hidden_dim': AGENT_HIDDEN_DIM,
                'agent_rnn_hidden_dim': AGENT_RNN_HIDDEN_DIM,
                'batch_size': BATCH_SIZE,
                'num_episodes': NUM_EPISODES,
                'num_agents': len(tl_ids),
                'obs_dim': obs_dim,
                'state_dim': state_dim
            },
            'final_results': convert_to_native(final_results),
            'episode_results': convert_to_native(episode_results)
        }, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    return controller, final_results


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"Working directory: {os.getcwd()}")
    print(f"Configuration: {SUMO_CONFIG}\n")
    
    controller, results = train_qmix()
