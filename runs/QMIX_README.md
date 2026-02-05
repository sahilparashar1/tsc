# QMIX Multi-Agent Reinforcement Learning for Traffic Signal Control

## Overview

This implementation provides a **QMIX (Qmix Value Decomposition)** based multi-agent reinforcement learning (MARL) system for optimizing traffic signal timing in SUMO (Simulation of Urban Mobility). QMIX enables cooperative learning where multiple agents coordinate through a value function decomposition mechanism.

## Architecture

### QMIX Components

**1. Agent Networks (DRQN)**
- Each traffic light agent has its own **Deep Recurrent Q-Network (DRQN)**
- Uses GRU (Gated Recurrent Unit) cells to maintain temporal memory
- Input: Local observation (queue lengths, phase info, waiting times)
- Output: Q-values for each action (KEEP or SWITCH phase)
- Architecture: Input → Dense(hidden_dim) → ReLU → GRU → Dense(action_dim)

**2. Value Mixing Network (Hypernetwork)**
- Combines individual agent Q-values into a global **Q_tot** with monotonicity constraints
- Uses hypernetworks to generate mixing weights conditioned on global state
- Two-layer mixing:
  - First layer: Maps agent Q-values to intermediate representation
  - Second layer: Produces final Q_tot
- State value function allows non-monotonic components

**3. Experience Replay Buffer**
- Episode-based storage to maintain GRU state consistency
- Stores complete episodes with observations, actions, rewards, and states
- Allows mini-batch training with proper temporal structure

### Key Design Decisions

```
Multi-Agent Signal Control:
- Each agent manages ONE traffic light independently
- Agents coordinate through SHARED MIXING NETWORK
- Global state = concatenation of all agent observations
- Reward = -queue_length (encourages reducing congestion)
```

**Monotonic Value Factorization:**
```
Q_tot ≥ 0 when all Q_i ≥ 0  (monotonicity constraint)
This ensures greedy agent actions contribute to global optimality
```

## Installation

### Prerequisites
- Python 3.8+
- SUMO 1.12+ (for simulation)
- PyTorch 1.12+ (CPU or CUDA capable)

### Setup

```bash
# Install dependencies
pip install torch numpy traci

# Set SUMO_HOME environment variable
# Windows:
set SUMO_HOME=C:\Program Files\sumo  # or your SUMO installation path

# Linux:
export SUMO_HOME=/usr/share/sumo
```

### File Structure
```
runs/
├── run_qmix.py              # Main QMIX training script
├── run_qlearning.py         # Baseline Q-Learning (for comparison)
└── ...

cloud/src/agents/
├── qmix_network.py          # QMIX architecture implementation
├── controller_agent.py       # Distribution controller
├── regional_agent.py         # Regional agent
└── ...

Sioux/data/
├── config_qmix_sioux.ini    # QMIX configuration parameters
├── config_ma2c_Sioux.ini    # MA2C reference configuration
├── exp_0.sumocfg            # SUMO scenario configuration
└── ...
```

## Configuration

All training parameters can be configured at the top of `run_qmix.py`:

### Key Parameters

**Network Architecture:**
```python
AGENT_HIDDEN_DIM = 64           # Agent dense layer hidden dim
AGENT_RNN_HIDDEN_DIM = 64       # GRU hidden state dimension
MIXING_EMBED_DIM = 32           # Mixing network embedding
```

**Learning Hyperparameters:**
```python
LEARNING_RATE = 5e-4            # Adam optimizer learning rate
DISCOUNT_FACTOR = 0.99          # Gamma for future discounting
TARGET_UPDATE_FREQ = 200        # Update target network frequency

EPSILON_START = 1.0             # Initial exploration
EPSILON_MIN = 0.05              # Minimum exploration
EPSILON_DECAY = 0.995           # Exponential decay
```

**Training:**
```python
BATCH_SIZE = 32                 # Mini-batch size
REPLAY_BUFFER_CAPACITY = 5000   # Max episodes stored
TRAIN_STEPS_PER_EPISODE = 10    # Gradient steps per episode
NUM_EPISODES = 50               # Total training episodes
```

**Traffic Signal Control:**
```python
MIN_GREEN_TIME = 15             # Minimum green duration
MAX_GREEN_TIME = 120            # Maximum green duration
DECISION_INTERVAL = 2           # Steps between decisions
```

**Reward Function:**
```python
REWARD_SCALE = 1.0              # Global reward scale
QUEUE_WEIGHT = 1.0              # Queue length weight
FLICKERING_PENALTY = 0.1        # Phase change penalty
```

## Usage

### Basic Training

```bash
cd runs
python run_qmix.py
```

This will:
1. Initialize QMIX networks for all traffic lights in the scenario
2. Run training episodes with epsilon-greedy exploration
3. Perform mini-batch gradient updates after each episode
4. Decay exploration rate according to schedule
5. Save checkpoints and final results

### Output

During training, you'll see:
```
Episode 1/50  | Vehicles: 123 | Wait: 45.32s | Speed:  8.45m/s | Reward:  -234.5 | Loss:  2.34567 | ε: 0.995
Episode 2/50  | Vehicles: 125 | Wait: 42.18s | Speed:  8.92m/s | Reward:  -198.3 | Loss:  1.89234 | ε: 0.990
...
```

**Final Results:**
```
FINAL RESULTS - QMIX TRAFFIC SIGNAL CONTROL
===============================
Simulation Duration: 3600 steps
Total Vehicles: 847

--- Step-Based Metrics ---
Average Waiting Time: 35.42 seconds
Average Speed: 9.23 m/s (33.23 km/h)

--- Vehicle-Based Metrics ---
Average Accumulated Waiting Time: 38.21 seconds
Average Speed per Vehicle: 9.15 m/s (32.94 km/h)

--- Improvement vs Episode 1 ---
Waiting Time Change: -21.3%
Speed Change: +15.7%
```

### Checkpoints

Models are saved to `qmix_checkpoints/`:
- `qmix_episode_10.pt` - Checkpoint every SAVE_CHECKPOINT_INTERVAL episodes
- `qmix_best.pt` - Best model (highest episode reward)

Results are saved to `qmix_results/`:
- `qmix_training_results.json` - Complete training history and metrics

### Advanced Usage

#### Load and Continue Training

```python
# In run_qmix.py, after creating controller:
controller.load_checkpoint('./qmix_checkpoints/qmix_episode_20.pt')

# Training resumes from checkpoint
```

#### Evaluation Only

```python
# Set NUM_EPISODES = 1
# Set EPSILON_START = 0.0 (no exploration)
python run_qmix.py
```

#### Different Scenarios

Modify `SUMO_CONFIG` to use different SUMO scenarios:
```python
SUMO_CONFIG = r"..\Sioux\data\sioux.sumocfg"     # Full Sioux Falls
SUMO_CONFIG = r"..\runs\Grid_Network\Fixed_Timing.sumocfg"  # Grid network
```

## Algorithm Details

### QMIX Training Loop

```
1. EPISODE COLLECTION:
   - Initialize hidden states for all agent RNNs
   - For each step:
     a. Get agent observations from SUMO
     b. Select actions using epsilon-greedy policy (with RNN)
     c. Execute actions in SUMO
     d. Collect rewards from environment
     e. Store transition in replay buffer

2. GRADIENT UPDATE:
   - Sample batch of complete episodes from replay buffer
   - Forward pass through online network
   - Forward pass through target network
   - Compute TD targets: r + γ * Q_tot(s', a')
   - Backpropagate through mixing network
   - Gradient clipping and optimizer step

3. TARGET UPDATE:
   - Every TARGET_UPDATE_FREQ steps:
     * Copy online network weights to target network

4. EXPLORATION DECAY:
   - After each episode: ε = max(ε_min, ε * ε_decay)
```

### Action Constraints

Agents must respect traffic signal constraints:
- **Min Green**: Cannot SWITCH before MIN_GREEN_TIME steps
- **Max Green**: Must SWITCH after MAX_GREEN_TIME steps
- **Decision Rate**: Only make decisions every DECISION_INTERVAL steps

### Reward Function

**Max Pressure Reward** (with modifications):
```
R_t = -∑(queue_length_i) - flickering_penalty
```

Where:
- `queue_length_i` = vehicles waiting on incoming lanes to agent i
- `flickering_penalty` = penalty if phase changed from last decision

This encourages:
- Reducing vehicle congestion
- Stable phase patterns (avoiding constant switching)

### RNN Memory

GRU cells allow agents to:
- Remember previous phase history
- Learn temporal patterns in traffic flow
- Predict future queue changes based on time in phase

Example behaviors learned:
```
Observation: "Queue is building up during yellow phase"
→ Agent learns to switch sooner in future episodes

Observation: "Queue is decreasing during green phase"  
→ Agent learns to extend green phase
```

## Comparison with Q-Learning

### Q-Learning (Baseline)
- **State Space**: Discrete (binned queue levels, phase, time bin)
- **Storage**: Q-table (limited scalability)
- **Memory**: No temporal dependency
- **Scalability**: O(|S| × |A|) space complexity
- **Coordination**: Independent agents (no mixing)

### QMIX (Advanced)
- **State Space**: Continuous (raw observation values)
- **Storage**: Neural networks (scalable)
- **Memory**: RNN with temporal reasoning (GRU)
- **Scalability**: O(n_agents × hidden_dim) space complexity
- **Coordination**: Value decomposition mixing network

**When to Use Each:**
- **Q-Learning**: Small problems, discrete states, quick prototyping
- **QMIX**: Large problems, continuous observations, multi-agent coordination

## Hyperparameter Tuning Guide

### For Faster Convergence:
```python
LEARNING_RATE = 1e-3          # Increase learning rate
EPSILON_DECAY = 0.99          # Faster exploration decay
BATCH_SIZE = 64               # Larger batches
```

### For More Stable Training:
```python
LEARNING_RATE = 1e-4          # Decrease learning rate
EPSILON_DECAY = 0.998         # Slower exploration decay
AGENT_HIDDEN_DIM = 128        # Larger networks
TARGET_UPDATE_FREQ = 500      # Less frequent updates
```

### For Exploration:
```python
EPSILON_START = 1.0           # Start with full exploration
EPSILON_MIN = 0.1             # Higher minimum exploration
EPSILON_DECAY = 0.99          # Slower decay
```

### For Exploitation:
```python
EPSILON_START = 0.5           # Start with some exploitation
EPSILON_MIN = 0.01            # Lower minimum
EPSILON_DECAY = 0.95          # Fast decay
```

## Monitoring Training

### Key Metrics:

1. **Average Waiting Time** - Primary metric (lower is better)
   - Cumulative waiting time per vehicle
   - Goal: < 30 seconds per vehicle

2. **Average Speed** - Secondary metric (higher is better)
   - Typical values: 8-12 m/s
   - Improvement indicates smoother flow

3. **Episode Reward** - Training metric (higher is better)
   - Shows QMIX network's perceived improvement
   - May not perfectly correlate with waiting time

4. **Loss** - Convergence metric
   - Should decrease over time
   - High loss indicates network is still learning

5. **Epsilon (ε)** - Exploration rate
   - Starts at 1.0 (100% exploration)
   - Decays to EPSILON_MIN (5% exploration)
   - Controls exploration vs. exploitation trade-off

## Troubleshooting

### Issue: SUMO connection errors
```
Solution: Ensure SUMO_HOME is set correctly
$env:SUMO_HOME = "C:\Program Files\sumo"  # PowerShell
set SUMO_HOME=C:\Program Files\sumo       # CMD
```

### Issue: GPU out of memory
```
Solution: Reduce batch size or network dimensions
BATCH_SIZE = 16  (was 32)
AGENT_HIDDEN_DIM = 32  (was 64)
```

### Issue: Training doesn't improve
```
Solution: Check learning rate and decay schedule
Try: LEARNING_RATE = 1e-3, EPSILON_DECAY = 0.99
Or: Increase NUM_EPISODES for longer training
```

### Issue: Actions not changing during execution
```
Solution: Check epsilon - exploration may be too low
EPSILON_MIN = 0.1  (was 0.05)
Or: Reduce EPSILON_DECAY for slower decay
```

## File Structure Explained

```python
# run_qmix.py components:

TrafficObservationCollector
├─ get_agent_observation()      # Local observation per agent
├─ get_global_state()           # Concatenation of all obs
└─ get_incoming_queues()        # Queue info for rewards

QMIXController
├─ __init__()                   # Initialize networks
├─ select_actions()             # Epsilon-greedy + constraints
├─ execute_actions()            # Apply to SUMO
├─ compute_rewards()            # Max pressure rewards
└─ save/load_checkpoint()        # Persistence

Training Loop
├─ run_episode()                # Collect trajectory
├─ trainer.train_step()         # Gradient update
├─ trainer.replay_buffer.push() # Store episode
└─ train_qmix()                 # Main loop
```

## References

**QMIX Paper:**
- "QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent Reinforcement Learning" (Rashid et al., 2018)

**Related Work:**
- QPLEX, QTRAN (value decomposition variants)
- MAPPO, MADDPG (other MARL algorithms)
- DQN, DRQN (single-agent baselines)

## Future Extensions

1. **Algorithm Variants:**
   - QPLEX: Factorization with more expressive mixing
   - QTRAN: Alternative factorization approach
   - FA2C: Actor-Critic with value decomposition

2. **Multi-Intersection Coordination:**
   - Network communication graph
   - Hierarchical agents (regional controllers)
   - Attention mechanisms for agent interaction

3. **Real-World Deployment:**
   - SUMO-to-vehicle bridge (via API)
   - Uncertainty handling (missing observations)
   - Robustness to distribution shift

4. **Reward Shaping:**
   - Inverse max pressure (prefer clearance)
   - Equity metrics (fairness between directions)
   - Multi-objective rewards (throughput + comfort)

## Support

For issues or questions:
1. Check configuration parameters
2. Verify SUMO installation and environment
3. Review train/eval logs in json output
4. Check SUMO simulation validity

---

**Last Updated:** February 2026
