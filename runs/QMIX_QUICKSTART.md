# QMIX Quick Start Guide

## 5-Minute Setup

### Step 1: Verify Dependencies
```bash
# Check Python version (3.8+)
python --version

# Install required packages
pip install torch numpy

# Verify SUMO installation
echo %SUMO_HOME%  # Windows
echo $SUMO_HOME   # Linux/Mac
```

### Step 2: Run Basic Training
```bash
cd runs
python run_qmix.py
```

This runs the default QMIX training with preset hyperparameters.

### Step 3: Check Results
```bash
# View results
type qmix_results\qmix_training_results.json  # Windows
cat qmix_results/qmix_training_results.json   # Linux

# Results include:
# - Final waiting times and speeds
# - Episode-by-episode improvements
# - Model configuration used
```

---

## 10-Minute Customization

### Option A: Modify Parameters Directly

Edit `runs/run_qmix.py` at the top:

```python
# Slower learning for stability
LEARNING_RATE = 1e-4          # Changed from 5e-4
BATCH_SIZE = 64               # Changed from 32

# Larger networks for complex scenarios
AGENT_HIDDEN_DIM = 128        # Changed from 64
AGENT_RNN_HIDDEN_DIM = 128    # Changed from 64

# Longer training
NUM_EPISODES = 100            # Changed from 50
```

Then run: `python run_qmix.py`

### Option B: Use Configuration File

Edit `Sioux\data\config_qmix_sioux.ini`:

```ini
[TRAINING_CONFIG]
learning_rate = 1e-4         # Slower learning
batch_size = 64              # Larger batches
num_episodes = 100           # More episodes
```

Then use utility to generate script:
```bash
python qmix_utils.py generate-script Sioux\data\config_qmix_sioux.ini
python run_qmix_configured.py
```

---

## Common Use Cases

### 1. Quick Prototype (5 min training)
```python
# In run_qmix.py:
NUM_EPISODES = 5
AGENT_HIDDEN_DIM = 32
BATCH_SIZE = 16
```

### 2. Stable Training (1-2 hours)
```python
LEARNING_RATE = 1e-4
EPSILON_DECAY = 0.998
NUM_EPISODES = 100
AGENT_HIDDEN_DIM = 128
```

### 3. Exploration Focus
```python
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.99  # Slower decay
TRAIN_STEPS_PER_EPISODE = 5
```

### 4. Exploitation Focus
```python
EPSILON_START = 0.5
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.93  # Fast decay
TRAIN_STEPS_PER_EPISODE = 20
```

---

## Monitoring Training

Watch the console output:

```
Episode 1/50  | Vehicles: 123 | Wait: 45.32s | Speed:  8.45m/s | Reward: -234.5 | ε: 0.995
             ↑                ↑              ↑                 ↑
          Episode #     Vehicle count   Metric to improve    Exploration rate
```

**Key metrics to watch:**
- **Wait**: Should decrease over time (goal: < 30s)
- **Speed**: Should increase over time (goal: > 8 m/s)  
- **ε**: Should decrease from 1.0 to ~0.05
- **Reward**: Trend should improve (become less negative)

**What's normal:**
- Episode 1: High waiting time, low speed
- Mid-training: Fluctuations as network learns
- Late training: Metrics stabilize (good sign!)
- If metrics don't improve: Check learning rate, try higher LEARNING_RATE

---

## Comparing Results

### After training, compare with Q-Learning:

```bash
# Assuming you have Q-Learning results
python qmix_utils.py compare qlearning_results.json qmix_results/qmix_training_results.json
```

Expected output:
```
Q-Learning Results:
  Average Waiting Time: 48.32s
  Average Speed: 7.85m/s

QMIX Results:
  Average Waiting Time: 41.21s
  Average Speed: 9.12m/s

Improvement (QMIX over Q-Learning):
  Waiting Time Reduction: +14.7%
  Speed Improvement: +16.2%
```

### View training trends:

```bash
python qmix_utils.py plot qmix_results/qmix_training_results.json
# Creates: training_curves.png (with 4 plots)
```

### Print detailed stats:

```bash
python qmix_utils.py stats qmix_results/qmix_training_results.json
```

---

## Troubleshooting

### "SUMO connection failed"
```python
# In run_qmix.py, set:
USE_GUI = False  # Disable GUI

# And check SUMO_HOME:
set SUMO_HOME=C:\Program%s(sumo_installation)
```

### "Out of memory"
```python
# In run_qmix.py, reduce:
BATCH_SIZE = 16  # was 32
AGENT_HIDDEN_DIM = 32  # was 64
REPLAY_BUFFER_CAPACITY = 2000  # was 5000
```

### "Training doesn't improve"
```python
# Try higher learning rate:
LEARNING_RATE = 1e-3  # was 5e-4

# Or more exploration:
EPSILON_MIN = 0.1  # was 0.05
EPSILON_DECAY = 0.99  # was 0.995

# Or longer training:
NUM_EPISODES = 100  # was 50
TRAIN_STEPS_PER_EPISODE = 20  # was 10
```

### "Simulation runs but metrics don't improve"
Check reward configuration:
```python
QUEUE_WEIGHT = 1.0      # Increase to 2.0 for stronger signal
FLICKERING_PENALTY = 0.1  # May be too high, try 0.05
```

---

## Understanding Output

### qmix_checkpoints/
- `qmix_episode_10.pt`: Model checkpoint at episode 10
- `qmix_episode_20.pt`: Model checkpoint at episode 20
- `qmix_best.pt`: Best model (highest episode reward)

Load a checkpoint:
```python
# In run_qmix.py, after creating controller:
controller.load_checkpoint('./qmix_checkpoints/qmix_best.pt')
```

### qmix_results/qmix_training_results.json

Structure:
```json
{
  "configuration": {
    "learning_rate": 0.0005,
    "num_agents": 24,
    ...
  },
  "final_results": {
    "avg_waiting_vehicle": 41.23,
    "avg_speed_vehicle": 9.12,
    ...
  },
  "episode_results": [
    { "avg_waiting_vehicle": 48.32, ... },
    { "avg_waiting_vehicle": 46.18, ... },
    ...
  ]
}
```

---

## Next Steps

### 1. Experiment with Different Scenarios
```python
# In run_qmix.py:
SUMO_CONFIG = r"..\Sioux\data\sioux.sumocfg"  # Try different scenario
```

### 2. Try Different Configurations
Edit `Sioux\data\config_qmix_sioux.ini` and use:
```bash
python qmix_utils.py generate-script config_qmix_sioux.ini
```

### 3. Load and Continue Training
```python
# Continue from checkpoint:
controller = QMIXController(tl_ids, obs_dim, state_dim)
controller.load_checkpoint('./qmix_checkpoints/qmix_best.pt')
# Then run training loop again
```

### 4. Analyze Results Programmatically
```python
from qmix_utils import ResultsAnalyzer

results = ResultsAnalyzer.load_results('qmix_results/qmix_training_results.json')
episodes = results['episode_results']

# Plot custom metrics
import matplotlib.pyplot as plt
waiting_times = [ep['avg_waiting_vehicle'] for ep in episodes]
plt.plot(waiting_times)
plt.show()
```

---

## Performance Expectations

### Typical Results (Sioux Falls Network)

**Q-Learning Baseline:**
- Initial waiting time: ~50s
- Final waiting time: ~40s
- Improvement: 20%

**QMIX:**
- Initial waiting time: ~45s
- Final waiting time: ~35s
- Improvement: 22%

**QMIX Advantages:**
- More stable learning curve
- Better coordination between agents
- Scalable to larger networks
- Learns temporal patterns through RNN

---

## Further Learning

### QMIX Paper:
"QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent Reinforcement Learning"
https://arxiv.org/abs/1803.11485

### Related Algorithms:
- **QPLEX**: More expressive mixing function
- **QTRAN**: Alternative value decomposition
- **MAPPO**: Actor-Critic approach
- **MADDPG**: Centralized critic

### Key Concepts:
1. **Value Decomposition**: Factorizing Q_tot into individual agent Q-values
2. **Monotonicity**: Ensuring greedy actions optimize global reward
3. **RNN Memory**: Temporal dependencies in traffic flow
4. **Replay Buffer**: Episode-based sampling for DRQN

---

## Common Questions

**Q: Should I use GPU or CPU?**
A: GPU recommended for faster training (10-20x speedup). Set DEVICE = 'cpu' if GPU unavailable.

**Q: How long does training take?**
A: Full training (50 episodes): 1-2 hours on GPU, 5-10 hours on CPU.

**Q: Can I use different SUMO scenarios?**
A: Yes! Just change SUMO_CONFIG path. Agent count and observation dim auto-detected.

**Q: What if results vary between runs?**
A: Normal due to randomness. Run multiple times and average results.

**Q: How do I interpret episode reward?**
A: Negative value = penalty being applied (good). Higher (less negative) = better.

**Q: Can I use pre-trained models?**
A: Yes! Load checkpoint with `controller.load_checkpoint(path)` before training.

---

## Support Commands

```bash
# View configuration
python qmix_utils.py print-config Sioux\data\config_qmix_sioux.ini

# Generate custom script from config
python qmix_utils.py generate-script Sioux\data\config_qmix_sioux.ini my_qmix.py

# Compare two results files
python qmix_utils.py compare ql_results.json qmix_results.json

# Plot training curves
python qmix_utils.py plot qmix_results/qmix_training_results.json my_plot.png

# Print detailed statistics
python qmix_utils.py stats qmix_results/qmix_training_results.json
```

---

**Ready to train?** → `python run_qmix.py`

**Questions?** → Check QMIX_README.md for detailed documentation
