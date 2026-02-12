"""
Q-Learning Traffic Signal Control for SUMO
Independent Q-learning agent at each intersection to optimize signal timing.

KEY DESIGN: Agents only decide WHEN to switch to the next phase, not WHICH phase.
Phases are cycled through sequentially to ensure safe, conflict-free transitions.

Action space:
  - 0: KEEP current phase (extend green time)
  - 1: SWITCH to next phase (proceed to next safe phase in cycle)
"""

import os
import sys
import random
import numpy as np
from collections import defaultdict
import traci

# Configuration
SUMO_CONFIG = r"..\Sioux\data\netowrk\exp.sumocfg"
USE_GUI = False  # Set to True to use sumo-gui

# Q-Learning Hyperparameters
LEARNING_RATE = 0.1          # Alpha
DISCOUNT_FACTOR = 0       # Gamma
EPSILON_START = 1.0          # Initial exploration rate
EPSILON_MIN = 0.01           # Minimum exploration rate
EPSILON_DECAY = 0.995        # Decay rate per episode

# Traffic Light Configuration
MIN_GREEN_TIME = 15          # Minimum green time before considering switch
MAX_GREEN_TIME = 120          # Maximum green time before forced switch
YELLOW_TIME = 5              # Yellow time (handled by SUMO)
DECISION_INTERVAL = 2        # Steps between decisions

# Training Configuration
NUM_EPISODES = 10            # Number of training episodes

# Action constants
ACTION_KEEP = 0    # Keep current phase
ACTION_SWITCH = 1  # Switch to next phase
NUM_ACTIONS = 2


class QLearningAgent:
    """
    Independent Q-learning agent for a single traffic light.
    
    The agent only decides WHEN to switch (timing), not which phase to go to.
    Phases are always cycled in their predefined safe order.
    """
    
    def __init__(self, tl_id):
        self.tl_id = tl_id
        self.q_table = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.epsilon = EPSILON_START
        
        # Get phase information from SUMO
        program = traci.trafficlight.getAllProgramLogics(tl_id)[0]
        self.num_phases = len(program.phases)
        self.phase_durations = [phase.duration for phase in program.phases]
        
        # Track state for learning
        self.last_state = None
        self.last_action = None
        self.current_phase_time = 0
        self.last_waiting_time = 0
        
    def get_state(self):
        """
        Get the current state for this traffic light.
        State: (queue_levels_per_direction, current_phase, time_in_phase_binned)
        """
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        unique_lanes = list(set(controlled_lanes))
        
        # Get queue lengths per lane, binned
        queue_levels = []
        for lane in unique_lanes:
            queue = traci.lane.getLastStepHaltingNumber(lane)
            # Bin: 0=empty, 1=light(1-3), 2=medium(4-7), 3=heavy(8+)
            if queue == 0:
                queue_levels.append(0)
            elif queue <= 3:
                queue_levels.append(1)
            elif queue <= 7:
                queue_levels.append(2)
            else:
                queue_levels.append(3)
        
        current_phase = traci.trafficlight.getPhase(self.tl_id)
        
        # Bin time in phase: 0=short(<15s), 1=medium(15-30s), 2=long(30-45s), 3=very_long(45s+)
        if self.current_phase_time < 15:
            time_bin = 0
        elif self.current_phase_time < 30:
            time_bin = 1
        elif self.current_phase_time < 45:
            time_bin = 2
        else:
            time_bin = 3
        
        return (tuple(queue_levels), current_phase, time_bin)
    
    def get_waiting_time(self):
        """Get total waiting time for vehicles at this intersection."""
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        unique_lanes = list(set(controlled_lanes))
        
        total_waiting = 0
        for lane in unique_lanes:
            total_waiting += traci.lane.getWaitingTime(lane)
        
        return total_waiting
    
    def get_queue_length(self):
        """Get total queue length at this intersection."""
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        unique_lanes = list(set(controlled_lanes))
        
        total_queue = 0
        for lane in unique_lanes:
            total_queue += traci.lane.getLastStepHaltingNumber(lane)
        
        return total_queue
    
    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        
        Actions:
          0 = KEEP: Stay in current phase
          1 = SWITCH: Move to next phase in cycle
        """
        # Force switch if at max green time
        if self.current_phase_time >= MAX_GREEN_TIME:
            return ACTION_SWITCH
            
        # Don't allow switch if below minimum green time
        if self.current_phase_time < MIN_GREEN_TIME:
            return ACTION_KEEP
        
        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        """Update Q-value using Q-learning update rule."""
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


class TrafficLightController:
    """Controller managing all Q-learning agents with safe phase cycling."""
    
    def __init__(self):
        self.agents = {}
        self.tl_ids = []
        
    def initialize(self):
        """Initialize agents for all traffic lights."""
        self.tl_ids = traci.trafficlight.getIDList()
        
        for tl_id in self.tl_ids:
            self.agents[tl_id] = QLearningAgent(tl_id)
            
        print(f"Initialized {len(self.agents)} Q-learning agents")
        print(f"Traffic lights: {self.tl_ids}")
        print(f"Action space: KEEP (extend) or SWITCH (to next phase)")
    
    def get_reward(self, tl_id):
        """
        Calculate reward for a traffic light.
        
        Reward is based on reduction in waiting time (positive = good).
        """
        agent = self.agents[tl_id]
        current_waiting = agent.get_waiting_time()
        queue_length = agent.get_queue_length()
        
        # Reward is the improvement in waiting time
        waiting_change = agent.last_waiting_time - current_waiting
        
        # Small penalty for very long queues
        queue_penalty = -0.1 * queue_length
        
        # Update last waiting time
        agent.last_waiting_time = current_waiting
        
        return waiting_change + queue_penalty
    
    def step(self, learning=True):
        """Execute one decision step for all agents."""
        for tl_id in self.tl_ids:
            agent = self.agents[tl_id]
            agent.current_phase_time += DECISION_INTERVAL
            
            # Get current state
            current_state = agent.get_state()
            
            # Calculate reward and update Q-values from previous action
            if learning and agent.last_state is not None and agent.last_action is not None:
                reward = self.get_reward(tl_id)
                agent.update(agent.last_state, agent.last_action, reward, current_state)
            
            # Select action
            action = agent.select_action(current_state)
            
            # Execute action
            if action == ACTION_SWITCH:
                # Switch to next phase in the predefined cycle
                current_phase = traci.trafficlight.getPhase(tl_id)
                # Next phase logic: just increment and wrap modulo the number of phases in current program
                next_phase = (current_phase + 1) % agent.num_phases
                try:
                    traci.trafficlight.setPhase(tl_id, next_phase)
                    agent.current_phase_time = 0
                except traci.exceptions.TraCIException as e:
                    # Fallback: if phase index is invalid (e.g., dynamic program mismatch), reset to 0
                    print(f"Warning: Invalid phase {next_phase} for TL {tl_id}. Resetting to phase 0. Error: {e}")
                    traci.trafficlight.setPhase(tl_id, 0)
                    agent.current_phase_time = 0
            # If ACTION_KEEP, do nothing - SUMO continues current phase
            
            # Store for next update
            agent.last_state = current_state
            agent.last_action = action
    
    def reset_agents(self):
        """Reset agent states for new episode."""
        for tl_id in self.tl_ids:
            agent = self.agents[tl_id]
            agent.current_phase_time = 0
            agent.last_state = None
            agent.last_action = None
            agent.last_waiting_time = 0
    
    def decay_all_epsilon(self):
        """Decay epsilon for all agents."""
        for agent in self.agents.values():
            agent.decay_epsilon()
    
    def get_avg_epsilon(self):
        """Get average epsilon across all agents."""
        if not self.agents:
            return 0
        return sum(a.epsilon for a in self.agents.values()) / len(self.agents)


def run_episode(controller, episode, learning=True):
    """Run a single simulation episode."""
    
    if 'SUMO_HOME' not in os.environ:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
    
    sumo_binary = "sumo-gui" if USE_GUI else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-c", SUMO_CONFIG,
        "--no-step-log",
        "--waiting-time-memory", "1000",
        "--random",
    ]
    
    traci.start(sumo_cmd)
    
    # Initialize controller on first episode
    if episode == 0:
        controller.initialize()
    else:
        controller.reset_agents()
    
    # Metrics tracking
    total_waiting_time = 0.0
    total_speed = 0.0
    total_vehicle_steps = 0
    all_vehicles = set()
    vehicle_waiting_times = {}
    vehicle_speeds = {}
    
    step = 0
    
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            step += 1
            
            # Make decisions at intervals
            if step % DECISION_INTERVAL == 0:
                controller.step(learning=learning)
            
            # Collect metrics
            current_vehicles = traci.vehicle.getIDList()
            
            for veh_id in current_vehicles:
                all_vehicles.add(veh_id)
                
                waiting_time = traci.vehicle.getWaitingTime(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                
                total_waiting_time += waiting_time
                total_speed += speed
                total_vehicle_steps += 1
                
                vehicle_waiting_times[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                
                if veh_id not in vehicle_speeds:
                    vehicle_speeds[veh_id] = []
                vehicle_speeds[veh_id].append(speed)
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted")
    except traci.exceptions.FatalTraCIError:
        print("TraCI connection closed")
    
    finally:
        traci.close()
    
    # Calculate metrics
    results = {
        "steps": step,
        "vehicles": len(all_vehicles),
        "avg_waiting_step": total_waiting_time / total_vehicle_steps if total_vehicle_steps > 0 else 0,
        "avg_speed_step": total_speed / total_vehicle_steps if total_vehicle_steps > 0 else 0,
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


def train_and_evaluate():
    """Train Q-learning agents and evaluate performance."""
    
    print("=" * 70)
    print("Q-LEARNING TRAFFIC SIGNAL TIMING CONTROL")
    print("=" * 70)
    print("\n*** SAFE PHASE CYCLING MODE ***")
    print("Agents only decide WHEN to switch, not which phase.")
    print("Phases cycle through predefined safe states sequentially.")
    print(f"\nTraining for {NUM_EPISODES} episodes...")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Discount Factor: {DISCOUNT_FACTOR}")
    print(f"Min Green Time: {MIN_GREEN_TIME} steps")
    print(f"Max Green Time: {MAX_GREEN_TIME} steps")
    print(f"Decision Interval: {DECISION_INTERVAL} steps")
    print("-" * 70)
    
    controller = TrafficLightController()
    episode_results = []
    
    # Training phase
    for episode in range(NUM_EPISODES):
        results = run_episode(controller, episode, learning=True)
        episode_results.append(results)
        
        controller.decay_all_epsilon()
        
        print(f"Episode {episode + 1}/{NUM_EPISODES}: "
              f"Vehicles={results['vehicles']}, "
              f"Avg Wait={results['avg_waiting_vehicle']:.2f}s, "
              f"Avg Speed={results['avg_speed_vehicle']:.2f}m/s, "
              f"ε={controller.get_avg_epsilon():.3f}")
    
    # Final evaluation (no exploration)
    print("\n" + "-" * 70)
    print("Running final evaluation (no exploration, pure exploitation)...")
    
    for agent in controller.agents.values():
        agent.epsilon = 0
    
    final_results = run_episode(controller, NUM_EPISODES, learning=False)
    
    # Print results
    print("\n" + "=" * 70)
    print("FINAL RESULTS (Q-Learning Timing Control)")
    print("=" * 70)
    
    print(f"\nSimulation Duration: {final_results['steps']} steps")
    print(f"Total Vehicles: {final_results['vehicles']}")
    
    print("\n--- Step-Based Metrics ---")
    print(f"Average Waiting Time: {final_results['avg_waiting_step']:.2f} seconds")
    print(f"Average Speed: {final_results['avg_speed_step']:.2f} m/s ({final_results['avg_speed_step'] * 3.6:.2f} km/h)")
    
    print("\n--- Vehicle-Based Metrics ---")
    print(f"Average Accumulated Waiting Time: {final_results['avg_waiting_vehicle']:.2f} seconds")
    print(f"Average Speed per Vehicle: {final_results['avg_speed_vehicle']:.2f} m/s ({final_results['avg_speed_vehicle'] * 3.6:.2f} km/h)")
    
    # Compare with first episode
    if episode_results:
        first = episode_results[0]
        print("\n--- Improvement vs Episode 1 ---")
        wait_change = (first['avg_waiting_vehicle'] - final_results['avg_waiting_vehicle']) / first['avg_waiting_vehicle'] * 100 if first['avg_waiting_vehicle'] > 0 else 0
        speed_change = (final_results['avg_speed_vehicle'] - first['avg_speed_vehicle']) / first['avg_speed_vehicle'] * 100 if first['avg_speed_vehicle'] > 0 else 0
        print(f"Waiting Time Change: {wait_change:+.1f}%")
        print(f"Speed Change: {speed_change:+.1f}%")
    
    print("\n" + "=" * 70)
    
    # Q-table statistics
    print("\nQ-TABLE STATISTICS:")
    total_states = 0
    for tl_id, agent in controller.agents.items():
        num_states = len(agent.q_table)
        total_states += num_states
        print(f"  {tl_id}: {num_states} states, {agent.num_phases} phases")
    print(f"  Total unique states learned: {total_states}")
    
    return controller, final_results


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"Working directory: {os.getcwd()}")
    print(f"Configuration: {SUMO_CONFIG}")
    
    controller, results = train_and_evaluate()
