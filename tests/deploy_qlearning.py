"""
Q-Learning Traffic Signal Control - Deployment Module

Load and deploy pre-trained Q-learning agents for traffic signal optimization.
"""

import os
import sys
import numpy as np
from collections import defaultdict
import traci
import pickle


# Configuration
SUMO_CONFIG = r"..\..\Sioux\data\network\exp.sumocfg"
USE_GUI = True  # Set to False for headless deployment
CHECKPOINT_DIR = "../runs/qlearning_checkpoints"
BEST_AGENTS_FILE = os.path.join(CHECKPOINT_DIR, "best_agents.pkl")

# Q-Learning Constants
NUM_ACTIONS = 2
ACTION_KEEP = 0    # Keep current phase
ACTION_SWITCH = 1  # Switch to next phase
MIN_GREEN_TIME = 15
MAX_GREEN_TIME = 120
DECISION_INTERVAL = 10


class DeployedQLearningAgent:
    """
    Deployed Q-learning agent that uses learned Q-table without exploration.
    Pure exploitation mode - only uses best learned actions.
    """
    
    def __init__(self, tl_id, q_table=None):
        self.tl_id = tl_id
        self.q_table = q_table if q_table is not None else defaultdict(lambda: np.zeros(NUM_ACTIONS))
        self.current_phase_time = 0
        self.num_phases = 0
        
    def get_state(self):
        """Get the current state for this traffic light."""
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        unique_lanes = list(set(controlled_lanes))
        
        # Get queue lengths per lane, binned
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
        
        current_phase = traci.trafficlight.getPhase(self.tl_id)
        
        if self.current_phase_time < 15:
            time_bin = 0
        elif self.current_phase_time < 30:
            time_bin = 1
        elif self.current_phase_time < 45:
            time_bin = 2
        else:
            time_bin = 3
        
        return (tuple(queue_levels), current_phase, time_bin)
    
    def select_action(self, state):
        """
        Select best action based on learned Q-values (pure exploitation).
        
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
        
        # Always use best learned action (exploitation)
        return np.argmax(self.q_table[state])


class DeployedTrafficLightController:
    """Controller managing deployed Q-learning agents."""
    
    def __init__(self):
        self.agents = {}
        self.tl_ids = []
        
    def initialize(self):
        """Initialize agents for all traffic lights."""
        self.tl_ids = traci.trafficlight.getIDList()
        
        for tl_id in self.tl_ids:
            agent = DeployedQLearningAgent(tl_id)
            program = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            agent.num_phases = len(program.phases)
            self.agents[tl_id] = agent
            
        print(f"Initialized {len(self.agents)} deployed Q-learning agents")
        print(f"Traffic lights: {self.tl_ids}")
    
    def step(self):
        """Execute one decision step for all agents."""
        for tl_id in self.tl_ids:
            agent = self.agents[tl_id]
            agent.current_phase_time += DECISION_INTERVAL
            
            # Get current state
            current_state = agent.get_state()
            
            # Select best action
            action = agent.select_action(current_state)
            
            # Execute action
            if action == ACTION_SWITCH:
                current_phase = traci.trafficlight.getPhase(tl_id)
                next_phase = (current_phase + 1) % agent.num_phases
                try:
                    traci.trafficlight.setPhase(tl_id, next_phase)
                    agent.current_phase_time = 0
                except traci.exceptions.TraCIException as e:
                    print(f"Warning: Invalid phase {next_phase} for TL {tl_id}. Error: {e}")
                    traci.trafficlight.setPhase(tl_id, 0)
                    agent.current_phase_time = 0
    
    def reset_agents(self):
        """Reset agent states for new simulation."""
        for tl_id in self.tl_ids:
            agent = self.agents[tl_id]
            agent.current_phase_time = 0


def load_trained_agents(controller, filepath):
    """
    Load trained Q-learning agents from saved checkpoint.
    
    Args:
        controller: DeployedTrafficLightController to load agents into
        filepath: Path to the saved agents pickle file
    
    Returns:
        True if agents were loaded successfully, False otherwise
    """
    if not os.path.exists(filepath):
        print(f"Error: No saved agents found at {filepath}")
        return False
    
    with open(filepath, 'rb') as f:
        agents_data = pickle.load(f)
    
    # Restore agent data
    for tl_id, data in agents_data.items():
        if tl_id in controller.agents:
            agent = controller.agents[tl_id]
            # Restore Q-table as defaultdict
            agent.q_table = defaultdict(lambda: np.zeros(NUM_ACTIONS))
            for state, q_values in data['q_table'].items():
                agent.q_table[state] = np.array(q_values)
    
    print(f"Agents loaded from {filepath}")
    return True


def run_deployment(duration_seconds=600):
    """
    Run deployed Q-learning agents for traffic control.
    
    Args:
        duration_seconds: Duration to run the simulation (optional, runs until vehicles finish if set to None)
    """
    
    if 'SUMO_HOME' not in os.environ:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
    
    sumo_binary = "sumo-gui" if USE_GUI else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-c", SUMO_CONFIG,
        "--no-step-log",
        "--waiting-time-memory", "1000",
    ]
    
    traci.start(sumo_cmd)
    
    controller = DeployedTrafficLightController()
    controller.initialize()
    
    # Load trained agents
    print(f"\nLoading trained agents from {BEST_AGENTS_FILE}...")
    if not load_trained_agents(controller, BEST_AGENTS_FILE):
        print("Failed to load trained agents!")
        traci.close()
        return None
    
    # Metrics tracking
    total_waiting_time = 0.0
    total_speed = 0.0
    total_vehicle_steps = 0
    all_vehicles = set()
    vehicle_waiting_times = {}
    vehicle_speeds = {}
    
    step = 0
    
    try:
        print(f"\nRunning deployment for {'unlimited time' if duration_seconds is None else f'{duration_seconds}s'}...")
        print("-" * 70)
        
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            step += 1
            
            # Stop after duration if specified
            if duration_seconds is not None and step * 0.1 >= duration_seconds:  # SUMO step = 0.1s
                break
            
            # Make decisions at intervals
            if step % DECISION_INTERVAL == 0:
                controller.step()
            
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
        print("\nDeployment interrupted")
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


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("=" * 70)
    print("Q-LEARNING TRAFFIC SIGNAL CONTROL - DEPLOYMENT")
    print("=" * 70)
    print(f"\nWorking directory: {os.getcwd()}")
    print(f"Configuration: {SUMO_CONFIG}")
    print(f"Agent checkpoint: {BEST_AGENTS_FILE}")
    print(f"GUI mode: {USE_GUI}")
    print("-" * 70)
    
    results = run_deployment(duration_seconds=None)
    
    if results:
        print("\n" + "=" * 70)
        print("DEPLOYMENT RESULTS")
        print("=" * 70)
        
        print(f"\nSimulation Duration: {results['steps']} steps ({results['steps'] * 0.1:.1f} seconds)")
        print(f"Total Vehicles: {results['vehicles']}")
        
        print("\n--- Step-Based Metrics ---")
        print(f"Average Waiting Time: {results['avg_waiting_step']:.2f} seconds")
        print(f"Average Speed: {results['avg_speed_step']:.2f} m/s ({results['avg_speed_step'] * 3.6:.2f} km/h)")
        
        print("\n--- Vehicle-Based Metrics ---")
        print(f"Average Accumulated Waiting Time: {results['avg_waiting_vehicle']:.2f} seconds")
        print(f"Average Speed per Vehicle: {results['avg_speed_vehicle']:.2f} m/s ({results['avg_speed_vehicle'] * 3.6:.2f} km/h)")
        print("\n" + "=" * 70)
