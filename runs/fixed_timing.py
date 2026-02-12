"""
SUMO Fixed Timing Simulation Script
Runs the simulation and calculates average waiting time and speed across all vehicles.
"""

import os
import sys
import traci

# Configuration
SUMO_CONFIG = r"..\Sioux\data\network\exp.sumocfg"
USE_GUI = False  # Set to True to use sumo-gui instead of sumo

def run_simulation():
    """Run the SUMO simulation and collect metrics."""
    
    # Check if SUMO_HOME is set
    if 'SUMO_HOME' not in os.environ:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
    
    # Build the SUMO command
    sumo_binary = "sumo-gui" if USE_GUI else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-c", SUMO_CONFIG,
        "--no-step-log",  # Suppress step logs for cleaner output
        "--waiting-time-memory", "1000",  # Track waiting time
    ]
    
    # Start SUMO simulation
    print("Starting SUMO simulation...")
    traci.start(sumo_cmd)
    
    # Metrics tracking
    total_waiting_time = 0.0
    total_speed = 0.0
    total_vehicle_steps = 0  # Total vehicle-steps (for averaging)
    
    # Track all vehicles that appeared in the simulation
    all_vehicles = set()
    vehicle_waiting_times = {}  # Final waiting time per vehicle
    vehicle_speeds = {}  # List of speeds per vehicle for averaging
    
    step = 0
    
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            step += 1
            
            # Get all vehicles currently in the simulation
            current_vehicles = traci.vehicle.getIDList()
            
            for veh_id in current_vehicles:
                all_vehicles.add(veh_id)
                
                # Get instantaneous waiting time (seconds waiting at current step)
                waiting_time = traci.vehicle.getWaitingTime(veh_id)
                
                # Get current speed (m/s)
                speed = traci.vehicle.getSpeed(veh_id)
                
                # Accumulate for step-based averaging
                total_waiting_time += waiting_time
                total_speed += speed
                total_vehicle_steps += 1
                
                # Track accumulated waiting time per vehicle
                vehicle_waiting_times[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                
                # Track speeds for per-vehicle averaging
                if veh_id not in vehicle_speeds:
                    vehicle_speeds[veh_id] = []
                vehicle_speeds[veh_id].append(speed)
            
            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"Step {step}: {len(current_vehicles)} vehicles in simulation")
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    finally:
        traci.close()
    
    # Calculate final metrics
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    
    print(f"\nSimulation Duration: {step} steps")
    print(f"Total Unique Vehicles: {len(all_vehicles)}")
    
    # Method 1: Step-based averaging (average across all vehicle-steps)
    if total_vehicle_steps > 0:
        avg_waiting_time_step = total_waiting_time / total_vehicle_steps
        avg_speed_step = total_speed / total_vehicle_steps
        avg_speed_kmh_step = avg_speed_step * 3.6  # Convert m/s to km/h
        
        print("\n--- Step-Based Metrics (averaging across all vehicle-steps) ---")
        print(f"Average Waiting Time: {avg_waiting_time_step:.2f} seconds")
        print(f"Average Speed: {avg_speed_step:.2f} m/s ({avg_speed_kmh_step:.2f} km/h)")
    
    # Method 2: Vehicle-based averaging (average per vehicle, then average across vehicles)
    if len(all_vehicles) > 0:
        # Average accumulated waiting time per vehicle
        total_accumulated_waiting = sum(vehicle_waiting_times.values())
        avg_waiting_time_vehicle = total_accumulated_waiting / len(all_vehicles)
        
        # Average speed per vehicle (mean of each vehicle's mean speed)
        vehicle_avg_speeds = [sum(speeds) / len(speeds) for speeds in vehicle_speeds.values() if speeds]
        avg_speed_vehicle = sum(vehicle_avg_speeds) / len(vehicle_avg_speeds) if vehicle_avg_speeds else 0
        avg_speed_kmh_vehicle = avg_speed_vehicle * 3.6
        
        print("\n--- Vehicle-Based Metrics (averaging across all vehicles) ---")
        print(f"Average Accumulated Waiting Time per Vehicle: {avg_waiting_time_vehicle:.2f} seconds")
        print(f"Average Speed per Vehicle: {avg_speed_vehicle:.2f} m/s ({avg_speed_kmh_vehicle:.2f} km/h)")
    
    print("\n" + "=" * 60)
    
    # Return metrics for potential further use
    return {
        "simulation_steps": step,
        "total_vehicles": len(all_vehicles),
        "avg_waiting_time_per_step": avg_waiting_time_step if total_vehicle_steps > 0 else 0,
        "avg_speed_per_step_ms": avg_speed_step if total_vehicle_steps > 0 else 0,
        "avg_waiting_time_per_vehicle": avg_waiting_time_vehicle if len(all_vehicles) > 0 else 0,
        "avg_speed_per_vehicle_ms": avg_speed_vehicle if len(all_vehicles) > 0 else 0,
    }


if __name__ == "__main__":
    # Change to script directory to ensure relative paths work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"Working directory: {os.getcwd()}")
    print(f"Running simulation: {SUMO_CONFIG}")
    print("-" * 60)
    
    results = run_simulation()
