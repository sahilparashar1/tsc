"""
Run multiple SUMO simulations for Fixed-Timing and QMIX (best) controllers using randomTrips.
Generates routes with SUMO's randomTrips.py, runs 100 simulations per approach, collects
avg waiting time (per vehicle) and avg speed (per vehicle) for each simulation, and
saves results to JSON files under `tests/results/`.

Usage:
    python tests/test_simulations.py

Requirements:
    - SUMO_HOME environment variable set
    - QMIX checkpoint available at ../qmix_checkpoints/qmix_best.pt (relative to repo root)
    - Python environment with `traci`, `numpy` installed
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
import tempfile

# Add project root to path to import run_qmix utilities if available
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

# Optional imports for QMIX evaluation
try:
    from runs.run_qmix import TrafficObservationCollector, QMIXController, DECISION_INTERVAL as QMIX_DECISION_INTERVAL
    import torch
    QMIX_AVAILABLE = True
except Exception:
    QMIX_AVAILABLE = False

# Optional imports for Q-Learning deployment evaluation
try:
    # deploy_qlearning.py lives in the tests/ directory so it can be imported when cwd/tests on sys.path
    from deploy_qlearning import DeployedTrafficLightController, load_trained_agents, DECISION_INTERVAL as QL_DECISION_INTERVAL
    QLEARNING_AVAILABLE = True
except Exception:
    QLEARNING_AVAILABLE = False

import traci

# Configuration
SUMO_HOME = os.environ.get('SUMO_HOME')
if not SUMO_HOME:
    sys.exit("Please set the SUMO_HOME environment variable before running tests")

SUMO_BINARY = 'sumo'  # or 'sumo-gui' for visual
NET_FILE = os.path.join('..', 'Sioux', 'data', 'network', 'exp.net.xml')
TEST_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
Path(TEST_RESULTS_DIR).mkdir(parents=True, exist_ok=True)

NUM_SIMULATIONS = 10
SIM_DURATION = 3600  # seconds
RANDOM_TRIPS_P = 1.5  # average time between generated trips (seconds)

QMIX_CHECKPOINT = os.path.join('..', 'qmix_checkpoints', 'qmix_best.pt')
QLEARNING_CHECKPOINT = os.path.join('..', 'qlearning_checkpoints', 'best_agents.pkl')


def generate_routes(output_path):
    """Generate a random routes file using SUMO's randomTrips.py"""
    rt = os.path.join(SUMO_HOME, 'tools', 'randomTrips.py')
    cmd = [sys.executable, rt,
           '-n', os.path.abspath(NET_FILE),
           '-o', os.path.abspath(output_path),
           '-e', str(SIM_DURATION),
           '-p', str(RANDOM_TRIPS_P),
           '--seed', str(int(time.time() * 1000) % 100000)]
    subprocess.check_call(cmd)


def run_fixed_simulation(routes_file):
    """Run one fixed-timing simulation using the provided routes file and return metrics."""
    sumo_cmd = [SUMO_BINARY,
                '-n', os.path.abspath(NET_FILE),
                '-r', os.path.abspath(routes_file),
                '--no-step-log',
                '--waiting-time-memory', '1000',
                '--step-length', '0.1',
                '--threads', '4']

    traci.start(sumo_cmd)

    total_acc_waiting = {}
    vehicle_speeds = {}
    all_vehicles = set()

    step = 0
    try:
        while traci.simulation.getMinExpectedNumber() > 0 and step * 0.1 < SIM_DURATION:
            traci.simulationStep()
            step += 1

            # collect per-vehicle accumulated waiting and speeds
            for vid in traci.vehicle.getIDList():
                all_vehicles.add(vid)
                total_acc_waiting[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)
                if vid not in vehicle_speeds:
                    vehicle_speeds[vid] = []
                vehicle_speeds[vid].append(traci.vehicle.getSpeed(vid))

    finally:
        traci.close()

    # compute metrics (vehicle-based)
    if all_vehicles:
        avg_wait = float(sum(total_acc_waiting.values()) / len(all_vehicles))
        vehicle_avg_speeds = [sum(speeds) / len(speeds) for speeds in vehicle_speeds.values() if speeds]
        avg_speed = float(sum(vehicle_avg_speeds) / len(vehicle_avg_speeds)) if vehicle_avg_speeds else 0.0
    else:
        avg_wait = 0.0
        avg_speed = 0.0

    return {
        'steps': step,
        'vehicles': len(all_vehicles),
        'avg_waiting_vehicle': avg_wait,
        'avg_speed_vehicle': avg_speed,
    }


def run_qmix_simulation(routes_file):
    """Run one simulation using a trained QMIX controller (best checkpoint)."""
    if not QMIX_AVAILABLE:
        raise RuntimeError('QMIX components not available in environment (runs.run_qmix import failed)')

    sumo_cmd = [SUMO_BINARY,
                '-n', os.path.abspath(NET_FILE),
                '-r', os.path.abspath(routes_file),
                '--no-step-log',
                '--waiting-time-memory', '1000',
                '--step-length', '0.1',
                '--threads', '4']

    traci.start(sumo_cmd)

    # initialize controller for this simulation
    tl_ids = traci.trafficlight.getIDList()
    collector = TrafficObservationCollector(tl_ids)

    sample_obs = collector.get_agent_observation(tl_ids[0])
    sample_state = collector.get_global_state()
    obs_dim = len(sample_obs)
    state_dim = len(sample_state)

    controller = QMIXController(tl_ids, obs_dim, state_dim)
    # Load checkpoint if exists
    if os.path.exists(os.path.join('..', 'qmix_checkpoints', 'qmix_best.pt')):
        try:
            controller.load_checkpoint(os.path.join('..', 'qmix_checkpoints', 'qmix_best.pt'))
        except Exception as e:
            print('Warning: failed to load QMIX checkpoint:', e)
    controller.epsilon = 0.0
    controller.hidden_states = [None] * controller.n_agents
    controller.current_phase_times = {tl_id: 0 for tl_id in tl_ids}

    total_acc_waiting = {}
    vehicle_speeds = {}
    all_vehicles = set()

    step = 0
    try:
        while traci.simulation.getMinExpectedNumber() > 0 and step * 0.1 < SIM_DURATION:
            traci.simulationStep()
            step += 1

            if step % 10 == 0:  # decision interval (controller uses DECISION_INTERVAL)
                action_dict, _ = controller.select_actions()
                controller.execute_actions(action_dict)

            for vid in traci.vehicle.getIDList():
                all_vehicles.add(vid)
                total_acc_waiting[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)
                if vid not in vehicle_speeds:
                    vehicle_speeds[vid] = []
                vehicle_speeds[vid].append(traci.vehicle.getSpeed(vid))

    finally:
        traci.close()

    if all_vehicles:
        avg_wait = float(sum(total_acc_waiting.values()) / len(all_vehicles))
        vehicle_avg_speeds = [sum(speeds) / len(speeds) for speeds in vehicle_speeds.values() if speeds]
        avg_speed = float(sum(vehicle_avg_speeds) / len(vehicle_avg_speeds)) if vehicle_avg_speeds else 0.0
    else:
        avg_wait = 0.0
        avg_speed = 0.0

    return {
        'steps': step,
        'vehicles': len(all_vehicles),
        'avg_waiting_vehicle': avg_wait,
        'avg_speed_vehicle': avg_speed,
    }


def run_qlearning_simulation(routes_file):
    """Run one simulation using deployed Q-learning agents (best checkpoint)."""
    if not QLEARNING_AVAILABLE:
        raise RuntimeError('Q-Learning deployment utilities not available (deploy_qlearning import failed)')

    sumo_cmd = [SUMO_BINARY,
                '-n', os.path.abspath(NET_FILE),
                '-r', os.path.abspath(routes_file),
                '--no-step-log',
                '--waiting-time-memory', '1000',
                '--step-length', '0.1',
                '--threads', '4']

    traci.start(sumo_cmd)

    # initialize controller for this simulation
    controller = DeployedTrafficLightController()
    controller.initialize()

    # Load checkpoint if exists
    if os.path.exists(QLEARNING_CHECKPOINT):
        try:
            load_trained_agents(controller, QLEARNING_CHECKPOINT)
        except Exception as e:
            print('Warning: failed to load Q-Learning checkpoint:', e)
    else:
        print(f'Warning: Q-Learning checkpoint not found at {QLEARNING_CHECKPOINT}; running with default agents')

    total_acc_waiting = {}
    vehicle_speeds = {}
    all_vehicles = set()

    step = 0
    try:
        while traci.simulation.getMinExpectedNumber() > 0 and step * 0.1 < SIM_DURATION:
            traci.simulationStep()
            step += 1

            # decision interval for Q-Learning
            if step % (QL_DECISION_INTERVAL if 'QL_DECISION_INTERVAL' in globals() else 10) == 0:
                controller.step()

            for vid in traci.vehicle.getIDList():
                all_vehicles.add(vid)
                total_acc_waiting[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)
                if vid not in vehicle_speeds:
                    vehicle_speeds[vid] = []
                vehicle_speeds[vid].append(traci.vehicle.getSpeed(vid))

    finally:
        traci.close()

    if all_vehicles:
        avg_wait = float(sum(total_acc_waiting.values()) / len(all_vehicles))
        vehicle_avg_speeds = [sum(speeds) / len(speeds) for speeds in vehicle_speeds.values() if speeds]
        avg_speed = float(sum(vehicle_avg_speeds) / len(vehicle_avg_speeds)) if vehicle_avg_speeds else 0.0
    else:
        avg_wait = 0.0
        avg_speed = 0.0

    return {
        'steps': step,
        'vehicles': len(all_vehicles),
        'avg_waiting_vehicle': avg_wait,
        'avg_speed_vehicle': avg_speed,
    }


def run_all():
    fixed_results = []
    qmix_results = []
    qlearning_results = []

    for i in range(1, NUM_SIMULATIONS + 1):
        print(f"=== Simulation {i}/{NUM_SIMULATIONS} (Fixed-Timing) ===")
        routes_file = os.path.join(TEST_RESULTS_DIR, f"routes_fixed_{i}.rou.xml")
        generate_routes(routes_file)
        try:
            res = run_fixed_simulation(routes_file)
            fixed_results.append(res)
            print(f"Fixed: avg_wait={res['avg_waiting_vehicle']:.3f}, avg_speed={res['avg_speed_vehicle']:.3f}")
        finally:
            try:
                if os.path.exists(routes_file):
                    os.remove(routes_file)
            except Exception as e:
                print(f"Warning: failed to remove route file {routes_file}: {e}")

        if QMIX_AVAILABLE:
            print(f"=== Simulation {i}/{NUM_SIMULATIONS} (QMIX) ===")
            qroutes_file = os.path.join(TEST_RESULTS_DIR, f"routes_qmix_{i}.rou.xml")
            generate_routes(qroutes_file)
            try:
                qres = run_qmix_simulation(qroutes_file)
                qmix_results.append(qres)
                print(f"QMIX: avg_wait={qres['avg_waiting_vehicle']:.3f}, avg_speed={qres['avg_speed_vehicle']:.3f}")
            finally:
                try:
                    if os.path.exists(qroutes_file):
                        os.remove(qroutes_file)
                except Exception as e:
                    print(f"Warning: failed to remove route file {qroutes_file}: {e}")
        else:
            print('QMIX not available; skipping QMIX simulations')

        if QLEARNING_AVAILABLE:
            print(f"=== Simulation {i}/{NUM_SIMULATIONS} (Q-Learning) ===")
            ql_routes_file = os.path.join(TEST_RESULTS_DIR, f"routes_qlearning_{i}.rou.xml")
            generate_routes(ql_routes_file)
            try:
                qlres = run_qlearning_simulation(ql_routes_file)
                qlearning_results.append(qlres)
                print(f"Q-Learning: avg_wait={qlres['avg_waiting_vehicle']:.3f}, avg_speed={qlres['avg_speed_vehicle']:.3f}")
            finally:
                try:
                    if os.path.exists(ql_routes_file):
                        os.remove(ql_routes_file)
                except Exception as e:
                    print(f"Warning: failed to remove route file {ql_routes_file}: {e}")
        else:
            print('Q-Learning deployment not available; skipping Q-Learning simulations')

    # Save JSON results
    fixed_out = os.path.join(TEST_RESULTS_DIR, 'fixed_simulations.json')
    with open(fixed_out, 'w') as f:
        json.dump({'config': {'num_simulations': NUM_SIMULATIONS, 'sim_duration': SIM_DURATION}, 'results': fixed_results}, f, indent=2)

    print(f"Saved fixed simulation results to {fixed_out}")

    if QMIX_AVAILABLE:
        qmix_out = os.path.join(TEST_RESULTS_DIR, 'qmix_simulations.json')
        with open(qmix_out, 'w') as f:
            json.dump({'config': {'num_simulations': NUM_SIMULATIONS, 'sim_duration': SIM_DURATION}, 'results': qmix_results}, f, indent=2)
        print(f"Saved QMIX simulation results to {qmix_out}")

    if QLEARNING_AVAILABLE:
        ql_out = os.path.join(TEST_RESULTS_DIR, 'qlearning_simulations.json')
        with open(ql_out, 'w') as f:
            json.dump({'config': {'num_simulations': NUM_SIMULATIONS, 'sim_duration': SIM_DURATION}, 'results': qlearning_results}, f, indent=2)
        print(f"Saved Q-Learning simulation results to {ql_out}")

    # Print aggregate averages
    def aggregate(results):
        arr_wait = [r['avg_waiting_vehicle'] for r in results]
        arr_speed = [r['avg_speed_vehicle'] for r in results]
        return float(np.mean(arr_wait)), float(np.mean(arr_speed))

    fw, fs = aggregate(fixed_results)
    print('\n=== Fixed Timing Aggregate ===')
    print(f'Average waiting (mean over sims): {fw:.3f}s')
    print(f'Average speed (mean over sims): {fs:.3f} m/s')

    if QMIX_AVAILABLE:
        qw, qs = aggregate(qmix_results)
        print('\n=== QMIX Aggregate ===')
        print(f'Average waiting (mean over sims): {qw:.3f}s')
        print(f'Average speed (mean over sims): {qs:.3f} m/s')

    if QLEARNING_AVAILABLE:
        qlw, qls = aggregate(qlearning_results)
        print('\n=== Q-LEARNING Aggregate ===')
        print(f'Average waiting (mean over sims): {qlw:.3f}s')
        print(f'Average speed (mean over sims): {qls:.3f} m/s')


if __name__ == '__main__':
    run_all()
