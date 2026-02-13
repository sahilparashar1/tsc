"""
Simple analysis script to compare Fixed-Timing and QMIX simulation results.
Reads JSON results saved by `test_simulations.py` and creates comparison charts
(saved under `tests/results/plots/`). Prints summary statistics.

Usage:
    python tests/compare_analysis.py
"""
import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

TEST_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
PLOTS_DIR = os.path.join(TEST_RESULTS_DIR, 'plots')
Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)

fixed_file = os.path.join(TEST_RESULTS_DIR, 'fixed_simulations.json')
qmix_file = os.path.join(TEST_RESULTS_DIR, 'qmix_simulations.json')
ql_file = os.path.join(TEST_RESULTS_DIR, 'qlearning_simulations.json')

if not os.path.exists(fixed_file):
    print('Fixed results file not found:', fixed_file)
    raise SystemExit(1)

with open(fixed_file, 'r') as f:
    fixed_data = json.load(f)
fixed_results = fixed_data['results']

if os.path.exists(qmix_file):
    with open(qmix_file, 'r') as f:
        qmix_data = json.load(f)
    qmix_results = qmix_data['results']
else:
    qmix_results = None

# Extract metrics
fixed_waits = np.array([r['avg_waiting_vehicle'] for r in fixed_results])
fixed_speeds = np.array([r['avg_speed_vehicle'] for r in fixed_results])

print('\nFixed Timing:')
print(f'  sims: {len(fixed_results)}')
print(f'  avg waiting mean: {fixed_waits.mean():.3f}s (std {fixed_waits.std():.3f})')
print(f'  avg speed mean: {fixed_speeds.mean():.3f} m/s (std {fixed_speeds.std():.3f})')

# Plot distributions
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.hist(fixed_waits, bins=20, color='skyblue', edgecolor='black')
plt.title('Fixed: Avg Waiting (per sim)')
plt.xlabel('Avg waiting (s)')

plt.subplot(1,2,2)
plt.hist(fixed_speeds, bins=20, color='lightgreen', edgecolor='black')
plt.title('Fixed: Avg Speed (per sim)')
plt.xlabel('Avg speed (m/s)')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'fixed_distributions.png'), dpi=200)
print('Saved fixed distributions to plots')

if qmix_results is not None:
    qmix_waits = np.array([r['avg_waiting_vehicle'] for r in qmix_results])
    qmix_speeds = np.array([r['avg_speed_vehicle'] for r in qmix_results])

    print('\nQMIX:')
    print(f'  sims: {len(qmix_results)}')
    print(f'  avg waiting mean: {qmix_waits.mean():.3f}s (std {qmix_waits.std():.3f})')
    print(f'  avg speed mean: {qmix_speeds.mean():.3f} m/s (std {qmix_speeds.std():.3f})')

    # Combined boxplots
    plt.figure(figsize=(8,6))
    plt.subplot(2,1,1)
    plt.boxplot([fixed_waits, qmix_waits], labels=['Fixed', 'QMIX'])
    plt.title('Avg Waiting Comparison')

    plt.subplot(2,1,2)
    plt.boxplot([fixed_speeds, qmix_speeds], labels=['Fixed', 'QMIX'])
    plt.title('Avg Speed Comparison')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fixed_vs_qmix_boxplots.png'), dpi=200)
    print('Saved comparison boxplots to plots')

if os.path.exists(ql_file):
    with open(ql_file, 'r') as f:
        ql_data = json.load(f)
    qlearning_results = ql_data.get('results', [])

    if qlearning_results:
        ql_waits = np.array([r['avg_waiting_vehicle'] for r in qlearning_results])
        ql_speeds = np.array([r['avg_speed_vehicle'] for r in qlearning_results])

        print('\nQ-LEARNING:')
        print(f'  sims: {len(qlearning_results)}')
        print(f'  avg waiting mean: {ql_waits.mean():.3f}s (std {ql_waits.std():.3f})')
        print(f'  avg speed mean: {ql_speeds.mean():.3f} m/s (std {ql_speeds.std():.3f})')

        # Combined boxplots including Q-Learning if QMIX present
        labels = ['Fixed']
        waits_to_plot = [fixed_waits]
        speeds_to_plot = [fixed_speeds]

        if qmix_results is not None:
            labels.append('QMIX')
            waits_to_plot.append(qmix_waits)
            speeds_to_plot.append(qmix_speeds)

        labels.append('Q-Learning')
        waits_to_plot.append(ql_waits)
        speeds_to_plot.append(ql_speeds)

        plt.figure(figsize=(8,6))
        plt.subplot(2,1,1)
        plt.boxplot(waits_to_plot, labels=labels)
        plt.title('Avg Waiting Comparison')

        plt.subplot(2,1,2)
        plt.boxplot(speeds_to_plot, labels=labels)
        plt.title('Avg Speed Comparison')

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'fixed_qmix_qlearning_boxplots.png'), dpi=200)
        print('Saved Q-Learning comparison boxplots to plots')

print('\nAnalysis complete. Plots saved in', PLOTS_DIR)
