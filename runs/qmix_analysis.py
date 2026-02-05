"""
QMIX Results Visualization and Analysis Script

Generates comprehensive visualizations and analysis reports from QMIX training results.

Usage:
    python qmix_analysis.py analyze <results_json>
    python qmix_analysis.py compare <qlearning_json> <qmix_json>
    python qmix_analysis.py simulate <results_json> <output_dir>
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime


class QMIXResultsAnalyzer:
    """Comprehensive analysis of QMIX training results."""
    
    def __init__(self, results_file: str):
        """Load results from JSON file."""
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        self.config = self.results['configuration']
        self.final_results = self.results['final_results']
        self.episode_results = self.results['episode_results']
        
        self._compute_metrics()
    
    def _compute_metrics(self):
        """Compute derived metrics."""
        episodes = self.episode_results
        
        # Extract time series
        self.waiting_times = np.array([ep['avg_waiting_vehicle'] for ep in episodes])
        self.speeds = np.array([ep['avg_speed_vehicle'] for ep in episodes])
        self.rewards = np.array([ep.get('episode_reward', 0) for ep in episodes])
        self.vehicles = np.array([ep['vehicles'] for ep in episodes])
        
        # Compute statistics
        self.episode_numbers = np.arange(1, len(episodes) + 1)
        
        # Improvement metrics
        self.waiting_improvement = (self.waiting_times[0] - self.waiting_times[-1]) / self.waiting_times[0] * 100
        self.speed_improvement = (self.speeds[-1] - self.speeds[0]) / self.speeds[0] * 100
        
        # Convergence metrics
        last_n = 5
        self.final_waiting_avg = np.mean(self.waiting_times[-last_n:])
        self.final_speed_avg = np.mean(self.speeds[-last_n:])
        self.final_waiting_std = np.std(self.waiting_times[-last_n:])
        self.final_speed_std = np.std(self.speeds[-last_n:])
    
    def print_summary(self):
        """Print text summary of results."""
        print("\n" + "=" * 90)
        print("QMIX TRAINING RESULTS SUMMARY")
        print("=" * 90)
        
        print("\nConfiguration:")
        print(f"  Agents (Traffic Lights): {self.config['num_agents']}")
        print(f"  Observation Dimension: {self.config['obs_dim']}")
        print(f"  State Dimension: {self.config['state_dim']}")
        print(f"  Learning Rate: {self.config['learning_rate']}")
        print(f"  Discount Factor: {self.config['discount_factor']}")
        print(f"  Episodes: {self.config['num_episodes']}")
        
        print("\nPERFORMANCE METRICS:")
        print("-" * 90)
        
        print("\nWaiting Time (per Vehicle):")
        print(f"  Episode 1:           {self.waiting_times[0]:7.2f}s")
        print(f"  Episode {len(self.waiting_times)}:           {self.waiting_times[-1]:7.2f}s")
        print(f"  Best Episode:        {np.min(self.waiting_times):7.2f}s (episode {np.argmin(self.waiting_times) + 1})")
        print(f"  Worst Episode:       {np.max(self.waiting_times):7.2f}s (episode {np.argmax(self.waiting_times) + 1})")
        print(f"  Average (last 5):    {self.final_waiting_avg:7.2f}s ± {self.final_waiting_std:.2f}s")
        print(f"  Improvement:         {self.waiting_improvement:+7.1f}%")
        
        print("\nAverage Speed (per Vehicle):")
        print(f"  Episode 1:           {self.speeds[0]:7.2f}m/s")
        print(f"  Episode {len(self.speeds)}:           {self.speeds[-1]:7.2f}m/s")
        print(f"  Best Episode:        {np.max(self.speeds):7.2f}m/s (episode {np.argmax(self.speeds) + 1})")
        print(f"  Worst Episode:       {np.min(self.speeds):7.2f}m/s (episode {np.argmin(self.speeds) + 1})")
        print(f"  Average (last 5):    {self.final_speed_avg:7.2f}m/s ± {self.final_speed_std:.2f}m/s")
        print(f"  Improvement:         {self.speed_improvement:+7.1f}%")
        
        print("\nEpisode Reward:")
        print(f"  Best Reward:         {np.max(self.rewards):+7.3f}")
        print(f"  Worst Reward:        {np.min(self.rewards):+7.3f}")
        print(f"  Average Reward:      {np.mean(self.rewards):+7.3f}")
        print(f"  Final Reward:        {self.rewards[-1]:+7.3f}")
        
        print("\nFinal Simulation Results:")
        print(f"  Total Vehicles:      {self.final_results['vehicles']:7d}")
        print(f"  Simulation Steps:    {self.final_results['steps']:7d}")
        print(f"  Avg Waiting (step):  {self.final_results['avg_waiting_step']:7.2f}s")
        print(f"  Avg Speed (step):    {self.final_results['avg_speed_step']:7.2f}m/s")
        
        print("\n" + "=" * 90 + "\n")
    
    def generate_full_report(self, output_dir: str = "./qmix_analysis"):
        """Generate complete analysis report with visualizations."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Generating analysis report in {output_dir}...")
        
        # 1. Full dashboard
        self._plot_dashboard(str(output_path / "01_dashboard.png"))
        
        # 2. Single metrics
        self._plot_waiting_time(str(output_path / "02_waiting_time.png"))
        self._plot_speed(str(output_path / "03_speed.png"))
        self._plot_reward(str(output_path / "04_reward.png"))
        
        # 3. Analysis
        self._plot_convergence(str(output_path / "05_convergence.png"))
        self._plot_correlation(str(output_path / "06_correlation.png"))
        
        # 4. Statistics
        self._generate_statistics_table(str(output_path / "07_statistics.csv"))
        
        # 5. Summary
        self._generate_text_report(str(output_path / "08_summary.txt"))
        
        print("✓ Analysis report generated successfully!")
        print(f"  Output directory: {output_path.absolute()}")
    
    def _plot_dashboard(self, output_file: str):
        """Plot comprehensive dashboard."""
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        fig.suptitle('QMIX Training Dashboard', fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Waiting time with trend
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.episode_numbers, self.waiting_times, 'b-', linewidth=2.5, label='Waiting Time')
        z = np.polyfit(self.episode_numbers, self.waiting_times, 2)
        p = np.poly1d(z)
        ax1.plot(self.episode_numbers, p(self.episode_numbers), 'r--', linewidth=2, alpha=0.7, label='Trend')
        ax1.set_xlabel('Episode', fontsize=10)
        ax1.set_ylabel('Avg Waiting Time (s)', fontsize=10)
        ax1.set_title('Waiting Time per Episode', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
        
        # 2. Speed with trend
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.episode_numbers, self.speeds, 'g-', linewidth=2.5, label='Speed')
        z = np.polyfit(self.episode_numbers, self.speeds, 2)
        p = np.poly1d(z)
        ax2.plot(self.episode_numbers, p(self.episode_numbers), 'r--', linewidth=2, alpha=0.7, label='Trend')
        ax2.set_xlabel('Episode', fontsize=10)
        ax2.set_ylabel('Avg Speed (m/s)', fontsize=10)
        ax2.set_title('Speed per Episode', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        
        # 3. Reward
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.episode_numbers, self.rewards, 'r-', linewidth=2.5, marker='o', markersize=4, label='Reward')
        ax3.set_xlabel('Episode', fontsize=10)
        ax3.set_ylabel('Episode Reward', fontsize=10)
        ax3.set_title('QMIX Episode Reward', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)
        
        # 4. Improvement histogram
        ax4 = fig.add_subplot(gs[1, 0])
        improvements = [(self.waiting_times[0] - wt) for wt in self.waiting_times]
        colors = ['green' if i > 0 else 'red' for i in improvements]
        ax4.bar(self.episode_numbers, improvements, color=colors, alpha=0.7, edgecolor='black')
        ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
        ax4.set_xlabel('Episode', fontsize=10)
        ax4.set_ylabel('Waiting Time Improvement (s)', fontsize=10)
        ax4.set_title('Improvement vs Episode 1', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Distribution box plot
        ax5 = fig.add_subplot(gs[1, 1])
        bins = [
            self.waiting_times[0:max(1, len(self.waiting_times)//3)],
            self.waiting_times[max(1, len(self.waiting_times)//3):2*max(1, len(self.waiting_times)//3)],
            self.waiting_times[2*max(1, len(self.waiting_times)//3):]
        ]
        ax5.boxplot(bins, labels=['Early', 'Mid', 'Late'])
        ax5.set_ylabel('Waiting Time (s)', fontsize=10)
        ax5.set_title('Waiting Time Distribution', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Vehicles per episode
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.bar(self.episode_numbers, self.vehicles, color='purple', alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Episode', fontsize=10)
        ax6.set_ylabel('Number of Vehicles', fontsize=10)
        ax6.set_title('Vehicles per Episode', fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 7. Statistics text
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        stats_text = f"""
        SUMMARY STATISTICS
        
        Waiting Time:
            Initial: {self.waiting_times[0]:.2f}s  →  Final: {self.waiting_times[-1]:.2f}s  |  Improvement: {self.waiting_improvement:+.1f}%
            Best: {np.min(self.waiting_times):.2f}s (ep{np.argmin(self.waiting_times)+1})  |  Worst: {np.max(self.waiting_times):.2f}s (ep{np.argmax(self.waiting_times)+1})
            Last 5 avg: {self.final_waiting_avg:.2f}s ± {self.final_waiting_std:.2f}s
        
        Speed:
            Initial: {self.speeds[0]:.2f}m/s  →  Final: {self.speeds[-1]:.2f}m/s  |  Improvement: {self.speed_improvement:+.1f}%
            Best: {np.max(self.speeds):.2f}m/s (ep{np.argmax(self.speeds)+1})  |  Worst: {np.min(self.speeds):.2f}m/s (ep{np.argmin(self.speeds)+1})
            Last 5 avg: {self.final_speed_avg:.2f}m/s ± {self.final_speed_std:.2f}m/s
        
        Network Config:  {self.config['num_agents']} agents  |  Obs dim: {self.config['obs_dim']}  |  State dim: {self.config['state_dim']}
        Training:  {self.config['num_episodes']} episodes  |  Learning rate: {self.config['learning_rate']}  |  Batch size: {self.config['batch_size']}
        """
        
        ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()
    
    def _plot_waiting_time(self, output_file: str):
        """Plot detailed waiting time analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Waiting Time Analysis', fontsize=16, fontweight='bold')
        
        # Time series
        axes[0, 0].plot(self.episode_numbers, self.waiting_times, 'b-', linewidth=2.5, marker='o', markersize=5)
        axes[0, 0].fill_between(self.episode_numbers, self.waiting_times, alpha=0.3)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Waiting Time (s)')
        axes[0, 0].set_title('Waiting Time per Episode')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distribution
        axes[0, 1].hist(self.waiting_times, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(np.mean(self.waiting_times), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.waiting_times):.2f}s')
        axes[0, 1].set_xlabel('Waiting Time (s)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Waiting Times')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # CDF
        sorted_wt = np.sort(self.waiting_times)
        cdf = np.arange(1, len(sorted_wt) + 1) / len(sorted_wt)
        axes[1, 0].plot(sorted_wt, cdf, linewidth=2.5)
        axes[1, 0].set_xlabel('Waiting Time (s)')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].set_title('CDF of Waiting Times')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Moving average
        window = max(3, len(self.waiting_times) // 10)
        moving_avg = pd.Series(self.waiting_times).rolling(window=window).mean()
        axes[1, 1].plot(self.episode_numbers, self.waiting_times, 'b-', alpha=0.3, label='Raw')
        axes[1, 1].plot(self.episode_numbers, moving_avg, 'r-', linewidth=2.5, label=f'Moving Avg ({window})')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Waiting Time (s)')
        axes[1, 1].set_title('Moving Average')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()
    
    def _plot_speed(self, output_file: str):
        """Plot detailed speed analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Speed Analysis', fontsize=16, fontweight='bold')
        
        # Time series
        axes[0, 0].plot(self.episode_numbers, self.speeds, 'g-', linewidth=2.5, marker='s', markersize=5)
        axes[0, 0].fill_between(self.episode_numbers, self.speeds, alpha=0.3, color='green')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Speed (m/s)')
        axes[0, 0].set_title('Average Speed per Episode')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distribution
        axes[0, 1].hist(self.speeds, bins=10, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(np.mean(self.speeds), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.speeds):.2f}m/s')
        axes[0, 1].set_xlabel('Speed (m/s)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Speeds')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # CDF
        sorted_sp = np.sort(self.speeds)
        cdf = np.arange(1, len(sorted_sp) + 1) / len(sorted_sp)
        axes[1, 0].plot(sorted_sp, cdf, linewidth=2.5, color='green')
        axes[1, 0].set_xlabel('Speed (m/s)')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].set_title('CDF of Speeds')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Moving average
        window = max(3, len(self.speeds) // 10)
        moving_avg = pd.Series(self.speeds).rolling(window=window).mean()
        axes[1, 1].plot(self.episode_numbers, self.speeds, 'g-', alpha=0.3, label='Raw')
        axes[1, 1].plot(self.episode_numbers, moving_avg, 'r-', linewidth=2.5, label=f'Moving Avg ({window})')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Speed (m/s)')
        axes[1, 1].set_title('Moving Average')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()
    
    def _plot_reward(self, output_file: str):
        """Plot reward analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Reward Analysis', fontsize=16, fontweight='bold')
        
        # Time series
        colors = ['green' if r >= 0 else 'red' for r in self.rewards]
        axes[0, 0].bar(self.episode_numbers, self.rewards, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.8)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Distribution
        axes[0, 1].hist(self.rewards, bins=10, color='salmon', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(np.mean(self.rewards), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.rewards):.3f}')
        axes[0, 1].set_xlabel('Reward')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Rewards')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Cumulative
        cumsum = np.cumsum(self.rewards)
        axes[1, 0].plot(self.episode_numbers, cumsum, 'b-', linewidth=2.5, marker='o', markersize=4)
        axes[1, 0].fill_between(self.episode_numbers, cumsum, alpha=0.3)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Cumulative Reward')
        axes[1, 0].set_title('Cumulative Episode Rewards')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Correlation with waiting time
        axes[1, 1].scatter(self.rewards, self.waiting_times, s=80, alpha=0.6, edgecolor='black')
        z = np.polyfit(self.rewards, self.waiting_times, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(self.rewards, p(self.rewards), 'r--', linewidth=2, label='Trend')
        correlation = np.corrcoef(self.rewards, self.waiting_times)[0, 1]
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Waiting Time (s)')
        axes[1, 1].set_title(f'Reward vs Waiting Time (r={correlation:.3f})')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()
    
    def _plot_convergence(self, output_file: str):
        """Plot convergence analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Convergence Analysis', fontsize=16, fontweight='bold')
        
        # Waiting time variance
        window = max(3, len(self.episode_numbers) // 10)
        wt_var = pd.Series(self.waiting_times).rolling(window=window).std()
        axes[0, 0].plot(self.episode_numbers, wt_var, 'b-', linewidth=2.5, label='Rolling Std Dev')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Variance')
        axes[0, 0].set_title(f'Waiting Time Variance (window={window})')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Speed variance
        sp_var = pd.Series(self.speeds).rolling(window=window).std()
        axes[0, 1].plot(self.episode_numbers, sp_var, 'g-', linewidth=2.5, label='Rolling Std Dev')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Variance')
        axes[0, 1].set_title(f'Speed Variance (window={window})')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Relative improvement
        rel_improvement_wt = [(self.waiting_times[0] - wt) / self.waiting_times[0] * 100 for wt in self.waiting_times]
        rel_improvement_sp = [(sp - self.speeds[0]) / self.speeds[0] * 100 for sp in self.speeds]
        
        axes[1, 0].plot(self.episode_numbers, rel_improvement_wt, 'b-', linewidth=2.5, marker='o', markersize=4, label='Waiting Time')
        axes[1, 0].plot(self.episode_numbers, rel_improvement_sp, 'g-', linewidth=2.5, marker='s', markersize=4, label='Speed')
        axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.8)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Relative Improvement (%)')
        axes[1, 0].set_title('Relative Improvement vs Episode 1')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Normalized metrics (0-1 scale)
        wt_norm = 1 - (self.waiting_times - np.min(self.waiting_times)) / (np.max(self.waiting_times) - np.min(self.waiting_times))
        sp_norm = (self.speeds - np.min(self.speeds)) / (np.max(self.speeds) - np.min(self.speeds))
        
        axes[1, 1].plot(self.episode_numbers, wt_norm, 'b-', linewidth=2.5, marker='o', markersize=4, label='Waiting Score')
        axes[1, 1].plot(self.episode_numbers, sp_norm, 'g-', linewidth=2.5, marker='s', markersize=4, label='Speed Score')
        axes[1, 1].fill_between(self.episode_numbers, 0, wt_norm, alpha=0.2, color='blue')
        axes[1, 1].fill_between(self.episode_numbers, 0, sp_norm, alpha=0.2, color='green')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Normalized Score (0-1)')
        axes[1, 1].set_title('Normalized Performance Scores')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()
    
    def _plot_correlation(self, output_file: str):
        """Plot correlation analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')
        
        # Waiting vs Speed
        axes[0, 0].scatter(self.waiting_times, self.speeds, s=100, alpha=0.6, edgecolor='black')
        z = np.polyfit(self.waiting_times, self.speeds, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(self.waiting_times, p(self.waiting_times), 'r--', linewidth=2)
        corr_ws = np.corrcoef(self.waiting_times, self.speeds)[0, 1]
        axes[0, 0].set_xlabel('Waiting Time (s)')
        axes[0, 0].set_ylabel('Speed (m/s)')
        axes[0, 0].set_title(f'Waiting Time vs Speed (r={corr_ws:.3f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reward vs Waiting
        axes[0, 1].scatter(self.rewards, self.waiting_times, s=100, alpha=0.6, edgecolor='black', color='orange')
        z = np.polyfit(self.rewards, self.waiting_times, 1)
        p = np.poly1d(z)
        axes[0, 1].plot(self.rewards, p(self.rewards), 'r--', linewidth=2)
        corr_rw = np.corrcoef(self.rewards, self.waiting_times)[0, 1]
        axes[0, 1].set_xlabel('Reward')
        axes[0, 1].set_ylabel('Waiting Time (s)')
        axes[0, 1].set_title(f'Reward vs Waiting Time (r={corr_rw:.3f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Reward vs Speed
        axes[1, 0].scatter(self.rewards, self.speeds, s=100, alpha=0.6, edgecolor='black', color='purple')
        z = np.polyfit(self.rewards, self.speeds, 1)
        p = np.poly1d(z)
        axes[1, 0].plot(self.rewards, p(self.rewards), 'r--', linewidth=2)
        corr_rs = np.corrcoef(self.rewards, self.speeds)[0, 1]
        axes[1, 0].set_xlabel('Reward')
        axes[1, 0].set_ylabel('Speed (m/s)')
        axes[1, 0].set_title(f'Reward vs Speed (r={corr_rs:.3f})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Correlation heatmap
        corr_matrix = np.array([
            [1.0, corr_ws, corr_rw],
            [corr_ws, 1.0, corr_rs],
            [corr_rw, corr_rs, 1.0]
        ])
        
        im = axes[1, 1].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        axes[1, 1].set_xticks([0, 1, 2])
        axes[1, 1].set_yticks([0, 1, 2])
        axes[1, 1].set_xticklabels(['Wait', 'Speed', 'Reward'])
        axes[1, 1].set_yticklabels(['Wait', 'Speed', 'Reward'])
        axes[1, 1].set_title('Correlation Matrix')
        
        # Add correlation values
        for i in range(3):
            for j in range(3):
                text = axes[1, 1].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                      ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=axes[1, 1], label='Correlation')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()
    
    def _generate_statistics_table(self, output_file: str):
        """Generate detailed statistics CSV."""
        data = {
            'Episode': self.episode_numbers,
            'Waiting_Time_s': self.waiting_times,
            'Speed_m_s': self.speeds,
            'Episode_Reward': self.rewards,
            'Vehicles': self.vehicles.astype(int),
        }
        
        df = pd.DataFrame(data)
        
        # Add rolling statistics
        df['Waiting_MA5'] = df['Waiting_Time_s'].rolling(window=5, center=True).mean()
        df['Speed_MA5'] = df['Speed_m_s'].rolling(window=5, center=True).mean()
        
        df.to_csv(output_file, index=False)
        print(f"  ✓ Saved: {output_file}")
    
    def _generate_text_report(self, output_file: str):
        """Generate comprehensive text report."""
        report = f"""
QMIX TRAINING ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
CONFIGURATION
{'='*80}

Network Architecture:
  Agents (Traffic Lights): {self.config['num_agents']}
  Observation Dimension: {self.config['obs_dim']}
  State Dimension: {self.config['state_dim']}

Training Parameters:
  Learning Rate: {self.config['learning_rate']}
  Discount Factor: {self.config['discount_factor']}
  Batch Size: {self.config['batch_size']}
  Episodes: {self.config['num_episodes']}

{'='*80}
PERFORMANCE SUMMARY
{'='*80}

PRIMARY METRIC - Waiting Time (Lower is Better):
  Episode 1:              {self.waiting_times[0]:.2f} seconds
  Episode {len(self.waiting_times)}:              {self.waiting_times[-1]:.2f} seconds
  Best Episode:           {np.min(self.waiting_times):.2f} seconds (Episode {np.argmin(self.waiting_times)+1})
  Worst Episode:          {np.max(self.waiting_times):.2f} seconds (Episode {np.argmax(self.waiting_times)+1})
  Average:                {np.mean(self.waiting_times):.2f} ± {np.std(self.waiting_times):.2f} seconds
  Last 5 Episodes:        {self.final_waiting_avg:.2f} ± {self.final_waiting_std:.2f} seconds
  
  Overall Improvement:    {self.waiting_improvement:+.1f}%
  Per-Episode (Avg):      {(self.waiting_times[0] - np.mean(self.waiting_times[1:])) / len(self.waiting_times[1:]):.2f} seconds

SECONDARY METRIC - Speed (Higher is Better):
  Episode 1:              {self.speeds[0]:.2f} m/s ({self.speeds[0]*3.6:.2f} km/h)
  Episode {len(self.speeds)}:              {self.speeds[-1]:.2f} m/s ({self.speeds[-1]*3.6:.2f} km/h)
  Best Episode:           {np.max(self.speeds):.2f} m/s ({np.max(self.speeds)*3.6:.2f} km/h) (Episode {np.argmax(self.speeds)+1})
  Worst Episode:          {np.min(self.speeds):.2f} m/s ({np.min(self.speeds)*3.6:.2f} km/h) (Episode {np.argmin(self.speeds)+1})
  Average:                {np.mean(self.speeds):.2f} ± {np.std(self.speeds):.2f} m/s
  Last 5 Episodes:        {self.final_speed_avg:.2f} ± {self.final_speed_std:.2f} m/s
  
  Overall Improvement:    {self.speed_improvement:+.1f}%

TRAINING METRIC - Episode Reward:
  Best Reward:            {np.max(self.rewards):+.3f}
  Worst Reward:           {np.min(self.rewards):+.3f}
  Average Reward:         {np.mean(self.rewards):+.3f} ± {np.std(self.rewards):+.3f}
  Final Episode Reward:   {self.rewards[-1]:+.3f}
  Total Cumulative:       {np.sum(self.rewards):+.3f}

{'='*80}
CONVERGENCE ANALYSIS
{'='*80}

Early Phase (Episodes 1-{len(self.episode_numbers)//3}):
  Avg Waiting Time: {np.mean(self.waiting_times[:len(self.episode_numbers)//3]):.2f}s
  Avg Speed: {np.mean(self.speeds[:len(self.episode_numbers)//3]):.2f}m/s

Mid Phase (Episodes {len(self.episode_numbers)//3+1}-{2*len(self.episode_numbers)//3}):
  Avg Waiting Time: {np.mean(self.waiting_times[len(self.episode_numbers)//3:2*len(self.episode_numbers)//3]):.2f}s
  Avg Speed: {np.mean(self.speeds[len(self.episode_numbers)//3:2*len(self.episode_numbers)//3]):.2f}m/s

Late Phase (Episodes {2*len(self.episode_numbers)//3+1}-{len(self.episode_numbers)}):
  Avg Waiting Time: {np.mean(self.waiting_times[2*len(self.episode_numbers)//3:]):.2f}s
  Avg Speed: {np.mean(self.speeds[2*len(self.episode_numbers)//3:]):.2f}m/s

Stability (Std Dev in Late Phase):
  Waiting Time: {np.std(self.waiting_times[-5:]):.2f}s
  Speed: {np.std(self.speeds[-5:]):.2f}m/s

{'='*80}
CORRELATION ANALYSIS
{'='*80}

Waiting Time vs Speed:      r = {np.corrcoef(self.waiting_times, self.speeds)[0,1]:+.3f}
Reward vs Waiting Time:     r = {np.corrcoef(self.rewards, self.waiting_times)[0,1]:+.3f}
Reward vs Speed:            r = {np.corrcoef(self.rewards, self.speeds)[0,1]:+.3f}

Interpretation:
  - Strong negative correlation between waiting time and speed indicates network
    is learning the correct optimization objective
  - Reward signal should correlate strongly with waiting time for good learning

{'='*80}
FINAL SIMULATION RESULTS
{'='*80}

Total Vehicles:             {self.final_results['vehicles']:,}
Simulation Duration:        {self.final_results['steps']:,} steps
Avg Waiting Time (Step):    {self.final_results['avg_waiting_step']:.2f} seconds
Avg Speed (Step):           {self.final_results['avg_speed_step']:.2f} m/s ({self.final_results['avg_speed_step']*3.6:.2f} km/h)
Avg Waiting Time (Vehicle): {self.final_results['avg_waiting_vehicle']:.2f} seconds
Avg Speed (Vehicle):        {self.final_results['avg_speed_vehicle']:.2f} m/s ({self.final_results['avg_speed_vehicle']*3.6:.2f} km/h)

{'='*80}
ANALYSIS COMPLETE
{'='*80}

The training shows {'successful convergence' if self.waiting_improvement > 0 else 'challenges in convergence'}.
Check generated visualizations for detailed trends and analysis.
"""
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"  ✓ Saved: {output_file}")


def compare_ql_vs_qmix(ql_file: str, qmix_file: str, output_dir: str = "./comparison"):
    """Compare Q-Learning and QMIX results."""
    
    with open(ql_file, 'r') as f:
        ql_results = json.load(f)
    
    with open(qmix_file, 'r') as f:
        qmix_results = json.load(f)
    
    ql_final = ql_results['final_results']
    qmix_final = qmix_results['final_results']
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Q-Learning vs QMIX Comparison', fontsize=16, fontweight='bold')
    
    algorithms = ['Q-Learning', 'QMIX']
    colors = ['skyblue', 'lightgreen']
    
    # Waiting time
    waiting_times = [ql_final['avg_waiting_vehicle'], qmix_final['avg_waiting_vehicle']]
    axes[0, 0].bar(algorithms, waiting_times, color=colors, edgecolor='black', linewidth=2)
    axes[0, 0].set_ylabel('Avg Waiting Time (s)', fontsize=11)
    axes[0, 0].set_title('Average Waiting Time (Lower is Better)', fontweight='bold')
    for i, (alg, wt) in enumerate(zip(algorithms, waiting_times)):
        axes[0, 0].text(i, wt + 1, f'{wt:.2f}s', ha='center', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Speed
    speeds = [ql_final['avg_speed_vehicle'], qmix_final['avg_speed_vehicle']]
    axes[0, 1].bar(algorithms, speeds, color=colors, edgecolor='black', linewidth=2)
    axes[0, 1].set_ylabel('Avg Speed (m/s)', fontsize=11)
    axes[0, 1].set_title('Average Speed (Higher is Better)', fontweight='bold')
    for i, (alg, sp) in enumerate(zip(algorithms, speeds)):
        axes[0, 1].text(i, sp + 0.2, f'{sp:.2f}m/s', ha='center', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Vehicles
    vehicles = [ql_final['vehicles'], qmix_final['vehicles']]
    axes[1, 0].bar(algorithms, vehicles, color=colors, edgecolor='black', linewidth=2)
    axes[1, 0].set_ylabel('Number of Vehicles', fontsize=11)
    axes[1, 0].set_title('Total Vehicles Processed', fontweight='bold')
    for i, (alg, vh) in enumerate(zip(algorithms, vehicles)):
        axes[1, 0].text(i, vh + 10, f'{vh}', ha='center', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Improvement metrics
    wait_improvement = (ql_final['avg_waiting_vehicle'] - qmix_final['avg_waiting_vehicle']) / ql_final['avg_waiting_vehicle'] * 100
    speed_improvement = (qmix_final['avg_speed_vehicle'] - ql_final['avg_speed_vehicle']) / ql_final['avg_speed_vehicle'] * 100
    
    metrics = [
        f"Waiting Time\nImprovement\\n{wait_improvement:+.1f}%",
        f"Speed\nImprovement\\n{speed_improvement:+.1f}%"
    ]
    values = [wait_improvement, speed_improvement]
    colors_imp = ['green' if v > 0 else 'red' for v in values]
    
    axes[1, 1].bar(range(len(metrics)), values, color=colors_imp, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=1)
    axes[1, 1].set_ylabel('Improvement (%)', fontsize=11)
    axes[1, 1].set_title('QMIX Improvement over Q-Learning', fontweight='bold')
    axes[1, 1].set_xticks(range(len(metrics)))
    axes[1, 1].set_xticklabels(['Waiting', 'Speed'])
    for i, v in enumerate(values):
        axes[1, 1].text(i, v + (5 if v > 0 else -10), f'{v:+.1f}%', ha='center', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(str(output_path / "comparison.png"), dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_path / 'comparison.png'}")
    plt.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python qmix_analysis.py analyze <results.json>")
        print("  python qmix_analysis.py compare <ql_results.json> <qmix_results.json>")
        sys.exit(1)
    
    if sys.argv[1] == "analyze" and len(sys.argv) > 2:
        analyzer = QMIXResultsAnalyzer(sys.argv[2])
        analyzer.print_summary()
        analyzer.generate_full_report()
    
    elif sys.argv[1] == "compare" and len(sys.argv) > 3:
        compare_ql_vs_qmix(sys.argv[2], sys.argv[3])
    
    else:
        print("Invalid arguments")
        sys.exit(1)
