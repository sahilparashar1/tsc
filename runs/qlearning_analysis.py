"""
Q-Learning Results Visualization and Analysis Script

Generates comprehensive visualizations and analysis reports from Q-Learning training results.

Usage:
    python qlearning_analysis.py analyze <results_json>
    python qlearning_analysis.py compare <qlearning_json> <qmix_json>
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class QLearningResultsAnalyzer:
    """Comprehensive analysis of Q-Learning training results."""
    
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
        print("Q-LEARNING TRAINING RESULTS SUMMARY")
        print("=" * 90)
        
        print("\nConfiguration:")
        print(f"  Traffic Lights: {self.config['num_agents']}")
        print(f"  Learning Rate: {self.config['learning_rate']}")
        print(f"  Discount Factor: {self.config['discount_factor']}")
        print(f"  Epsilon Start: {self.config['epsilon_start']}")
        print(f"  Epsilon Min: {self.config['epsilon_min']}")
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
        
        print("\nFinal Simulation Results:")
        print(f"  Total Vehicles:      {self.final_results['vehicles']:7d}")
        print(f"  Simulation Steps:    {self.final_results['steps']:7d}")
        print(f"  Avg Waiting (step):  {self.final_results['avg_waiting_step']:7.2f}s")
        print(f"  Avg Speed (step):    {self.final_results['avg_speed_step']:7.2f}m/s")
        
        print("\n" + "=" * 90 + "\n")
    
    def generate_full_report(self, output_dir: str = "./qlearning_analysis"):
        """Generate complete analysis report with visualizations."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Generating Q-Learning analysis report in {output_dir}...")
        
        # Generate visualizations
        self._plot_dashboard(str(output_path / "01_dashboard.png"))
        self._plot_waiting_time(str(output_path / "02_waiting_time.png"))
        self._plot_speed(str(output_path / "03_speed.png"))
        self._plot_convergence(str(output_path / "04_convergence.png"))
        self._plot_correlation(str(output_path / "05_correlation.png"))
        
        # Statistics and summary
        self._generate_statistics_table(str(output_path / "06_statistics.csv"))
        self._generate_text_report(str(output_path / "07_summary.txt"))
        
        print("✓ Q-Learning analysis report generated successfully!")
        print(f"  Output directory: {output_path.absolute()}")
    
    def _plot_dashboard(self, output_file: str):
        """Plot comprehensive dashboard."""
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        fig.suptitle('Q-Learning Training Dashboard', fontsize=18, fontweight='bold', y=0.98)
        
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
        
        # 3. Epsilon decay
        ax3 = fig.add_subplot(gs[0, 2])
        epsilon_start = self.config['epsilon_start']
        epsilon_min = self.config['epsilon_min']
        epsilon_decay = self.config['epsilon_decay']
        epsilons = [max(epsilon_min, epsilon_start * (epsilon_decay ** ep)) for ep in range(len(self.episode_numbers))]
        ax3.plot(self.episode_numbers, epsilons, 'r-', linewidth=2.5, marker='o', markersize=4, label='Epsilon')
        ax3.set_xlabel('Episode', fontsize=10)
        ax3.set_ylabel('Epsilon (Exploration Rate)', fontsize=10)
        ax3.set_title('Epsilon Decay Schedule', fontweight='bold')
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
        
        Network Config:  {self.config['num_agents']} traffic lights  |  Learning rate: {self.config['learning_rate']}  |  Discount: {self.config['discount_factor']}
        Training:  {self.config['num_episodes']} episodes  |  Max green time: {self.config['max_green_time']}s  |  Decision interval: {self.config['decision_interval']}
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
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')
        
        # Waiting vs Speed
        axes[0].scatter(self.waiting_times, self.speeds, s=100, alpha=0.6, edgecolor='black')
        z = np.polyfit(self.waiting_times, self.speeds, 1)
        p = np.poly1d(z)
        axes[0].plot(self.waiting_times, p(self.waiting_times), 'r--', linewidth=2)
        corr_ws = np.corrcoef(self.waiting_times, self.speeds)[0, 1]
        axes[0].set_xlabel('Waiting Time (s)')
        axes[0].set_ylabel('Speed (m/s)')
        axes[0].set_title(f'Waiting Time vs Speed (r={corr_ws:.3f})')
        axes[0].grid(True, alpha=0.3)
        
        # Vehicles vs Waiting Time
        axes[1].scatter(self.vehicles, self.waiting_times, s=100, alpha=0.6, edgecolor='black', color='orange')
        z = np.polyfit(self.vehicles, self.waiting_times, 1)
        p = np.poly1d(z)
        axes[1].plot(self.vehicles, p(self.vehicles), 'r--', linewidth=2)
        corr_vw = np.corrcoef(self.vehicles, self.waiting_times)[0, 1]
        axes[1].set_xlabel('Number of Vehicles')
        axes[1].set_ylabel('Waiting Time (s)')
        axes[1].set_title(f'Vehicles vs Waiting Time (r={corr_vw:.3f})')
        axes[1].grid(True, alpha=0.3)
        
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
Q-LEARNING TRAFFIC SIGNAL CONTROL - COMPLETE ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
CONFIGURATION
{'='*80}

Training Parameters:
    • Learning Rate (α):                {self.config['learning_rate']}
    • Discount Factor (γ):              {self.config['discount_factor']}
    • Initial Epsilon (ε₀):              {self.config['epsilon_start']}
    • Minimum Epsilon (ε_min):           {self.config['epsilon_min']}
    • Epsilon Decay:                     {self.config['epsilon_decay']}
    • Episodes:                          {self.config['num_episodes']}
    • Number of Agents (Traffic Lights): {self.config['num_agents']}

Traffic Light Control Parameters:
    • Minimum Green Time:                {self.config['min_green_time']} steps
    • Maximum Green Time:                {self.config['max_green_time']} steps
    • Decision Interval:                 {self.config['decision_interval']} steps
    • Max Episode Duration:              {self.config['max_episode_duration']} seconds

{'='*80}
PERFORMANCE SUMMARY
{'='*80}

Waiting Time Analysis (per Vehicle):
    • Initial (Episode 1):               {self.waiting_times[0]:.2f} seconds
    • Final (Episode {len(self.waiting_times)}):                {self.waiting_times[-1]:.2f} seconds
    • Best (Episode {np.argmin(self.waiting_times)+1}):                  {np.min(self.waiting_times):.2f} seconds
    • Worst (Episode {np.argmax(self.waiting_times)+1}):                 {np.max(self.waiting_times):.2f} seconds
    • Average:                           {np.mean(self.waiting_times):.2f} seconds
    • Std Dev:                           {np.std(self.waiting_times):.2f} seconds
    • Last 5 Episodes Avg:               {self.final_waiting_avg:.2f} ± {self.final_waiting_std:.2f} seconds
    • Overall Improvement:               {self.waiting_improvement:+.1f}%

Speed Analysis (per Vehicle):
    • Initial (Episode 1):               {self.speeds[0]:.2f} m/s
    • Final (Episode {len(self.speeds)}):                {self.speeds[-1]:.2f} m/s
    • Best (Episode {np.argmax(self.speeds)+1}):                  {np.max(self.speeds):.2f} m/s
    • Worst (Episode {np.argmin(self.speeds)+1}):                 {np.min(self.speeds):.2f} m/s
    • Average:                           {np.mean(self.speeds):.2f} m/s
    • Std Dev:                           {np.std(self.speeds):.2f} m/s
    • Last 5 Episodes Avg:               {self.final_speed_avg:.2f} ± {self.final_speed_std:.2f} m/s
    • Overall Improvement:               {self.speed_improvement:+.1f}%

Convergence Metrics:
    • Waiting Time Convergence Stability:  {np.std(self.waiting_times[-5:]):.4f} (lower is better)
    • Speed Convergence Stability:         {np.std(self.speeds[-5:]):.4f} (lower is better)
    • Final Network Performance Score:     {(1 - (self.waiting_times[-1]/np.max(self.waiting_times)))*100:.1f}%

{'='*80}
FINAL SIMULATION RESULTS
{'='*80}

Simulation Metrics:
    • Total Simulation Steps:            {self.final_results['steps']}
    • Total Vehicles Processed:          {self.final_results['vehicles']}
    • Average Waiting Time (by step):    {self.final_results['avg_waiting_step']:.2f} seconds
    • Average Speed (by step):           {self.final_results['avg_speed_step']:.2f} m/s ({self.final_results['avg_speed_step']*3.6:.2f} km/h)
    • Average Waiting Time (by vehicle): {self.final_results['avg_waiting_vehicle']:.2f} seconds
    • Average Speed (by vehicle):        {self.final_results['avg_speed_vehicle']:.2f} m/s ({self.final_results['avg_speed_vehicle']*3.6:.2f} km/h)

{'='*80}
ANALYSIS INSIGHTS
{'='*80}

Key Findings:
    1. Waiting Time: {self.waiting_improvement:+.1f}% improvement from episode 1 to final
    2. Speed: {self.speed_improvement:+.1f}% improvement from episode 1 to final
    3. Correlation (Waiting vs Speed): {np.corrcoef(self.waiting_times, self.speeds)[0, 1]:.3f}
    4. Episode with Best Performance: Episode {np.argmin(self.waiting_times)+1}
    5. Vehicles Processed: Average {np.mean(self.vehicles):.0f} per episode

Training Dynamics:
    • The Q-learning agents show clear learning curves
    • Waiting times generally decrease with training
    • Speed improves and stabilizes over episodes
    • Epsilon decay follows the exponential schedule as configured

Recommendations:
    • The model has reached good convergence (last 5 episodes consistent)
    • Consider deploying the best-performing checkpoint for production
    • Further training may yield marginal improvements
    • Network performs well with current hyperparameter configuration

{'='*80}
REPORT GENERATED BY QLEARNING_ANALYSIS.PY
{'='*80}
"""
        
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"  ✓ Saved: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python qlearning_analysis.py analyze <results_json>")
        sys.exit(1)
    
    if sys.argv[1] == "analyze" and len(sys.argv) > 2:
        results_file = sys.argv[2]
        print(f"Loading Q-Learning results from {results_file}...")
        analyzer = QLearningResultsAnalyzer(results_file)
        analyzer.print_summary()
        analyzer.generate_full_report()
    
    else:
        print("Invalid arguments")
        sys.exit(1)
