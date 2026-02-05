"""
QMIX Configuration Loader and Utility Functions

Provides functionality to:
1. Load QMIX configuration from INI files
2. Create configurable training runs
3. Compare Q-Learning vs QMIX results
"""

import configparser
from pathlib import Path
from typing import Dict, Any, Optional
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class QMIXConfigLoader:
    """Load and parse QMIX configuration from INI files."""
    
    def __init__(self, config_path: str):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to .ini configuration file
        """
        self.config_path = Path(config_path)
        self.config = configparser.ConfigParser()
        self.config.read(self.config_path)
        
    def get_network_config(self) -> Dict[str, int]:
        """Get neural network architecture parameters."""
        return {
            'agent_hidden_dim': self.config.getint('NETWORK_CONFIG', 'agent_hidden_dim'),
            'agent_rnn_hidden_dim': self.config.getint('NETWORK_CONFIG', 'agent_rnn_hidden_dim'),
            'mixing_embed_dim': self.config.getint('NETWORK_CONFIG', 'mixing_embed_dim'),
            'hypernet_embed_dim': self.config.getint('NETWORK_CONFIG', 'hypernet_embed_dim'),
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get QMIX training hyperparameters."""
        return {
            'learning_rate': self.config.getfloat('TRAINING_CONFIG', 'learning_rate'),
            'discount_factor': self.config.getfloat('TRAINING_CONFIG', 'discount_factor'),
            'target_update_freq': self.config.getint('TRAINING_CONFIG', 'target_update_freq'),
            'grad_clip': self.config.getfloat('TRAINING_CONFIG', 'grad_clip'),
            'epsilon_start': self.config.getfloat('TRAINING_CONFIG', 'epsilon_start'),
            'epsilon_min': self.config.getfloat('TRAINING_CONFIG', 'epsilon_min'),
            'epsilon_decay': self.config.getfloat('TRAINING_CONFIG', 'epsilon_decay'),
            'batch_size': self.config.getint('TRAINING_CONFIG', 'batch_size'),
            'replay_buffer_capacity': self.config.getint('TRAINING_CONFIG', 'replay_buffer_capacity'),
            'train_steps_per_episode': self.config.getint('TRAINING_CONFIG', 'train_steps_per_episode'),
            'max_episode_length': self.config.getint('TRAINING_CONFIG', 'max_episode_length'),
        }
    
    def get_reward_config(self) -> Dict[str, float]:
        """Get reward shaping parameters."""
        return {
            'reward_scale': self.config.getfloat('REWARD_CONFIG', 'reward_scale'),
            'queue_weight': self.config.getfloat('REWARD_CONFIG', 'queue_weight'),
            'flickering_penalty': self.config.getfloat('REWARD_CONFIG', 'flickering_penalty'),
        }
    
    def get_signal_config(self) -> Dict[str, int]:
        """Get traffic signal timing parameters."""
        return {
            'min_green_time': self.config.getint('SIGNAL_CONFIG', 'min_green_time'),
            'max_green_time': self.config.getint('SIGNAL_CONFIG', 'max_green_time'),
            'yellow_time': self.config.getint('SIGNAL_CONFIG', 'yellow_time'),
        }
    
    def get_simulation_config(self) -> Dict[str, Any]:
        """Get SUMO simulation parameters."""
        return {
            'sumo_config': self.config.get('SIMULATION_CONFIG', 'sumo_config'),
            'use_gui': self.config.getboolean('SIMULATION_CONFIG', 'use_gui'),
            'decision_interval': self.config.getint('SIMULATION_CONFIG', 'decision_interval'),
            'waiting_time_memory': self.config.getint('SIMULATION_CONFIG', 'waiting_time_memory'),
        }
    
    def get_training_control(self) -> Dict[str, Any]:
        """Get training control parameters."""
        return {
            'num_episodes': self.config.getint('TRAINING_CONTROL', 'num_episodes'),
            'test_interval': self.config.getint('TRAINING_CONTROL', 'test_interval'),
            'save_checkpoint_interval': self.config.getint('TRAINING_CONTROL', 'save_checkpoint_interval'),
            'device': self.config.get('TRAINING_CONTROL', 'device'),
        }
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all configuration sections."""
        return {
            'network': self.get_network_config(),
            'training': self.get_training_config(),
            'reward': self.get_reward_config(),
            'signal': self.get_signal_config(),
            'simulation': self.get_simulation_config(),
            'control': self.get_training_control(),
        }
    
    def print_config(self):
        """Pretty print all configuration."""
        configs = self.get_all_configs()
        print("=" * 70)
        print("QMIX CONFIGURATION")
        print("=" * 70)
        
        for section, params in configs.items():
            print(f"\n[{section.upper()}]")
            for key, value in params.items():
                print(f"  {key:<30} = {value}")
        
        print("\n" + "=" * 70)


class ResultsAnalyzer:
    """Analyze and compare training results."""
    
    @staticmethod
    def load_results(result_file: str) -> Dict:
        """Load results from JSON file."""
        with open(result_file, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def compare_algorithms(qlearning_file: str, qmix_file: str) -> Dict:
        """
        Compare Q-Learning and QMIX results.
        
        Args:
            qlearning_file: Path to Q-Learning results JSON
            qmix_file: Path to QMIX results JSON
            
        Returns:
            Dictionary with comparison statistics
        """
        qlearning = ResultsAnalyzer.load_results(qlearning_file)
        qmix = ResultsAnalyzer.load_results(qmix_file)
        
        qlearning_final = qlearning['final_results']
        qmix_final = qmix['final_results']
        
        comparison = {
            'algorithm': {
                'Q-Learning': {
                    'avg_waiting_vehicle': qlearning_final['avg_waiting_vehicle'],
                    'avg_speed_vehicle': qlearning_final['avg_speed_vehicle'],
                    'steps': qlearning_final['steps'],
                    'vehicles': qlearning_final['vehicles'],
                },
                'QMIX': {
                    'avg_waiting_vehicle': qmix_final['avg_waiting_vehicle'],
                    'avg_speed_vehicle': qmix_final['avg_speed_vehicle'],
                    'steps': qmix_final['steps'],
                    'vehicles': qmix_final['vehicles'],
                }
            },
            'improvements': {
                'waiting_time_reduction': (
                    (qlearning_final['avg_waiting_vehicle'] - qmix_final['avg_waiting_vehicle']) /
                    qlearning_final['avg_waiting_vehicle'] * 100
                ) if qlearning_final['avg_waiting_vehicle'] > 0 else 0,
                'speed_improvement': (
                    (qmix_final['avg_speed_vehicle'] - qlearning_final['avg_speed_vehicle']) /
                    qlearning_final['avg_speed_vehicle'] * 100
                ) if qlearning_final['avg_speed_vehicle'] > 0 else 0,
            }
        }
        
        return comparison
    
    @staticmethod
    def print_comparison(comparison: Dict):
        """Print formatted comparison."""
        print("\n" + "=" * 80)
        print("Q-LEARNING vs QMIX COMPARISON")
        print("=" * 80)
        
        print("\nQ-Learning Results:")
        ql = comparison['algorithm']['Q-Learning']
        print(f"  Average Waiting Time: {ql['avg_waiting_vehicle']:.2f}s")
        print(f"  Average Speed: {ql['avg_speed_vehicle']:.2f}m/s ({ql['avg_speed_vehicle']*3.6:.2f}km/h)")
        print(f"  Total Vehicles: {ql['vehicles']}")
        
        print("\nQMIX Results:")
        qm = comparison['algorithm']['QMIX']
        print(f"  Average Waiting Time: {qm['avg_waiting_vehicle']:.2f}s")
        print(f"  Average Speed: {qm['avg_speed_vehicle']:.2f}m/s ({qm['avg_speed_vehicle']*3.6:.2f}km/h)")
        print(f"  Total Vehicles: {qm['vehicles']}")
        
        print("\nImprovement (QMIX over Q-Learning):")
        imp = comparison['improvements']
        print(f"  Waiting Time Reduction: {imp['waiting_time_reduction']:+.1f}%")
        print(f"  Speed Improvement: {imp['speed_improvement']:+.1f}%")
        
        if imp['waiting_time_reduction'] > 0:
            print(f"\n✓ QMIX achieved {imp['waiting_time_reduction']:.1f}% less waiting time!")
        else:
            print(f"\n✗ Q-Learning was {abs(imp['waiting_time_reduction']):.1f}% better")
        
        print("=" * 80 + "\n")
    
    @staticmethod
    def plot_training_curves(qmix_file: str, output_file: str = "./training_curves.png"):
        """
        Plot training curves from QMIX results.
        
        Args:
            qmix_file: Path to QMIX results JSON
            output_file: Where to save the plot
        """
        results = ResultsAnalyzer.load_results(qmix_file)
        episodes = results['episode_results']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('QMIX Training Curves', fontsize=16, fontweight='bold')
        
        # Extract metrics
        episode_nums = list(range(1, len(episodes) + 1))
        waiting_times = [ep['avg_waiting_vehicle'] for ep in episodes]
        speeds = [ep['avg_speed_vehicle'] for ep in episodes]
        rewards = [ep.get('episode_reward', 0) for ep in episodes]
        vehicles = [ep['vehicles'] for ep in episodes]
        
        # Waiting time
        axes[0, 0].plot(episode_nums, waiting_times, 'b-', linewidth=2, marker='o')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Avg Waiting Time (s)')
        axes[0, 0].set_title('Average Waiting Time per Episode')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Speed
        axes[0, 1].plot(episode_nums, speeds, 'g-', linewidth=2, marker='s')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Avg Speed (m/s)')
        axes[0, 1].set_title('Average Speed per Episode')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Reward
        axes[1, 0].plot(episode_nums, rewards, 'r-', linewidth=2, marker='^')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Episode Reward')
        axes[1, 0].set_title('QMIX Episode Reward')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Vehicles
        axes[1, 1].plot(episode_nums, vehicles, 'purple', linewidth=2, marker='d')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Number of Vehicles')
        axes[1, 1].set_title('Vehicles per Episode')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {output_file}")
        plt.close()
    
    @staticmethod
    def print_statistics(result_file: str):
        """Print detailed statistics from results file."""
        results = ResultsAnalyzer.load_results(result_file)
        episodes = results['episode_results']
        final = results['final_results']
        config = results['configuration']
        
        print("\n" + "=" * 80)
        print("QMIX TRAINING STATISTICS")
        print("=" * 80)
        
        print("\nConfiguration:")
        print(f"  Number of Agents (Traffic Lights): {config['num_agents']}")
        print(f"  Observation Dimension: {config['obs_dim']}")
        print(f"  State Dimension: {config['state_dim']}")
        print(f"  Learning Rate: {config['learning_rate']}")
        print(f"  Discount Factor (γ): {config['discount_factor']}")
        print(f"  Total Episodes: {config['num_episodes']}")
        print(f"  Batch Size: {config['batch_size']}")
        
        waiting_times = [ep['avg_waiting_vehicle'] for ep in episodes]
        speeds = [ep['avg_speed_vehicle'] for ep in episodes]
        rewards = [ep.get('episode_reward', 0) for ep in episodes]
        
        print("\nWaiting Time Statistics:")
        print(f"  First Episode:  {waiting_times[0]:.2f}s")
        print(f"  Best Episode:   {min(waiting_times):.2f}s (episode {waiting_times.index(min(waiting_times)) + 1})")
        print(f"  Final Episode:  {final['avg_waiting_vehicle']:.2f}s")
        print(f"  Average:        {np.mean(waiting_times):.2f}s")
        print(f"  Std Dev:        {np.std(waiting_times):.2f}s")
        print(f"  Improvement:    {(waiting_times[0] - waiting_times[-1])/waiting_times[0]*100:+.1f}%")
        
        print("\nSpeed Statistics:")
        print(f"  First Episode:  {speeds[0]:.2f}m/s")
        print(f"  Best Episode:   {max(speeds):.2f}m/s (episode {speeds.index(max(speeds)) + 1})")
        print(f"  Final Episode:  {final['avg_speed_vehicle']:.2f}m/s")
        print(f"  Average:        {np.mean(speeds):.2f}m/s")
        print(f"  Std Dev:        {np.std(speeds):.2f}m/s")
        
        print("\nReward Statistics:")
        print(f"  Best Reward:    {max(rewards):.3f}")
        print(f"  Worst Reward:   {min(rewards):.3f}")
        print(f"  Final Reward:   {final.get('episode_reward', 0):.3f}")
        print(f"  Average Reward: {np.mean(rewards):.3f}")
        
        print("\nFinal Evaluation Results:")
        print(f"  Simulation Steps: {final['steps']}")
        print(f"  Total Vehicles: {final['vehicles']}")
        print(f"  Avg Waiting Time (step-based): {final['avg_waiting_step']:.2f}s")
        print(f"  Avg Speed (step-based): {final['avg_speed_step']:.2f}m/s")
        
        print("\n" + "=" * 80 + "\n")


def create_training_script(config_path: str, output_path: str = "./run_qmix_configured.py"):
    """
    Generate a customized training script based on configuration file.
    
    Args:
        config_path: Path to .ini config file
        output_path: Where to save generated script
    """
    loader = QMIXConfigLoader(config_path)
    configs = loader.get_all_configs()
    
    # Read original script
    original_script = Path(__file__).parent / "run_qmix.py"
    
    with open(original_script, 'r') as f:
        script_content = f.read()
    
    # Replace configuration values
    replacements = {
        # Network config
        'AGENT_HIDDEN_DIM = 64': f"AGENT_HIDDEN_DIM = {configs['network']['agent_hidden_dim']}",
        'AGENT_RNN_HIDDEN_DIM = 64': f"AGENT_RNN_HIDDEN_DIM = {configs['network']['agent_rnn_hidden_dim']}",
        'MIXING_EMBED_DIM = 32': f"MIXING_EMBED_DIM = {configs['network']['mixing_embed_dim']}",
        'HYPERNET_EMBED_DIM = 64': f"HYPERNET_EMBED_DIM = {configs['network']['hypernet_embed_dim']}",
        
        # Training config
        'LEARNING_RATE = 5e-4': f"LEARNING_RATE = {configs['training']['learning_rate']}",
        'DISCOUNT_FACTOR = 0.99': f"DISCOUNT_FACTOR = {configs['training']['discount_factor']}",
        'TARGET_UPDATE_FREQ = 200': f"TARGET_UPDATE_FREQ = {configs['training']['target_update_freq']}",
        'GRAD_CLIP = 10.0': f"GRAD_CLIP = {configs['training']['grad_clip']}",
        'EPSILON_START = 1.0': f"EPSILON_START = {configs['training']['epsilon_start']}",
        'EPSILON_MIN = 0.05': f"EPSILON_MIN = {configs['training']['epsilon_min']}",
        'EPSILON_DECAY = 0.995': f"EPSILON_DECAY = {configs['training']['epsilon_decay']}",
        'BATCH_SIZE = 32': f"BATCH_SIZE = {configs['training']['batch_size']}",
        'REPLAY_BUFFER_CAPACITY = 5000': f"REPLAY_BUFFER_CAPACITY = {configs['training']['replay_buffer_capacity']}",
        'TRAIN_STEPS_PER_EPISODE = 10': f"TRAIN_STEPS_PER_EPISODE = {configs['training']['train_steps_per_episode']}",
        'MAX_EPISODE_LENGTH = 200': f"MAX_EPISODE_LENGTH = {configs['training']['max_episode_length']}",
        
        # Signal config
        'MIN_GREEN_TIME = 15': f"MIN_GREEN_TIME = {configs['signal']['min_green_time']}",
        'MAX_GREEN_TIME = 120': f"MAX_GREEN_TIME = {configs['signal']['max_green_time']}",
        
        # Control config
        'NUM_EPISODES = 50': f"NUM_EPISODES = {configs['control']['num_episodes']}",
        'TEST_INTERVAL = 10': f"TEST_INTERVAL = {configs['control']['test_interval']}",
        'SAVE_CHECKPOINT_INTERVAL = 10': f"SAVE_CHECKPOINT_INTERVAL = {configs['control']['save_checkpoint_interval']}",
    }
    
    for old, new in replacements.items():
        script_content = script_content.replace(old, new)
    
    # Save new script
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    print(f"Generated configured training script: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python qmix_utils.py print-config <config_file>")
        print("  python qmix_utils.py generate-script <config_file> [output_file]")
        print("  python qmix_utils.py compare <qlearning_results> <qmix_results>")
        print("  python qmix_utils.py plot <qmix_results> [output_file]")
        print("  python qmix_utils.py stats <qmix_results>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "print-config" and len(sys.argv) >= 3:
        loader = QMIXConfigLoader(sys.argv[2])
        loader.print_config()
    
    elif command == "generate-script":
        config_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else "./run_qmix_configured.py"
        create_training_script(config_file, output_file)
    
    elif command == "compare" and len(sys.argv) >= 4:
        comparison = ResultsAnalyzer.compare_algorithms(sys.argv[2], sys.argv[3])
        ResultsAnalyzer.print_comparison(comparison)
    
    elif command == "plot" and len(sys.argv) >= 3:
        output_file = sys.argv[3] if len(sys.argv) > 3 else "./training_curves.png"
        ResultsAnalyzer.plot_training_curves(sys.argv[2], output_file)
    
    elif command == "stats" and len(sys.argv) >= 3:
        ResultsAnalyzer.print_statistics(sys.argv[2])
    
    else:
        print("Unknown command or missing arguments")
        sys.exit(1)
