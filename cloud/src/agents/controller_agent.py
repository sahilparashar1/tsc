"""
Controller Agent - Global Cooperative Logic
QMIX-based Multi-Agent Controller for Traffic Signal Optimization
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid

import torch
import paho.mqtt.client as mqtt

from .qmix_network import QMIXNetwork, QMIXTrainer, MaxPressureReward
from ..database import get_database, DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ControllerAgent:
    """
    Global Controller Agent using QMIX architecture.
    Coordinates all regional agents and computes global Q_tot.
    """
    
    def __init__(
        self,
        num_regions: int = 6,
        agents_per_region: int = 4,
        obs_dim: int = 20,  # Queue lengths + phase info per intersection
        action_dim: int = 4,  # Number of signal phases
        mqtt_broker: str = None,
        mqtt_port: int = 1883
    ):
        self.num_regions = num_regions
        self.agents_per_region = agents_per_region
        self.total_agents = num_regions * agents_per_region
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize QMIX network
        state_dim = obs_dim * self.total_agents  # Global state = concat of all observations
        self.qmix = QMIXNetwork(
            n_agents=self.total_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            hidden_dim=64,
            rnn_hidden_dim=64,
            mixing_embed_dim=32
        )
        
        # Trainer
        self.trainer = QMIXTrainer(
            self.qmix,
            lr=5e-4,
            gamma=0.99,
            target_update_freq=200,
            device=str(self.device)
        )
        
        # MQTT setup
        self.mqtt_broker = mqtt_broker or os.getenv('MQTT_BROKER', 'localhost')
        self.mqtt_port = mqtt_port
        self.mqtt_client = mqtt.Client(client_id="controller_agent")
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        
        # State management
        self.current_observations: Dict[str, Dict] = {}  # intersection_id -> obs
        self.hidden_states: List[torch.Tensor] = [None] * self.total_agents
        self.last_actions: Dict[str, int] = {}
        self.region_mapping: Dict[str, int] = {}  # intersection_id -> region_id
        self.agent_mapping: Dict[str, int] = {}   # intersection_id -> agent_index
        
        # Training state
        self.current_episode_id = uuid.uuid4()
        self.episode_step = 0
        self.episode_buffer = []
        
        # Database
        self.db: Optional[DatabaseManager] = None
        
        # Exploration
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        # Training control
        self.training_enabled = True
        self.update_interval = 5  # Train every N steps
        
    def initialize(self, region_mapping: Dict[str, int]):
        """
        Initialize controller with region mapping.
        
        Args:
            region_mapping: Dictionary mapping intersection_id to region_id
        """
        self.region_mapping = region_mapping
        
        # Create agent mapping (sorted by intersection_id within each region)
        intersections_by_region = {}
        for intersection_id, region_id in region_mapping.items():
            if region_id not in intersections_by_region:
                intersections_by_region[region_id] = []
            intersections_by_region[region_id].append(intersection_id)
            
        # Sort and assign agent indices
        agent_idx = 0
        for region_id in sorted(intersections_by_region.keys()):
            for intersection_id in sorted(intersections_by_region[region_id]):
                self.agent_mapping[intersection_id] = agent_idx
                agent_idx += 1
                
        logger.info(f"Initialized controller with {len(self.agent_mapping)} agents")
        
    def connect_mqtt(self):
        """Connect to MQTT broker."""
        try:
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
            logger.info(f"Connected to MQTT broker at {self.mqtt_broker}:{self.mqtt_port}")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT: {e}")
            raise
            
    def connect_database(self):
        """Connect to database."""
        try:
            self.db = get_database()
            logger.info("Connected to database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            # Subscribe to regional agent telemetry
            client.subscribe("traffic/region/+/state")
            # Subscribe to intersection aggregated data
            client.subscribe("traffic/intersection/+/aggregated")
            logger.info("Subscribed to traffic topics")
        else:
            logger.error(f"MQTT connection failed with code {rc}")
            
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            payload = json.loads(msg.payload.decode())
            topic_parts = msg.topic.split('/')
            
            if topic_parts[1] == 'intersection':
                intersection_id = topic_parts[2]
                self._process_intersection_state(intersection_id, payload)
            elif topic_parts[1] == 'region':
                region_id = int(topic_parts[2])
                self._process_region_update(region_id, payload)
                
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
            
    def _process_intersection_state(self, intersection_id: str, state: Dict):
        """Process aggregated state from an intersection controller."""
        self.current_observations[intersection_id] = {
            'timestamp': state.get('timestamp'),
            'queue_lengths': state.get('queue_lengths', {}),
            'current_phase': state.get('current_phase', 0),
            'ambulance_detected': state.get('ambulance_detected', False),
            'counts': {
                'car': state.get('count_car', 0),
                'bike': state.get('count_bike', 0),
                'hmv': state.get('count_hmv', 0),
                'auto': state.get('count_auto', 0),
                'ambulance': state.get('count_ambulance', 0)
            }
        }
        
        # Check for ambulance override
        if state.get('ambulance_detected', False):
            self._handle_ambulance_override(intersection_id, state)
            
    def _process_region_update(self, region_id: int, update: Dict):
        """Process update from a regional agent."""
        # Update regional metrics if needed
        pass
        
    def _handle_ambulance_override(self, intersection_id: str, state: Dict):
        """Handle ambulance detection with immediate override."""
        logger.warning(f"Ambulance detected at {intersection_id}! Issuing override.")
        
        # Determine which lane has ambulance
        ambulance_lane = state.get('ambulance_lane', 0)
        
        # Issue GREEN override for that lane
        override_command = {
            'intersection_id': intersection_id,
            'action': 'OVERRIDE',
            'phase': ambulance_lane,
            'duration': 30,  # 30 second override
            'reason': 'AMBULANCE_DETECTED'
        }
        
        self.mqtt_client.publish(
            f"traffic/intersection/{intersection_id}/command",
            json.dumps(override_command)
        )
        
        # Log to database
        if self.db:
            self.db.record_signal_command(
                intersection_id=intersection_id,
                phase=str(ambulance_lane),
                duration=30,
                is_override=True,
                override_reason='AMBULANCE_DETECTED'
            )
            
    def _build_observation_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build observation and global state tensors from current observations."""
        obs_list = []
        
        for intersection_id in sorted(self.agent_mapping.keys()):
            if intersection_id in self.current_observations:
                obs = self.current_observations[intersection_id]
                
                # Build observation vector
                queue_lengths = list(obs.get('queue_lengths', {}).values())
                # Pad to fixed size
                while len(queue_lengths) < 8:
                    queue_lengths.append(0)
                    
                # One-hot phase
                phase = obs.get('current_phase', 0)
                phase_onehot = [0] * self.action_dim
                if 0 <= phase < self.action_dim:
                    phase_onehot[phase] = 1
                    
                # Vehicle counts
                counts = obs.get('counts', {})
                count_vec = [
                    counts.get('car', 0) / 50.0,  # Normalize
                    counts.get('bike', 0) / 20.0,
                    counts.get('hmv', 0) / 10.0,
                    counts.get('auto', 0) / 30.0
                ]
                
                obs_vec = queue_lengths[:8] + phase_onehot + count_vec
            else:
                # Default zero observation
                obs_vec = [0] * self.obs_dim
                
            # Pad/truncate to obs_dim
            obs_vec = obs_vec[:self.obs_dim]
            while len(obs_vec) < self.obs_dim:
                obs_vec.append(0)
                
            obs_list.append(obs_vec)
            
        # [1, 1, n_agents, obs_dim]
        obs_tensor = torch.FloatTensor([obs_list]).unsqueeze(1)
        
        # Global state = flattened observations
        state_tensor = obs_tensor.view(1, 1, -1)
        
        return obs_tensor, state_tensor
        
    def compute_actions(self) -> Dict[str, int]:
        """Compute actions for all intersections."""
        obs, state = self._build_observation_tensor()
        obs = obs.to(self.device)
        state = state.to(self.device)
        
        # Select actions
        actions, new_hidden = self.trainer.select_actions(
            obs, self.hidden_states, epsilon=self.epsilon
        )
        self.hidden_states = new_hidden
        
        # Map actions to intersection IDs
        action_dict = {}
        actions_np = actions.cpu().numpy()[0, 0, :]  # [n_agents]
        
        for intersection_id, agent_idx in self.agent_mapping.items():
            action_dict[intersection_id] = int(actions_np[agent_idx])
            
        return action_dict
        
    def publish_actions(self, actions: Dict[str, int]):
        """Publish action commands to intersection controllers."""
        for intersection_id, action in actions.items():
            command = {
                'intersection_id': intersection_id,
                'action': 'SET_PHASE',
                'phase': action,
                'duration': 30,  # Default phase duration
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.mqtt_client.publish(
                f"traffic/intersection/{intersection_id}/command",
                json.dumps(command)
            )
            
        self.last_actions = actions
        
    def step(self) -> Dict[str, int]:
        """
        Execute one control step.
        
        Returns:
            actions: Dictionary of intersection_id -> action
        """
        # Compute actions
        actions = self.compute_actions()
        
        # Publish to intersections
        self.publish_actions(actions)
        
        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Training step
        self.episode_step += 1
        if self.training_enabled and self.episode_step % self.update_interval == 0:
            loss = self.trainer.train_step(batch_size=32)
            if loss is not None:
                logger.debug(f"Training loss: {loss:.4f}")
                
        return actions
        
    def run(self, control_interval: float = 5.0):
        """
        Main control loop.
        
        Args:
            control_interval: Seconds between control steps
        """
        logger.info("Starting controller agent main loop")
        
        try:
            while True:
                actions = self.step()
                logger.info(f"Step {self.episode_step}: Issued {len(actions)} actions, epsilon={self.epsilon:.3f}")
                asyncio.get_event_loop().run_until_complete(asyncio.sleep(control_interval))
                
        except KeyboardInterrupt:
            logger.info("Controller agent stopped")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Cleanup resources."""
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        
        # Save model checkpoint
        self.trainer.save('controller_checkpoint.pt')
        logger.info("Saved model checkpoint")


def main():
    """Entry point for controller agent."""
    # Configuration from environment
    num_regions = int(os.getenv('NUM_REGIONS', 6))
    agents_per_region = int(os.getenv('NUM_AGENTS_PER_REGION', 4))
    mqtt_broker = os.getenv('MQTT_BROKER', 'localhost')
    mqtt_port = int(os.getenv('MQTT_PORT', 1883))
    
    # Create controller
    controller = ControllerAgent(
        num_regions=num_regions,
        agents_per_region=agents_per_region,
        mqtt_broker=mqtt_broker,
        mqtt_port=mqtt_port
    )
    
    # Initialize with region mapping (will be updated from network parser)
    # This is a placeholder - actual mapping comes from intersection controllers
    region_mapping = {}
    for region in range(num_regions):
        for agent in range(agents_per_region):
            node_id = region * agents_per_region + agent + 1
            region_mapping[str(node_id)] = region
            
    controller.initialize(region_mapping)
    
    # Connect services
    controller.connect_mqtt()
    controller.connect_database()
    
    # Run
    controller.run(control_interval=5.0)


if __name__ == '__main__':
    main()
