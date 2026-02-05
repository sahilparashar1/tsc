"""
Regional Agent - Shared-Reward Regional Logic
DRQN-based agent for local traffic signal optimization
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import torch
import paho.mqtt.client as mqtt

from .qmix_network import DRQNAgent, MaxPressureReward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegionalAgent:
    """
    Regional Agent responsible for a cluster of intersections.
    Uses DRQN for local Q-value computation, reports to Controller.
    """ 
    
    def __init__(
        self,
        region_id: int,
        intersection_ids: List[str],
        obs_dim: int = 20,
        action_dim: int = 4,
        mqtt_broker: str = None,
        mqtt_port: int = 1883
    ):
        self.region_id = region_id
        self.intersection_ids = intersection_ids
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create local DRQN agents for each intersection
        self.agents: Dict[str, DRQNAgent] = {}
        self.hidden_states: Dict[str, torch.Tensor] = {}
        
        for intersection_id in intersection_ids:
            self.agents[intersection_id] = DRQNAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=64,
                rnn_hidden_dim=64
            ).to(self.device)
            self.hidden_states[intersection_id] = None
            
        # Reward function
        self.reward_fn = MaxPressureReward(flickering_penalty=0.1)
        
        # State tracking
        self.current_observations: Dict[str, Dict] = {}
        self.last_actions: Dict[str, int] = {}
        
        # MQTT setup
        self.mqtt_broker = mqtt_broker or os.getenv('MQTT_BROKER', 'localhost')
        self.mqtt_port = mqtt_port
        self.mqtt_client = mqtt.Client(client_id=f"regional_agent_{region_id}")
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        
    def connect_mqtt(self):
        """Connect to MQTT broker."""
        try:
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
            logger.info(f"Region {self.region_id}: Connected to MQTT")
        except Exception as e:
            logger.error(f"Region {self.region_id}: MQTT connection failed: {e}")
            raise
            
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            # Subscribe to intersection states in this region
            for intersection_id in self.intersection_ids:
                client.subscribe(f"traffic/intersection/{intersection_id}/aggregated")
                
            # Subscribe to commands from controller
            client.subscribe(f"traffic/region/{self.region_id}/command")
            
            logger.info(f"Region {self.region_id}: Subscribed to topics")
        else:
            logger.error(f"Region {self.region_id}: MQTT connection failed: {rc}")
            
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            payload = json.loads(msg.payload.decode())
            topic_parts = msg.topic.split('/')
            
            if topic_parts[1] == 'intersection':
                intersection_id = topic_parts[2]
                if intersection_id in self.intersection_ids:
                    self._update_observation(intersection_id, payload)
            elif topic_parts[1] == 'region':
                self._process_controller_command(payload)
                
        except Exception as e:
            logger.error(f"Region {self.region_id}: Error processing message: {e}")
            
    def _update_observation(self, intersection_id: str, state: Dict):
        """Update observation for an intersection."""
        self.current_observations[intersection_id] = {
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
        
    def _process_controller_command(self, command: Dict):
        """Process command from controller agent."""
        cmd_type = command.get('type')
        
        if cmd_type == 'UPDATE_WEIGHTS':
            # Update local agent weights from controller
            weights = command.get('weights')
            if weights:
                self._update_weights(weights)
        elif cmd_type == 'GET_Q_VALUES':
            # Controller requesting Q-values
            self._send_q_values()
            
    def _build_observation(self, intersection_id: str) -> torch.Tensor:
        """Build observation tensor for an intersection."""
        obs_vec = []
        
        if intersection_id in self.current_observations:
            obs = self.current_observations[intersection_id]
            
            # Queue lengths (normalized)
            queues = list(obs.get('queue_lengths', {}).values())
            while len(queues) < 8:
                queues.append(0)
            obs_vec.extend([q / 50.0 for q in queues[:8]])
            
            # Phase one-hot
            phase = obs.get('current_phase', 0)
            phase_onehot = [0] * self.action_dim
            if 0 <= phase < self.action_dim:
                phase_onehot[phase] = 1
            obs_vec.extend(phase_onehot)
            
            # Vehicle counts (normalized)
            counts = obs.get('counts', {})
            obs_vec.extend([
                counts.get('car', 0) / 50.0,
                counts.get('bike', 0) / 20.0,
                counts.get('hmv', 0) / 10.0,
                counts.get('auto', 0) / 30.0
            ])
        else:
            obs_vec = [0] * self.obs_dim
            
        # Ensure correct dimension
        obs_vec = obs_vec[:self.obs_dim]
        while len(obs_vec) < self.obs_dim:
            obs_vec.append(0)
            
        return torch.FloatTensor([[[obs_vec]]]).to(self.device)  # [1, 1, obs_dim]
        
    def compute_local_q_values(self, intersection_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Q-values for an intersection."""
        obs = self._build_observation(intersection_id)
        agent = self.agents[intersection_id]
        hidden = self.hidden_states[intersection_id]
        
        with torch.no_grad():
            q_values, new_hidden = agent(obs.squeeze(0), hidden)
            
        self.hidden_states[intersection_id] = new_hidden
        
        return q_values.squeeze(), new_hidden
        
    def compute_reward(self, intersection_id: str, action: int) -> float:
        """Compute Max Pressure reward for an intersection."""
        if intersection_id not in self.current_observations:
            return 0.0
            
        obs = self.current_observations[intersection_id]
        
        # Incoming queues
        incoming_queues = obs.get('queue_lengths', {})
        
        # Outgoing capacity (simplified - assume fixed capacity minus current)
        # In practice, this would come from neighboring intersections
        max_capacity = 50
        outgoing_capacity = {
            f"lane_{i}": max(0, max_capacity - incoming_queues.get(f"lane_{i}", 0))
            for i in range(4)
        }
        
        return self.reward_fn.compute(
            intersection_id=intersection_id,
            incoming_queues=incoming_queues,
            outgoing_capacity=outgoing_capacity,
            current_action=action
        )
        
    def _send_q_values(self):
        """Send Q-values to controller for QMIX mixing."""
        q_values_dict = {}
        
        for intersection_id in self.intersection_ids:
            q_vals, _ = self.compute_local_q_values(intersection_id)
            q_values_dict[intersection_id] = q_vals.cpu().numpy().tolist()
            
        message = {
            'region_id': self.region_id,
            'q_values': q_values_dict,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.mqtt_client.publish(
            f"traffic/region/{self.region_id}/q_values",
            json.dumps(message)
        )
        
    def _update_weights(self, weights: Dict):
        """Update agent weights from controller."""
        for intersection_id, agent_weights in weights.items():
            if intersection_id in self.agents:
                state_dict = {k: torch.tensor(v) for k, v in agent_weights.items()}
                self.agents[intersection_id].load_state_dict(state_dict)
                logger.info(f"Region {self.region_id}: Updated weights for {intersection_id}")
                
    def publish_region_state(self):
        """Publish aggregated region state to controller."""
        # Aggregate observations
        total_queue = 0
        total_vehicles = 0
        ambulance_detected = False
        
        for intersection_id, obs in self.current_observations.items():
            total_queue += sum(obs.get('queue_lengths', {}).values())
            counts = obs.get('counts', {})
            total_vehicles += sum(counts.values())
            if obs.get('ambulance_detected', False):
                ambulance_detected = True
                
        state = {
            'region_id': self.region_id,
            'total_queue_length': total_queue,
            'total_vehicles': total_vehicles,
            'ambulance_detected': ambulance_detected,
            'intersection_count': len(self.current_observations),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.mqtt_client.publish(
            f"traffic/region/{self.region_id}/state",
            json.dumps(state)
        )
        
    def step(self) -> Dict[str, int]:
        """
        Execute one step: compute Q-values, select actions, publish state.
        
        Returns:
            actions: Dictionary of intersection_id -> action
        """
        actions = {}
        
        for intersection_id in self.intersection_ids:
            q_values, _ = self.compute_local_q_values(intersection_id)
            action = q_values.argmax().item()
            actions[intersection_id] = action
            
            # Compute local reward for logging
            reward = self.compute_reward(intersection_id, action)
            logger.debug(f"Region {self.region_id}/{intersection_id}: action={action}, reward={reward:.3f}")
            
        # Publish region state
        self.publish_region_state()
        
        # Send Q-values to controller
        self._send_q_values()
        
        self.last_actions = actions
        return actions
        
    def run(self, step_interval: float = 5.0):
        """Main loop for regional agent."""
        import time
        
        logger.info(f"Region {self.region_id}: Starting main loop")
        
        try:
            while True:
                self.step()
                time.sleep(step_interval)
                
        except KeyboardInterrupt:
            logger.info(f"Region {self.region_id}: Stopped")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Cleanup resources."""
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()


def main():
    """Entry point for regional agent."""
    # Configuration from environment
    region_id = int(os.getenv('REGION_ID', 0))
    mqtt_broker = os.getenv('MQTT_BROKER', 'localhost')
    mqtt_port = int(os.getenv('MQTT_PORT', 1883))
    
    # Intersection IDs for this region (would come from configuration)
    agents_per_region = int(os.getenv('NUM_AGENTS_PER_REGION', 4))
    base_id = region_id * agents_per_region + 1
    intersection_ids = [str(i) for i in range(base_id, base_id + agents_per_region)]
    
    # Create agent
    agent = RegionalAgent(
        region_id=region_id,
        intersection_ids=intersection_ids,
        mqtt_broker=mqtt_broker,
        mqtt_port=mqtt_port
    )
    
    # Connect and run
    agent.connect_mqtt()
    agent.run(step_interval=5.0)


if __name__ == '__main__':
    main()
