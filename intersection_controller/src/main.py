"""
Intersection Controller Main Entry Point
Supports --mode real (MQTT from Raspberry Pis) and --mode simulation (TraCI/SUMO)
"""

import os
import sys
import json
import argparse
import logging
import asyncio
from typing import Dict, Optional
from datetime import datetime

import paho.mqtt.client as mqtt

from network_parser import NetworkParser
from aggregator import CameraAggregator, IntersectionState

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntersectionController:
    """
    Central controller for a single intersection.
    Handles both real hardware (MQTT from Raspberry Pis) and simulation (TraCI) modes.
    """
    
    def __init__(
        self,
        intersection_id: str,
        network_file: str,
        mode: str = 'simulation',
        mqtt_broker: str = 'localhost',
        mqtt_port: int = 1883,
        sumo_port: int = 8813
    ):
        """
        Initialize intersection controller.
        
        Args:
            intersection_id: ID of this intersection
            network_file: Path to sioux.net.xml
            mode: 'real' or 'simulation'
            mqtt_broker: MQTT broker address
            mqtt_port: MQTT broker port
            sumo_port: SUMO TraCI port
        """
        self.intersection_id = intersection_id
        self.network_file = network_file
        self.mode = mode
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.sumo_port = sumo_port
        
        # Parse network
        self.network_parser = NetworkParser(network_file)
        if not self.network_parser.parse():
            raise RuntimeError(f"Failed to parse network file: {network_file}")
            
        # Get expected camera count
        self.expected_cameras = self.network_parser.get_expected_pi_count(intersection_id)
        logger.info(f"Intersection {intersection_id}: expecting {self.expected_cameras} cameras")
        
        # Get region assignment
        self.region_id = self.network_parser.get_region_for_junction(intersection_id)
        logger.info(f"Intersection {intersection_id}: assigned to region {self.region_id}")
        
        # Camera aggregator
        self.aggregator = CameraAggregator(
            intersection_id=intersection_id,
            expected_cameras=self.expected_cameras
        )
        
        # MQTT client (used in both modes for cloud communication)
        self.mqtt_client = mqtt.Client(client_id=f"intersection_{intersection_id}")
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        
        # State
        self.current_phase: int = 0
        self.is_override_active: bool = False
        self.override_end_time: Optional[datetime] = None
        self.running: bool = False
        
        # TraCI connection (for simulation mode)
        self.traci_conn = None
        
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            logger.info(f"Connected to MQTT broker")
            
            if self.mode == 'real':
                # Subscribe to camera snapshots for this intersection
                for i in range(self.expected_cameras):
                    topic = f"traffic/edge/{self.intersection_id}/cam_{i}/snapshot"
                    client.subscribe(topic)
                    logger.info(f"Subscribed to {topic}")
                    
            # Subscribe to commands from cloud controller
            client.subscribe(f"traffic/intersection/{self.intersection_id}/command")
        else:
            logger.error(f"MQTT connection failed with code {rc}")
            
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            topic_parts = msg.topic.split('/')
            payload = json.loads(msg.payload.decode())
            
            if 'edge' in topic_parts:
                # Camera snapshot
                self._handle_camera_snapshot(payload)
            elif 'command' in topic_parts:
                # Controller command
                self._handle_command(payload)
                
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
            
    def _handle_camera_snapshot(self, data: Dict):
        """Handle incoming camera snapshot."""
        self.aggregator.add_snapshot_json(data)
        
        # Check for ambulance
        if data.get('ambulance_detected', False):
            self._trigger_ambulance_override(data.get('camera_id'))
            
    def _handle_command(self, command: Dict):
        """Handle command from cloud controller."""
        action = command.get('action')
        
        if action == 'SET_PHASE':
            if not self.is_override_active:
                self._set_phase(command.get('phase', 0))
        elif action == 'OVERRIDE':
            self._trigger_override(
                command.get('phase', 0),
                command.get('duration', 30),
                command.get('reason', 'COMMANDED')
            )
            
    def _trigger_ambulance_override(self, camera_id: str):
        """Trigger immediate GREEN for ambulance lane."""
        logger.warning(f"AMBULANCE OVERRIDE: Camera {camera_id}")
        
        # Map camera to phase (simplified mapping)
        # In practice, this would use detailed lane-to-phase mapping
        camera_num = int(camera_id.split('_')[-1]) if '_' in camera_id else 0
        phase = camera_num % 4  # Simplified: camera 0 -> phase 0, etc.
        
        self._trigger_override(phase, 30, 'AMBULANCE_DETECTED')
        
    def _trigger_override(self, phase: int, duration: int, reason: str):
        """Trigger a signal override."""
        self.is_override_active = True
        self.override_end_time = datetime.utcnow()
        
        logger.info(f"Override triggered: phase={phase}, duration={duration}s, reason={reason}")
        
        self._set_phase(phase)
        
        # Publish override event
        self.mqtt_client.publish(
            f"traffic/intersection/{self.intersection_id}/event",
            json.dumps({
                'event': 'OVERRIDE',
                'phase': phase,
                'duration': duration,
                'reason': reason,
                'timestamp': datetime.utcnow().isoformat()
            })
        )
        
    def _set_phase(self, phase: int):
        """Set signal phase."""
        self.current_phase = phase
        self.aggregator.set_current_phase(phase)
        
        if self.mode == 'simulation' and self.traci_conn:
            self._set_sumo_phase(phase)
            
        logger.debug(f"Set phase to {phase}")
        
    def _set_sumo_phase(self, phase: int):
        """Set phase in SUMO simulation."""
        try:
            import traci
            tl_id = self.intersection_id
            if tl_id in traci.trafficlight.getIDList():
                traci.trafficlight.setPhase(tl_id, phase)
        except Exception as e:
            logger.error(f"Failed to set SUMO phase: {e}")
            
    def _read_sumo_state(self) -> Dict:
        """Read vehicle state from SUMO simulation."""
        try:
            import traci
            
            state = {
                'cars': 0,
                'bikes': 0,
                'hmv': 0,
                'auto': 0,
                'ambulance': 0,
                'ambulance_detected': False,
                'queue_lengths': {}
            }
            
            # Get incoming edges for this intersection
            junction_info = self.network_parser.get_junction_info(self.intersection_id)
            if not junction_info:
                return state
                
            for edge_id in junction_info.incoming_edges:
                edge_vehicles = traci.edge.getLastStepVehicleNumber(edge_id)
                waiting_count = traci.edge.getWaitingTime(edge_id)
                
                state['queue_lengths'][edge_id] = edge_vehicles
                
                # Get vehicle types
                for veh_id in traci.edge.getLastStepVehicleIDs(edge_id):
                    vtype = traci.vehicle.getTypeID(veh_id)
                    
                    if 'car' in vtype.lower():
                        state['cars'] += 1
                    elif 'bike' in vtype.lower() or 'motorcycle' in vtype.lower():
                        state['bikes'] += 1
                    elif 'truck' in vtype.lower() or 'bus' in vtype.lower():
                        state['hmv'] += 1
                    elif 'auto' in vtype.lower() or 'rickshaw' in vtype.lower():
                        state['auto'] += 1
                    elif 'ambulance' in vtype.lower() or 'emergency' in vtype.lower():
                        state['ambulance'] += 1
                        state['ambulance_detected'] = True
                    else:
                        state['cars'] += 1  # Default to car
                        
            return state
            
        except Exception as e:
            logger.error(f"Failed to read SUMO state: {e}")
            return {}
            
    def _aggregate_and_publish(self):
        """Aggregate data and publish to cloud."""
        if self.mode == 'simulation':
            # Read from SUMO
            sumo_state = self._read_sumo_state()
            
            # Create synthetic snapshot
            from aggregator import CameraSnapshot
            for i, (edge_id, queue) in enumerate(sumo_state.get('queue_lengths', {}).items()):
                snapshot = CameraSnapshot(
                    camera_id=f"cam_{i}",
                    timestamp=datetime.utcnow(),
                    cars=sumo_state.get('cars', 0) // max(1, len(sumo_state.get('queue_lengths', {1:1}))),
                    bikes=sumo_state.get('bikes', 0) // max(1, len(sumo_state.get('queue_lengths', {1:1}))),
                    hmv=sumo_state.get('hmv', 0) // max(1, len(sumo_state.get('queue_lengths', {1:1}))),
                    auto=sumo_state.get('auto', 0) // max(1, len(sumo_state.get('queue_lengths', {1:1}))),
                    ambulance=sumo_state.get('ambulance', 0),
                    ambulance_detected=sumo_state.get('ambulance_detected', False)
                )
                self.aggregator.add_snapshot(snapshot)
                
            # Check for ambulance override
            if sumo_state.get('ambulance_detected', False):
                self._trigger_ambulance_override('sumo_detection')
                
        # Aggregate all camera data
        state = self.aggregator.aggregate()
        
        # Publish aggregated state to cloud
        self.mqtt_client.publish(
            f"traffic/intersection/{self.intersection_id}/aggregated",
            json.dumps(state.to_json())
        )
        
    def connect(self):
        """Connect to all services."""
        # Connect MQTT
        try:
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
            logger.info("MQTT connected")
        except Exception as e:
            logger.error(f"Failed to connect MQTT: {e}")
            if self.mode == 'real':
                raise
                
        # Connect TraCI for simulation
        if self.mode == 'simulation':
            self._connect_traci()
            
    def _connect_traci(self):
        """Connect to SUMO via TraCI."""
        try:
            import traci
            
            # Check if already connected
            try:
                traci.getVersion()
                logger.info("TraCI already connected")
                self.traci_conn = traci
                return
            except:
                pass
                
            # Connect to running SUMO instance
            traci.init(port=self.sumo_port)
            self.traci_conn = traci
            logger.info(f"TraCI connected on port {self.sumo_port}")
            
        except Exception as e:
            logger.warning(f"TraCI connection failed: {e}")
            logger.info("Running without SUMO simulation")
            
    def run(self, step_interval: float = 5.0):
        """
        Main control loop.
        
        Args:
            step_interval: Seconds between control steps
        """
        self.running = True
        logger.info(f"Starting intersection controller in {self.mode} mode")
        
        try:
            while self.running:
                # Check override timeout
                if self.is_override_active and self.override_end_time:
                    if datetime.utcnow() > self.override_end_time:
                        self.is_override_active = False
                        logger.info("Override ended")
                        
                # Aggregate and publish
                self._aggregate_and_publish()
                
                # Step SUMO simulation if connected
                if self.mode == 'simulation' and self.traci_conn:
                    try:
                        self.traci_conn.simulationStep()
                        # Log progress every 10 simulation seconds (assuming 1s step in SUMO)
                        ct = self.traci_conn.simulation.getTime()
                        if int(ct) > 0 and int(ct) % 10 == 0:
                            logger.info(f"Simulation time: {ct}")
                    except Exception as e:
                        logger.error(f"SUMO step failed: {e}")
                        
                asyncio.get_event_loop().run_until_complete(asyncio.sleep(step_interval))
                
        except KeyboardInterrupt:
            logger.info("Controller stopped by user")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Cleanup resources."""
        self.running = False
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        
        if self.traci_conn:
            try:
                self.traci_conn.close()
            except:
                pass
                
        logger.info("Cleanup complete")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='Intersection Controller - Real or Simulation Mode'
    )
    parser.add_argument(
        '--mode',
        choices=['real', 'simulation'],
        default='simulation',
        help='Operating mode: real (MQTT from Pis) or simulation (TraCI)'
    )
    parser.add_argument(
        '--intersection_id',
        type=str,
        required=True,
        help='Intersection/junction ID to control'
    )
    parser.add_argument(
        '--network',
        type=str,
        default='sioux.net.xml',
        help='Path to SUMO network file'
    )
    parser.add_argument(
        '--mqtt_broker',
        type=str,
        default=os.getenv('MQTT_BROKER', 'localhost'),
        help='MQTT broker address'
    )
    parser.add_argument(
        '--mqtt_port',
        type=int,
        default=int(os.getenv('MQTT_PORT', 1883)),
        help='MQTT broker port'
    )
    parser.add_argument(
        '--sumo_port',
        type=int,
        default=8813,
        help='SUMO TraCI port'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=0.5,
        help='Control step interval in seconds'
    )
    
    args = parser.parse_args()
    
    # Create and run controller
    controller = IntersectionController(
        intersection_id=args.intersection_id,
        network_file=args.network,
        mode=args.mode,
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port,
        sumo_port=args.sumo_port
    )
    
    controller.connect()
    controller.run(step_interval=args.interval)


if __name__ == '__main__':
    main()
