"""
Aggregator - Combines Camera Feeds into Intersection State
Handles data from 3-4 Raspberry Pi cameras per intersection.
"""

import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CameraSnapshot:
    """Single snapshot from a camera."""
    camera_id: str
    timestamp: datetime
    cars: int = 0
    bikes: int = 0
    hmv: int = 0
    auto: int = 0
    ambulance: int = 0
    ambulance_detected: bool = False
    
    @classmethod
    def from_json(cls, data: Dict) -> 'CameraSnapshot':
        """Create snapshot from JSON payload."""
        return cls(
            camera_id=data.get('camera_id', ''),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.utcnow().isoformat())),
            cars=data.get('cars', 0),
            bikes=data.get('bikes', 0),
            hmv=data.get('hmv', 0),
            auto=data.get('auto', 0),
            ambulance=data.get('ambulance', data.get('count_ambulance', 0)),
            ambulance_detected=data.get('ambulance_detected', False)
        )


@dataclass
class IntersectionState:
    """Aggregated state for an intersection."""
    intersection_id: str
    timestamp: datetime
    count_car: int = 0
    count_bike: int = 0
    count_hmv: int = 0
    count_auto: int = 0
    count_ambulance: int = 0
    ambulance_detected: bool = False
    current_phase: int = 0
    queue_lengths: Dict[str, int] = field(default_factory=dict)
    camera_count: int = 0
    ambulance_lane: Optional[str] = None
    
    def to_json(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            'intersection_id': self.intersection_id,
            'timestamp': self.timestamp.isoformat(),
            'count_car': self.count_car,
            'count_bike': self.count_bike,
            'count_hmv': self.count_hmv,
            'count_auto': self.count_auto,
            'count_ambulance': self.count_ambulance,
            'ambulance_detected': self.ambulance_detected,
            'current_phase': self.current_phase,
            'queue_lengths': self.queue_lengths,
            'camera_count': self.camera_count,
            'ambulance_lane': self.ambulance_lane
        }


class CameraAggregator:
    """
    Aggregates data from multiple cameras into a single intersection state.
    Handles timing, missing data, and priority detections.
    """
    
    def __init__(
        self,
        intersection_id: str,
        expected_cameras: int = 4,
        aggregation_window: float = 5.0,  # seconds
        stale_threshold: float = 10.0  # seconds
    ):
        """
        Initialize aggregator.
        
        Args:
            intersection_id: ID of the intersection
            expected_cameras: Expected number of cameras (3 or 4)
            aggregation_window: Time window to collect snapshots
            stale_threshold: Time after which data is considered stale
        """
        self.intersection_id = intersection_id
        self.expected_cameras = expected_cameras
        self.aggregation_window = timedelta(seconds=aggregation_window)
        self.stale_threshold = timedelta(seconds=stale_threshold)
        
        # Camera snapshots storage
        self.snapshots: Dict[str, CameraSnapshot] = {}
        self.lock = threading.Lock()
        
        # Current state
        self.current_state: Optional[IntersectionState] = None
        self.current_phase: int = 0
        
        # Statistics
        self.total_snapshots_received = 0
        self.last_aggregation_time: Optional[datetime] = None
        
    def add_snapshot(self, snapshot: CameraSnapshot):
        """
        Add a camera snapshot.
        
        Args:
            snapshot: Camera snapshot data
        """
        with self.lock:
            self.snapshots[snapshot.camera_id] = snapshot
            self.total_snapshots_received += 1
            
            # Check for immediate ambulance detection
            if snapshot.ambulance_detected:
                logger.warning(f"Ambulance detected on camera {snapshot.camera_id}")
                
    def add_snapshot_json(self, data: Dict):
        """Add snapshot from JSON data."""
        snapshot = CameraSnapshot.from_json(data)
        self.add_snapshot(snapshot)
        
    def set_current_phase(self, phase: int):
        """Update current signal phase."""
        self.current_phase = phase
        
    def _is_snapshot_valid(self, snapshot: CameraSnapshot, now: datetime) -> bool:
        """Check if snapshot is within valid time window."""
        age = now - snapshot.timestamp
        return age <= self.stale_threshold
        
    def aggregate(self) -> IntersectionState:
        """
        Aggregate all valid camera snapshots into intersection state.
        
        Returns:
            Aggregated intersection state
        """
        now = datetime.utcnow()
        
        with self.lock:
            # Filter valid snapshots
            valid_snapshots = {
                cam_id: snap for cam_id, snap in self.snapshots.items()
                if self._is_snapshot_valid(snap, now)
            }
            
        if not valid_snapshots:
            # Only warn every 60 seconds or if debug is enabled
            if self.total_snapshots_received % 60 == 0:
                logger.debug(f"No valid snapshots for {self.intersection_id} (waiting for vehicles)")
            return IntersectionState(
                    intersection_id=self.intersection_id,
                    timestamp=now,
                    current_phase=self.current_phase,
                    camera_count=0
                )
                
            # Aggregate counts
            total_cars = 0
            total_bikes = 0
            total_hmv = 0
            total_auto = 0
            total_ambulance = 0
            ambulance_detected = False
            ambulance_lane = None
            queue_lengths = {}
            
            for cam_id, snapshot in valid_snapshots.items():
                total_cars += snapshot.cars
                total_bikes += snapshot.bikes
                total_hmv += snapshot.hmv
                total_auto += snapshot.auto
                total_ambulance += snapshot.ambulance
                
                if snapshot.ambulance_detected or snapshot.ambulance > 0:
                    ambulance_detected = True
                    ambulance_lane = cam_id
                    
                # Estimate queue length from vehicle count
                # Simple heuristic: queue = cars + hmv*2 + auto
                queue = snapshot.cars + snapshot.hmv * 2 + snapshot.auto
                queue_lengths[cam_id] = queue
                
            # Create aggregated state
            self.current_state = IntersectionState(
                intersection_id=self.intersection_id,
                timestamp=now,
                count_car=total_cars,
                count_bike=total_bikes,
                count_hmv=total_hmv,
                count_auto=total_auto,
                count_ambulance=total_ambulance,
                ambulance_detected=ambulance_detected,
                current_phase=self.current_phase,
                queue_lengths=queue_lengths,
                camera_count=len(valid_snapshots),
                ambulance_lane=ambulance_lane
            )
            
            self.last_aggregation_time = now
            
            return self.current_state
            
    def get_current_state(self) -> Optional[IntersectionState]:
        """Get last aggregated state."""
        return self.current_state
        
    def has_ambulance(self) -> bool:
        """Check if any camera detected ambulance."""
        with self.lock:
            for snapshot in self.snapshots.values():
                if snapshot.ambulance_detected or snapshot.ambulance > 0:
                    return True
        return False
        
    def get_ambulance_lane(self) -> Optional[str]:
        """Get the camera/lane ID where ambulance was detected."""
        with self.lock:
            for cam_id, snapshot in self.snapshots.items():
                if snapshot.ambulance_detected or snapshot.ambulance > 0:
                    return cam_id
        return None
        
    def is_complete(self) -> bool:
        """Check if we have data from all expected cameras."""
        now = datetime.utcnow()
        with self.lock:
            valid_count = sum(
                1 for snap in self.snapshots.values()
                if self._is_snapshot_valid(snap, now)
            )
            return valid_count >= self.expected_cameras
            
    def get_statistics(self) -> Dict:
        """Get aggregator statistics."""
        now = datetime.utcnow()
        with self.lock:
            valid_count = sum(
                1 for snap in self.snapshots.values()
                if self._is_snapshot_valid(snap, now)
            )
            
        return {
            'intersection_id': self.intersection_id,
            'expected_cameras': self.expected_cameras,
            'active_cameras': valid_count,
            'total_snapshots': self.total_snapshots_received,
            'last_aggregation': self.last_aggregation_time.isoformat() if self.last_aggregation_time else None
        }
        
    def clear_stale(self):
        """Remove stale snapshots."""
        now = datetime.utcnow()
        with self.lock:
            stale_cameras = [
                cam_id for cam_id, snap in self.snapshots.items()
                if not self._is_snapshot_valid(snap, now)
            ]
            for cam_id in stale_cameras:
                del self.snapshots[cam_id]
                logger.debug(f"Removed stale snapshot from {cam_id}")


class MultiIntersectionAggregator:
    """
    Manages aggregators for multiple intersections.
    """
    
    def __init__(self, expected_cameras_map: Dict[str, int] = None):
        """
        Initialize multi-intersection aggregator.
        
        Args:
            expected_cameras_map: Dictionary of intersection_id -> expected camera count
        """
        self.aggregators: Dict[str, CameraAggregator] = {}
        self.expected_cameras_map = expected_cameras_map or {}
        self.lock = threading.Lock()
        
    def get_or_create_aggregator(
        self,
        intersection_id: str,
        expected_cameras: int = 4
    ) -> CameraAggregator:
        """Get existing or create new aggregator for an intersection."""
        with self.lock:
            if intersection_id not in self.aggregators:
                expected = self.expected_cameras_map.get(intersection_id, expected_cameras)
                self.aggregators[intersection_id] = CameraAggregator(
                    intersection_id=intersection_id,
                    expected_cameras=expected
                )
            return self.aggregators[intersection_id]
            
    def add_snapshot(self, intersection_id: str, data: Dict):
        """Add snapshot to appropriate aggregator."""
        aggregator = self.get_or_create_aggregator(intersection_id)
        aggregator.add_snapshot_json(data)
        
    def aggregate_all(self) -> Dict[str, IntersectionState]:
        """Aggregate all intersections."""
        results = {}
        with self.lock:
            for intersection_id, aggregator in self.aggregators.items():
                results[intersection_id] = aggregator.aggregate()
        return results
        
    def check_ambulances(self) -> List[str]:
        """Check all intersections for ambulance detections."""
        ambulance_intersections = []
        with self.lock:
            for intersection_id, aggregator in self.aggregators.items():
                if aggregator.has_ambulance():
                    ambulance_intersections.append(intersection_id)
        return ambulance_intersections
