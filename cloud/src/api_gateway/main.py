"""
API Gateway - FastAPI Backend for Traffic Dashboard
Real-time state cache with MQTT subscriber and REST endpoints
"""

import os
import json
import asyncio
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import paho.mqtt.client as mqtt

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================

class IntersectionStatus(BaseModel):
    """Status of a single intersection."""
    intersection_id: str
    queue_lengths: Dict[str, int] = {}
    active_phase: int = 0
    count_car: int = 0
    count_bike: int = 0
    count_hmv: int = 0
    count_auto: int = 0
    count_ambulance: int = 0
    ambulance_detected: bool = False
    last_update: Optional[str] = None


class NetworkStatusResponse(BaseModel):
    """Response for network status endpoint."""
    timestamp: str
    total_intersections: int
    active_intersections: int
    ambulance_alerts: List[str]
    intersections: Dict[str, IntersectionStatus]


class EmergencyTriggerRequest(BaseModel):
    """Request body for emergency trigger."""
    intersection_id: str
    reason: str = "MANUAL_OVERRIDE"
    duration: int = 30


class EmergencyTriggerResponse(BaseModel):
    """Response for emergency trigger."""
    success: bool
    message: str
    intersection_id: str
    timestamp: str


class HistoryLogEntry(BaseModel):
    """Single history log entry."""
    timestamp: str
    intersection_id: str
    region_id: int
    signal_phase: str
    count_car: int
    count_bike: int
    count_hmv: int
    count_auto: int
    count_ambulance: int


class HistoryResponse(BaseModel):
    """Response for history logs endpoint."""
    total_records: int
    records: List[HistoryLogEntry]


class StatsSummary(BaseModel):
    """Wait time statistics summary."""
    intersection_id: str
    avg_queue_length: float
    max_queue_length: int
    total_vehicles: int
    ambulance_events: int


# ============================================================================
# Real-Time State Cache
# ============================================================================

class StateCache:
    """
    In-memory cache for real-time intersection states.
    Updated by MQTT subscriber background thread.
    """
    
    def __init__(self, stale_threshold_seconds: int = 30):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._stale_threshold = timedelta(seconds=stale_threshold_seconds)
        
    def update(self, intersection_id: str, data: Dict[str, Any]):
        """Update cache for an intersection."""
        with self._lock:
            self._cache[intersection_id] = {
                **data,
                'last_update': datetime.utcnow().isoformat()
            }
            
    def get(self, intersection_id: str) -> Optional[Dict[str, Any]]:
        """Get cached state for an intersection."""
        with self._lock:
            return self._cache.get(intersection_id)
            
    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Get all cached states."""
        with self._lock:
            return dict(self._cache)
            
    def get_ambulance_alerts(self) -> List[str]:
        """Get list of intersections with active ambulance alerts."""
        alerts = []
        with self._lock:
            for intersection_id, data in self._cache.items():
                if data.get('ambulance_detected', False):
                    alerts.append(intersection_id)
        return alerts
        
    def get_active_count(self) -> int:
        """Get count of recently active intersections."""
        now = datetime.utcnow()
        count = 0
        with self._lock:
            for data in self._cache.values():
                last_update = data.get('last_update')
                if last_update:
                    try:
                        update_time = datetime.fromisoformat(last_update)
                        if now - update_time < self._stale_threshold:
                            count += 1
                    except:
                        pass
        return count


# ============================================================================
# MQTT Subscriber
# ============================================================================

class MQTTSubscriber:
    """
    Background MQTT subscriber that updates the state cache.
    """
    
    def __init__(
        self,
        cache: StateCache,
        broker: str = 'localhost',
        port: int = 1883
    ):
        self.cache = cache
        self.broker = broker
        self.port = port
        self.client: Optional[mqtt.Client] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            logger.info(f"MQTT connected to {self.broker}:{self.port}")
            # Subscribe to all traffic topics
            client.subscribe("traffic/#")
        else:
            logger.error(f"MQTT connection failed: {rc}")
            
    def _on_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            topic_parts = msg.topic.split('/')
            payload = json.loads(msg.payload.decode())
            
            # Handle intersection aggregated data
            if 'intersection' in topic_parts and 'aggregated' in topic_parts:
                intersection_id = topic_parts[2]
                self.cache.update(intersection_id, payload)
                
            # Handle edge device snapshots
            elif 'edge' in topic_parts and 'snapshot' in topic_parts:
                intersection_id = topic_parts[2]
                # Merge with existing cache entry
                existing = self.cache.get(intersection_id) or {}
                existing.update({
                    'count_car': payload.get('cars', 0),
                    'count_bike': payload.get('bikes', 0),
                    'count_hmv': payload.get('hmv', 0),
                    'count_auto': payload.get('auto', 0),
                    'count_ambulance': payload.get('ambulance', 0),
                    'ambulance_detected': payload.get('ambulance_detected', False)
                })
                self.cache.update(intersection_id, existing)
                
            # Handle region state updates
            elif 'region' in topic_parts and 'state' in topic_parts:
                # Region-level updates could be handled here
                pass
                
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
            
    def start(self):
        """Start the MQTT subscriber in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        
    def _run(self):
        """Background thread loop."""
        self.client = mqtt.Client(client_id="api_gateway_subscriber")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_forever()
        except Exception as e:
            logger.error(f"MQTT subscriber error: {e}")
            
    def stop(self):
        """Stop the MQTT subscriber."""
        self._running = False
        if self.client:
            self.client.disconnect()
            
    def publish_emergency(self, intersection_id: str, reason: str, duration: int) -> bool:
        """Publish emergency override command."""
        if not self.client:
            return False
            
        try:
            command = {
                'intersection_id': intersection_id,
                'action': 'EMERGENCY_OVERRIDE',
                'phase': 0,  # Green for emergency lane
                'duration': duration,
                'reason': reason,
                'priority': 'HIGH',
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Publish to emergency topic
            self.client.publish(
                f"traffic/emergency/{intersection_id}",
                json.dumps(command),
                qos=1  # At least once delivery
            )
            
            # Also publish to intersection command topic
            self.client.publish(
                f"traffic/intersection/{intersection_id}/command",
                json.dumps(command),
                qos=1
            )
            
            logger.info(f"Emergency override published for {intersection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish emergency: {e}")
            return False


# ============================================================================
# Database Connection
# ============================================================================

class DatabaseConnection:
    """SQLAlchemy database connection for historical queries."""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        
    def connect(self):
        """Establish database connection."""
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5432')
        user = os.getenv('POSTGRES_USER', 'traffic_admin')
        password = os.getenv('POSTGRES_PASSWORD', 'traffic_secure_pwd')
        database = os.getenv('POSTGRES_DB', 'traffic_control')
        
        connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        
        try:
            self.engine = create_engine(connection_string, pool_pre_ping=True)
            self.SessionLocal = sessionmaker(bind=self.engine)
            logger.info(f"Database connected: {host}:{port}/{database}")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            
    def get_history_logs(
        self,
        limit: int = 100,
        offset: int = 0,
        intersection_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """Query historical intersection logs."""
        if not self.engine:
            return []
            
        try:
            query = """
                SELECT timestamp, intersection_id, region_id, signal_phase,
                       count_car, count_bike, count_hmv, count_auto, count_ambulance
                FROM intersection_logs
                WHERE 1=1
            """
            params = {}
            
            if intersection_id:
                query += " AND intersection_id = :intersection_id"
                params['intersection_id'] = intersection_id
                
            if start_time:
                query += " AND timestamp >= :start_time"
                params['start_time'] = start_time
                
            if end_time:
                query += " AND timestamp <= :end_time"
                params['end_time'] = end_time
                
            query += " ORDER BY timestamp DESC LIMIT :limit OFFSET :offset"
            params['limit'] = limit
            params['offset'] = offset
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                rows = result.fetchall()
                
            return [
                {
                    'timestamp': row[0].isoformat() if row[0] else None,
                    'intersection_id': row[1],
                    'region_id': row[2],
                    'signal_phase': row[3],
                    'count_car': row[4],
                    'count_bike': row[5],
                    'count_hmv': row[6],
                    'count_auto': row[7],
                    'count_ambulance': row[8]
                }
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return []
            
    def get_stats_summary(
        self,
        hours: int = 24,
        intersection_id: Optional[str] = None
    ) -> List[Dict]:
        """Get aggregated statistics for intersections."""
        if not self.engine:
            return []
            
        try:
            query = """
                SELECT 
                    intersection_id,
                    AVG(count_car + count_bike + count_hmv + count_auto) as avg_queue,
                    MAX(count_car + count_bike + count_hmv + count_auto) as max_queue,
                    SUM(count_car + count_bike + count_hmv + count_auto) as total_vehicles,
                    SUM(CASE WHEN count_ambulance > 0 THEN 1 ELSE 0 END) as ambulance_events
                FROM intersection_logs
                WHERE timestamp >= NOW() - INTERVAL ':hours hours'
            """
            params = {'hours': hours}
            
            if intersection_id:
                query += " AND intersection_id = :intersection_id"
                params['intersection_id'] = intersection_id
                
            query += " GROUP BY intersection_id ORDER BY intersection_id"
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                rows = result.fetchall()
                
            return [
                {
                    'intersection_id': row[0],
                    'avg_queue_length': float(row[1]) if row[1] else 0,
                    'max_queue_length': int(row[2]) if row[2] else 0,
                    'total_vehicles': int(row[3]) if row[3] else 0,
                    'ambulance_events': int(row[4]) if row[4] else 0
                }
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"Stats query failed: {e}")
            return []


# ============================================================================
# FastAPI Application
# ============================================================================

# Global instances
state_cache = StateCache()
mqtt_subscriber: Optional[MQTTSubscriber] = None
db_connection = DatabaseConnection()

# FastAPI app
app = FastAPI(
    title="Traffic Control API Gateway",
    description="REST API for Traffic Dashboard - Real-time state and historical data",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup."""
    global mqtt_subscriber
    
    # Connect to database
    db_connection.connect()
    
    # Start MQTT subscriber
    mqtt_broker = os.getenv('MQTT_BROKER', 'localhost')
    mqtt_port = int(os.getenv('MQTT_PORT', '1883'))
    
    mqtt_subscriber = MQTTSubscriber(
        cache=state_cache,
        broker=mqtt_broker,
        port=mqtt_port
    )
    mqtt_subscriber.start()
    
    logger.info("API Gateway started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    if mqtt_subscriber:
        mqtt_subscriber.stop()
    logger.info("API Gateway stopped")


# ============================================================================
# REST Endpoints
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Traffic Control API Gateway",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/network/status", response_model=NetworkStatusResponse, tags=["Network"])
async def get_network_status():
    """
    Get current cached state of all intersections.
    Returns queue lengths, active phases, and ambulance alerts.
    """
    all_states = state_cache.get_all()
    ambulance_alerts = state_cache.get_ambulance_alerts()
    
    intersections = {}
    for intersection_id, data in all_states.items():
        intersections[intersection_id] = IntersectionStatus(
            intersection_id=intersection_id,
            queue_lengths=data.get('queue_lengths', {}),
            active_phase=data.get('current_phase', data.get('active_phase', 0)),
            count_car=data.get('count_car', 0),
            count_bike=data.get('count_bike', 0),
            count_hmv=data.get('count_hmv', 0),
            count_auto=data.get('count_auto', 0),
            count_ambulance=data.get('count_ambulance', 0),
            ambulance_detected=data.get('ambulance_detected', False),
            last_update=data.get('last_update')
        )
    
    return NetworkStatusResponse(
        timestamp=datetime.utcnow().isoformat(),
        total_intersections=24,  # Sioux Falls network
        active_intersections=state_cache.get_active_count(),
        ambulance_alerts=ambulance_alerts,
        intersections=intersections
    )


@app.get("/network/status/{intersection_id}", response_model=IntersectionStatus, tags=["Network"])
async def get_intersection_status(intersection_id: str):
    """Get status for a specific intersection."""
    data = state_cache.get(intersection_id)
    
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"Intersection {intersection_id} not found in cache"
        )
    
    return IntersectionStatus(
        intersection_id=intersection_id,
        queue_lengths=data.get('queue_lengths', {}),
        active_phase=data.get('current_phase', data.get('active_phase', 0)),
        count_car=data.get('count_car', 0),
        count_bike=data.get('count_bike', 0),
        count_hmv=data.get('count_hmv', 0),
        count_auto=data.get('count_auto', 0),
        count_ambulance=data.get('count_ambulance', 0),
        ambulance_detected=data.get('ambulance_detected', False),
        last_update=data.get('last_update')
    )


@app.post("/emergency/trigger", response_model=EmergencyTriggerResponse, tags=["Emergency"])
async def trigger_emergency(request: EmergencyTriggerRequest):
    """
    Trigger emergency override for an intersection.
    Immediately publishes high-priority message to force Green Light.
    """
    if not mqtt_subscriber:
        raise HTTPException(
            status_code=503,
            detail="MQTT connection not available"
        )
    
    success = mqtt_subscriber.publish_emergency(
        intersection_id=request.intersection_id,
        reason=request.reason,
        duration=request.duration
    )
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to publish emergency command"
        )
    
    return EmergencyTriggerResponse(
        success=True,
        message=f"Emergency override triggered for {request.intersection_id}",
        intersection_id=request.intersection_id,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/logs/history", response_model=HistoryResponse, tags=["History"])
async def get_history_logs(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    intersection_id: Optional[str] = Query(default=None),
    hours: Optional[int] = Query(default=24, ge=1, le=720)
):
    """
    Query historical intersection logs from PostgreSQL.
    Returns wait-time statistics for graphs.
    """
    start_time = datetime.utcnow() - timedelta(hours=hours) if hours else None
    
    records = db_connection.get_history_logs(
        limit=limit,
        offset=offset,
        intersection_id=intersection_id,
        start_time=start_time
    )
    
    history_entries = [
        HistoryLogEntry(
            timestamp=r['timestamp'],
            intersection_id=r['intersection_id'],
            region_id=r['region_id'],
            signal_phase=r['signal_phase'],
            count_car=r['count_car'],
            count_bike=r['count_bike'],
            count_hmv=r['count_hmv'],
            count_auto=r['count_auto'],
            count_ambulance=r['count_ambulance']
        )
        for r in records
    ]
    
    return HistoryResponse(
        total_records=len(history_entries),
        records=history_entries
    )


@app.get("/logs/stats", response_model=List[StatsSummary], tags=["History"])
async def get_stats_summary(
    hours: int = Query(default=24, ge=1, le=720),
    intersection_id: Optional[str] = Query(default=None)
):
    """Get aggregated statistics for dashboard graphs."""
    stats = db_connection.get_stats_summary(
        hours=hours,
        intersection_id=intersection_id
    )
    
    return [
        StatsSummary(
            intersection_id=s['intersection_id'],
            avg_queue_length=s['avg_queue_length'],
            max_queue_length=s['max_queue_length'],
            total_vehicles=s['total_vehicles'],
            ambulance_events=s['ambulance_events']
        )
        for s in stats
    ]


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the API Gateway server."""
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', '8000'))
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv('DEBUG', 'false').lower() == 'true'
    )


if __name__ == '__main__':
    main()
