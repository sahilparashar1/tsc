"""
Database Module - SQLAlchemy Setup for Cloud SQL
Distributed Traffic Control System
"""

import os
from datetime import datetime
from typing import Optional, List
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, Integer, BigInteger, String, Float, Boolean, DateTime, JSON, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID, JSONB, TIMESTAMP
import uuid


Base = declarative_base()


class IntersectionLog(Base):
    """Model for intersection telemetry logs."""
    __tablename__ = 'intersection_logs'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False, default=datetime.utcnow)
    intersection_id = Column(String(50), nullable=False)
    region_id = Column(Integer, nullable=False)
    signal_phase = Column(String(20), nullable=False)
    count_car = Column(Integer, nullable=False, default=0)
    count_bike = Column(Integer, nullable=False, default=0)
    count_hmv = Column(Integer, nullable=False, default=0)
    count_auto = Column(Integer, nullable=False, default=0)
    count_ambulance = Column(Integer, nullable=False, default=0)
    ambulance_detected = Column(Boolean, nullable=False, default=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, default=datetime.utcnow)


class TrainingEpisode(Base):
    """Model for QMIX episode-based replay buffer storage."""
    __tablename__ = 'training_episodes'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    episode_id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    step_number = Column(Integer, nullable=False)
    global_state = Column(JSONB, nullable=False)
    actions = Column(JSONB, nullable=False)
    rewards = Column(JSONB, nullable=False)
    next_global_state = Column(JSONB, nullable=False)
    done = Column(Boolean, nullable=False, default=False)
    hidden_states = Column(LargeBinary)  # Serialized GRU hidden states
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, default=datetime.utcnow)


class SignalCommand(Base):
    """Model for signal phase commands issued to intersections."""
    __tablename__ = 'signal_commands'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    intersection_id = Column(String(50), nullable=False)
    commanded_phase = Column(String(20), nullable=False)
    duration_seconds = Column(Integer, nullable=False)
    is_override = Column(Boolean, nullable=False, default=False)
    override_reason = Column(String(100))
    issued_at = Column(TIMESTAMP(timezone=True), nullable=False, default=datetime.utcnow)


class PerformanceMetric(Base):
    """Model for regional performance metrics."""
    __tablename__ = 'performance_metrics'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    region_id = Column(Integer, nullable=False)
    average_wait_time = Column(Float)
    total_throughput = Column(Integer)
    queue_lengths = Column(JSONB)
    reward_value = Column(Float)
    measured_at = Column(TIMESTAMP(timezone=True), nullable=False, default=datetime.utcnow)


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        user: str = None,
        password: str = None,
        database: str = None
    ):
        self.host = host or os.getenv('POSTGRES_HOST', 'localhost')
        self.port = port or int(os.getenv('POSTGRES_PORT', 5432))
        self.user = user or os.getenv('POSTGRES_USER', 'traffic_admin')
        self.password = password or os.getenv('POSTGRES_PASSWORD', 'traffic_secure_pwd')
        self.database = database or os.getenv('POSTGRES_DB', 'traffic_control')
        
        self.engine = None
        self.SessionLocal = None
        
    def connect(self):
        """Establish database connection."""
        connection_string = (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )
        self.engine = create_engine(connection_string, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        
    def create_tables(self):
        """Create all tables if they don't exist."""
        if self.engine:
            Base.metadata.create_all(bind=self.engine)
            
    @contextmanager
    def get_session(self) -> Session:
        """Context manager for database sessions."""
        if not self.SessionLocal:
            self.connect()
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
            
    def log_intersection_state(
        self,
        intersection_id: str,
        region_id: int,
        signal_phase: str,
        counts: dict,
        ambulance_detected: bool = False
    ) -> int:
        """Log intersection state to database."""
        with self.get_session() as session:
            log = IntersectionLog(
                intersection_id=intersection_id,
                region_id=region_id,
                signal_phase=signal_phase,
                count_car=counts.get('car', 0),
                count_bike=counts.get('bike', 0),
                count_hmv=counts.get('hmv', 0),
                count_auto=counts.get('auto', 0),
                count_ambulance=counts.get('ambulance', 0),
                ambulance_detected=ambulance_detected
            )
            session.add(log)
            session.flush()
            return log.id
            
    def store_episode_step(
        self,
        episode_id: uuid.UUID,
        step_number: int,
        global_state: dict,
        actions: dict,
        rewards: dict,
        next_global_state: dict,
        done: bool,
        hidden_states: bytes = None
    ) -> int:
        """Store a training episode step for QMIX replay buffer."""
        with self.get_session() as session:
            episode = TrainingEpisode(
                episode_id=episode_id,
                step_number=step_number,
                global_state=global_state,
                actions=actions,
                rewards=rewards,
                next_global_state=next_global_state,
                done=done,
                hidden_states=hidden_states
            )
            session.add(episode)
            session.flush()
            return episode.id
            
    def get_recent_episodes(self, limit: int = 100) -> List[TrainingEpisode]:
        """Retrieve recent training episodes for replay."""
        with self.get_session() as session:
            episodes = session.query(TrainingEpisode)\
                .order_by(TrainingEpisode.created_at.desc())\
                .limit(limit)\
                .all()
            return episodes
            
    def record_signal_command(
        self,
        intersection_id: str,
        phase: str,
        duration: int,
        is_override: bool = False,
        override_reason: str = None
    ) -> int:
        """Record a signal command issued to an intersection."""
        with self.get_session() as session:
            command = SignalCommand(
                intersection_id=intersection_id,
                commanded_phase=phase,
                duration_seconds=duration,
                is_override=is_override,
                override_reason=override_reason
            )
            session.add(command)
            session.flush()
            return command.id


# Singleton instance for global access
_db_manager: Optional[DatabaseManager] = None


def get_database() -> DatabaseManager:
    """Get or create the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.connect()
    return _db_manager
