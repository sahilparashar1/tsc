-- Initialize Traffic Control Database Schema
-- PostgreSQL 16+

-- Intersection Logs Table
CREATE TABLE IF NOT EXISTS intersection_logs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    intersection_id VARCHAR(50) NOT NULL,
    region_id INTEGER NOT NULL,
    signal_phase VARCHAR(20) NOT NULL,
    count_car INTEGER NOT NULL DEFAULT 0,
    count_bike INTEGER NOT NULL DEFAULT 0,
    count_hmv INTEGER NOT NULL DEFAULT 0,
    count_auto INTEGER NOT NULL DEFAULT 0,
    count_ambulance INTEGER NOT NULL DEFAULT 0,
    ambulance_detected BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for efficient querying
CREATE INDEX IF NOT EXISTS idx_intersection_logs_timestamp 
    ON intersection_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_intersection_logs_intersection 
    ON intersection_logs(intersection_id);
CREATE INDEX IF NOT EXISTS idx_intersection_logs_region 
    ON intersection_logs(region_id);

-- Training Episodes Table (for QMIX replay buffer)
CREATE TABLE IF NOT EXISTS training_episodes (
    id BIGSERIAL PRIMARY KEY,
    episode_id UUID NOT NULL,
    step_number INTEGER NOT NULL,
    global_state JSONB NOT NULL,
    actions JSONB NOT NULL,
    rewards JSONB NOT NULL,
    next_global_state JSONB NOT NULL,
    done BOOLEAN NOT NULL DEFAULT FALSE,
    hidden_states BYTEA,  -- Serialized GRU hidden states
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_episodes_episode 
    ON training_episodes(episode_id);

-- Signal Commands Table
CREATE TABLE IF NOT EXISTS signal_commands (
    id BIGSERIAL PRIMARY KEY,
    intersection_id VARCHAR(50) NOT NULL,
    commanded_phase VARCHAR(20) NOT NULL,
    duration_seconds INTEGER NOT NULL,
    is_override BOOLEAN NOT NULL DEFAULT FALSE,
    override_reason VARCHAR(100),
    issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Performance Metrics Table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    region_id INTEGER NOT NULL,
    average_wait_time FLOAT,
    total_throughput INTEGER,
    queue_lengths JSONB,
    reward_value FLOAT,
    measured_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_performance_metrics_region 
    ON performance_metrics(region_id, measured_at DESC);
