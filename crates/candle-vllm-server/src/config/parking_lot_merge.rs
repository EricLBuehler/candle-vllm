//! Configuration merging for parking lot scheduler.
//!
//! This module handles merging parking lot configurations from multiple sources
//! with proper priority: CLI > models.yaml > scheduler-config.json > defaults.

use super::models::{
    LimitsConfig, MailboxBackendConfig, ParkingLotConfig, QueueBackendConfig, WorkerPoolConfig,
};
use super::scheduler::{MailboxConfig, PoolConfig, QueueConfig, SchedulerConfig};
use tracing::info;

/// Merged parking lot configuration from all sources.
#[derive(Debug, Clone)]
pub struct MergedParkingLotConfig {
    pub worker_threads: usize,
    pub max_units: Option<usize>,
    pub max_queue_depth: usize,
    pub timeout_secs: u64,
}

impl Default for MergedParkingLotConfig {
    fn default() -> Self {
        Self {
            worker_threads: num_cpus::get(),
            max_units: None,
            max_queue_depth: 1000,
            timeout_secs: 120,
        }
    }
}

/// Merge parking lot configurations with proper priority.
///
/// Priority (highest to lowest):
/// 1. Per-model overrides (from models.yaml model.parking_lot)
/// 2. Global models.yaml (from models.yaml parking_lot)
/// 3. Scheduler config JSON (from scheduler-config.json)
/// 4. Defaults
pub fn merge_parking_lot_config(
    per_model: Option<&ParkingLotConfig>,
    global_yaml: Option<&ParkingLotConfig>,
    scheduler_json: Option<&SchedulerConfig>,
) -> MergedParkingLotConfig {
    let mut config = MergedParkingLotConfig::default();

    // Layer 1: scheduler-config.json (if present)
    if let Some(scheduler_cfg) = scheduler_json {
        if let Some(pool) = scheduler_cfg.pools.get("default") {
            info!("📋 CONFIG: Applying scheduler-config.json settings");
            config.max_units = Some(pool.max_units);
            config.max_queue_depth = pool.max_queue_depth;
            config.timeout_secs = pool.default_timeout_secs;
        }
    }

    // Layer 2: models.yaml global parking_lot section
    if let Some(yaml_cfg) = global_yaml {
        info!("📋 CONFIG: Applying models.yaml global parking_lot settings");
        config.worker_threads = yaml_cfg.pool.worker_threads;
        config.max_queue_depth = yaml_cfg.pool.max_queue_depth;
        config.timeout_secs = yaml_cfg.pool.timeout_secs;

        if yaml_cfg.limits.max_units.is_some() {
            config.max_units = yaml_cfg.limits.max_units;
        }
    }

    // Layer 3: Per-model overrides (highest priority)
    if let Some(model_cfg) = per_model {
        info!("📋 CONFIG: Applying per-model parking_lot overrides");
        config.worker_threads = model_cfg.pool.worker_threads;
        config.max_queue_depth = model_cfg.pool.max_queue_depth;
        config.timeout_secs = model_cfg.pool.timeout_secs;

        if model_cfg.limits.max_units.is_some() {
            config.max_units = model_cfg.limits.max_units;
        }
    }

    info!(
        "✅ CONFIG: Final merged config - workers={}, max_units={:?}, queue_depth={}, timeout={}s",
        config.worker_threads, config.max_units, config.max_queue_depth, config.timeout_secs
    );

    config
}

/// Convert models.yaml ParkingLotConfig to scheduler PoolConfig format.
pub fn parking_lot_to_pool_config(
    parking_lot: &ParkingLotConfig,
    default_max_units: usize,
) -> PoolConfig {
    PoolConfig {
        max_units: parking_lot.limits.max_units.unwrap_or(default_max_units),
        max_queue_depth: parking_lot.limits.max_queue_depth,
        default_timeout_secs: parking_lot.limits.timeout_secs,
        queue: QueueConfig::InMemory, // Default to in-memory for now
        mailbox: MailboxConfig::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MergedParkingLotConfig::default();
        assert!(config.worker_threads > 0);
        assert_eq!(config.max_queue_depth, 1000);
        assert_eq!(config.timeout_secs, 120);
    }

    #[test]
    fn test_merge_priority() {
        // Create a global config
        let global = ParkingLotConfig {
            pool: WorkerPoolConfig {
                worker_threads: 4,
                max_queue_depth: 500,
                timeout_secs: 60,
            },
            limits: LimitsConfig {
                max_units: Some(2000),
                max_queue_depth: 500,
                timeout_secs: 60,
            },
            queue: QueueBackendConfig::default(),
            mailbox: MailboxBackendConfig::default(),
        };

        // Create a per-model override
        let per_model = ParkingLotConfig {
            pool: WorkerPoolConfig {
                worker_threads: 8, // Override
                ..global.pool.clone()
            },
            limits: LimitsConfig {
                max_units: Some(4000), // Override
                ..global.limits.clone()
            },
            ..global.clone()
        };

        let merged = merge_parking_lot_config(Some(&per_model), Some(&global), None);

        // Per-model overrides should win
        assert_eq!(merged.worker_threads, 8);
        assert_eq!(merged.max_units, Some(4000));
        assert_eq!(merged.max_queue_depth, 500); // From per_model (which inherited from global)
    }
}
