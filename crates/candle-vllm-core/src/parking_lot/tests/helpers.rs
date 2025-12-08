//! Test helpers for parking_lot module tests.

use crate::openai::sampling_params::{EarlyStoppingCondition, SamplingParams};

/// Create default SamplingParams for testing.
pub fn default_sampling_params() -> SamplingParams {
    SamplingParams::new(
        1,
        None,
        0.0,
        0.0,
        None,
        Some(1.0),
        Some(1.0),
        None,
        Some(-1),
        false,
        1.0,
        EarlyStoppingCondition::UnlikelyBetterCandidates,
        None,
        vec![],
        false,
        16,
        None,
        None,
        true,
        None,
    )
    .expect("Failed to create default SamplingParams")
}
