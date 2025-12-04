use crate::models_config::{ModelsState, ModelLifecycleStatus, ModelStatus};
use crate::state::request_queue::{QueuedRequest, RequestQueue};
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct ModelSwitchRequest {
    pub model: String,
    pub requested_at: Instant,
}

#[derive(Debug)]
struct ManagerState {
    pub status: ModelLifecycleStatus,
    pub active_model: Option<String>,
    pub last_error: Option<String>,
    pub switch_requested_at: Option<Instant>,
    pub queue: VecDeque<ModelSwitchRequest>,
}

pub struct ModelManager {
    inner: Mutex<ManagerState>,
    models_state: ModelsState,
    pub max_queue: usize,
    /// Per-model request queues
    request_queues: Arc<Mutex<HashMap<String, Arc<RequestQueue>>>>,
    queue_size: usize,
    request_timeout: Duration,
}

impl ModelManager {
    pub fn new(models_state: ModelsState, max_queue: usize) -> Self {
        Self::with_queue_config(models_state, max_queue, 10, Duration::from_secs(30))
    }

    pub fn with_queue_config(
        models_state: ModelsState,
        max_queue: usize,
        queue_size: usize,
        request_timeout: Duration,
    ) -> Self {
        Self {
            inner: Mutex::new(ManagerState {
                status: ModelLifecycleStatus::Idle,
                active_model: None,
                last_error: None,
                switch_requested_at: None,
                queue: VecDeque::new(),
            }),
            models_state,
            max_queue,
            request_queues: Arc::new(Mutex::new(HashMap::new())),
            queue_size,
            request_timeout,
        }
    }

    /// Get or create the request queue for a model.
    pub fn get_or_create_queue(&self, model: &str) -> Arc<RequestQueue> {
        let mut queues = self.request_queues.lock();
        queues
            .entry(model.to_string())
            .or_insert_with(|| Arc::new(RequestQueue::new(self.queue_size, self.request_timeout)))
            .clone()
    }

    /// Drain requests for a model after it becomes active.
    pub fn drain_model_queue(&self, model: &str) -> Vec<QueuedRequest> {
        let queues = self.request_queues.lock();
        if let Some(queue) = queues.get(model) {
            queue.drain()
        } else {
            Vec::new()
        }
    }

    /// Remove timed-out requests from all queues.
    pub fn remove_timed_out_requests(&self) -> Vec<QueuedRequest> {
        let queues = self.request_queues.lock();
        let mut all_timed_out = Vec::new();
        for queue in queues.values() {
            all_timed_out.extend(queue.remove_timed_out());
        }
        all_timed_out
    }

    /// Get queue length for a model.
    pub fn queue_length(&self, model: &str) -> usize {
        let queues = self.request_queues.lock();
        queues.get(model).map(|q| q.len()).unwrap_or(0)
    }

    /// Enqueue a request to switch to a model.
    /// Returns Ok(true) if enqueued, Ok(false) if already active, Err if queue full or not found.
    pub fn enqueue_switch(&self, model: &str) -> anyhow::Result<SwitchResult> {
        let mut guard = self.inner.lock();
        if let Some(active) = &guard.active_model {
            if active == model {
                return Ok(SwitchResult::AlreadyActive);
            }
        }
        if guard.queue.len() >= self.max_queue {
            anyhow::bail!("queue full");
        }
        if self.models_state.resolve(model).is_none() {
            anyhow::bail!("model not found");
        }
        guard.queue.push_back(ModelSwitchRequest {
            model: model.to_string(),
            requested_at: Instant::now(),
        });
        if guard.status == ModelLifecycleStatus::Idle || guard.status == ModelLifecycleStatus::Ready
        {
            guard.status = ModelLifecycleStatus::Switching;
            guard.switch_requested_at = Some(Instant::now());
        }
        Ok(SwitchResult::Enqueued)
    }

    /// Start the next switch if pending. This is a placeholder hook for future loading logic.
    pub fn begin_next_switch(&self) -> Option<ModelSwitchRequest> {
        let mut guard = self.inner.lock();
        if guard.status == ModelLifecycleStatus::Loading {
            return None;
        }
        if let Some(req) = guard.queue.pop_front() {
            guard.status = ModelLifecycleStatus::Loading;
            guard.switch_requested_at = Some(Instant::now());
            return Some(req);
        }
        None
    }

    /// Mark switch success and drain queued requests.
    pub async fn complete_switch(&self, model: String) -> Vec<QueuedRequest> {
        {
            let mut guard = self.inner.lock();
            guard.active_model = Some(model.clone());
            guard.status = ModelLifecycleStatus::Ready;
            guard.last_error = None;
            guard.switch_requested_at = None;
        }
        self.models_state.set_active(model.clone()).await;

        // Drain queued requests for this model
        self.drain_model_queue(&model)
    }

    /// Mark switch failure.
    pub fn fail_switch(&self, err: String) {
        let mut guard = self.inner.lock();
        guard.last_error = Some(err);
        guard.status = ModelLifecycleStatus::Error;
        guard.switch_requested_at = None;
    }

    pub fn status(&self) -> ModelStatus {
        let guard = self.inner.lock();
        let queues = self.request_queues.lock();
        let total_queued: usize = queues.values().map(|q| q.len()).sum();
        let queue_lengths: std::collections::HashMap<String, usize> = queues
            .iter()
            .map(|(model, queue)| (model.clone(), queue.len()))
            .collect();

        ModelStatus {
            active_model: guard.active_model.clone(),
            status: guard.status.clone(),
            last_error: guard.last_error.clone(),
            in_flight_requests: total_queued,
            switch_requested_at: guard.switch_requested_at.map(|i| {
                i.elapsed()
                    .as_millis()
                    .try_into()
                    .unwrap_or(i64::MAX as u64)
            }),
            queue_lengths,
        }
    }
}

pub enum SwitchResult {
    Enqueued,
    AlreadyActive,
}
