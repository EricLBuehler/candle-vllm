//! Request queue for handling requests during model switching.
//!
//! This module provides queuing functionality to handle requests when
//! the requested model is not currently active.
//!
//! **DEPRECATED**: This module is being replaced by the parking-lot scheduler's
//! built-in queueing. The `LLMEngineV2` uses `prometheus_parking_lot::TaskQueue`
//! for request queueing with configurable backends.
//!
//! This module is retained for backward compatibility during the migration period.

use candle_vllm_core::openai::requests::ChatCompletionRequest;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

/// A request queued for processing after model switch.
pub struct QueuedRequest {
    /// The model this request is for
    pub model: String,
    /// The chat completion request
    pub request: ChatCompletionRequest,
    /// When the request was queued
    pub queued_at: Instant,
    /// Sender for the response
    pub response_tx: Option<oneshot::Sender<candle_vllm_core::openai::responses::ChatResponder>>,
    /// Whether this is a streaming request
    pub is_streaming: bool,
}

impl QueuedRequest {
    /// Create a new queued request.
    pub fn new(
        model: String,
        request: ChatCompletionRequest,
        response_tx: Option<oneshot::Sender<candle_vllm_core::openai::responses::ChatResponder>>,
    ) -> Self {
        Self {
            model,
            is_streaming: request.stream.is_some_and(|s| s),
            request,
            queued_at: Instant::now(),
            response_tx,
        }
    }

    /// Check if this request has timed out.
    pub fn is_timed_out(&self, timeout: Duration) -> bool {
        self.queued_at.elapsed() > timeout
    }
}

/// Queue for managing requests during model switches.
pub struct RequestQueue {
    inner: Mutex<VecDeque<QueuedRequest>>,
    max_size: usize,
    timeout: Duration,
}

impl RequestQueue {
    /// Create a new request queue.
    pub fn new(max_size: usize, timeout: Duration) -> Self {
        Self {
            inner: Mutex::new(VecDeque::new()),
            max_size,
            timeout,
        }
    }

    /// Enqueue a request. Returns Ok(()) if queued, Err if queue is full.
    pub fn enqueue(&self, req: QueuedRequest) -> Result<(), QueueError> {
        let mut guard = self.inner.lock();
        if guard.len() >= self.max_size {
            return Err(QueueError::QueueFull);
        }
        guard.push_back(req);
        Ok(())
    }

    /// Dequeue the next request.
    pub fn dequeue(&self) -> Option<QueuedRequest> {
        let mut guard = self.inner.lock();
        guard.pop_front()
    }

    /// Drain all requests from the queue.
    pub fn drain(&self) -> Vec<QueuedRequest> {
        let mut guard = self.inner.lock();
        guard.drain(..).collect()
    }

    /// Remove timed-out requests and return them.
    pub fn remove_timed_out(&self) -> Vec<QueuedRequest> {
        let mut guard = self.inner.lock();
        let mut timed_out = Vec::new();
        let mut i = 0;
        while i < guard.len() {
            if guard[i].is_timed_out(self.timeout) {
                if let Some(req) = guard.remove(i) {
                    timed_out.push(req);
                }
            } else {
                i += 1;
            }
        }
        timed_out
    }

    /// Get the current queue length.
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.lock().is_empty()
    }

    /// Get the maximum queue size.
    pub fn max_size(&self) -> usize {
        self.max_size
    }
}

/// Errors that can occur when queuing requests.
#[derive(Debug, Clone)]
pub enum QueueError {
    QueueFull,
    Timeout,
}

impl std::fmt::Display for QueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueueError::QueueFull => write!(f, "Request queue is full"),
            QueueError::Timeout => write!(f, "Request timed out in queue"),
        }
    }
}

impl std::error::Error for QueueError {}
