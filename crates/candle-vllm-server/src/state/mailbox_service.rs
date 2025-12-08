use crate::config::MailboxBackendConfig;
use crate::state::queue_backends::{build_mailbox_backend, MailboxBackend};
use candle_core::Error;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MailboxRecord {
    pub request_id: String,
    pub model: String,
    pub created: u64,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<serde_json::Value>,
}

#[derive(Clone)]
pub struct MailboxService {
    backend: MailboxBackend,
    memory: Arc<Mutex<HashMap<String, MailboxRecord>>>,
}

impl MailboxService {
    pub fn new(config: &MailboxBackendConfig) -> Result<Self, Error> {
        let backend = build_mailbox_backend(config)?;
        if let MailboxBackend::Postgres { .. } = backend {
            return Err(Error::Msg(
                "mailbox.backend=postgres is not yet implemented; choose memory".to_string(),
            ));
        }
        Ok(Self {
            backend,
            memory: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub fn store(&self, record: MailboxRecord) -> Result<(), Error> {
        match &self.backend {
            MailboxBackend::Memory { .. } => {
                self.memory.lock().insert(record.request_id.clone(), record);
                Ok(())
            }
            MailboxBackend::Postgres { .. } => unreachable!("handled in new()"),
        }
    }

    pub fn get(&self, request_id: &str) -> Option<MailboxRecord> {
        match &self.backend {
            MailboxBackend::Memory { .. } => self.memory.lock().get(request_id).cloned(),
            MailboxBackend::Postgres { .. } => unreachable!("handled in new()"),
        }
    }

    pub fn list(&self) -> Vec<MailboxRecord> {
        match &self.backend {
            MailboxBackend::Memory { .. } => {
                self.memory.lock().values().cloned().collect::<Vec<_>>()
            }
            MailboxBackend::Postgres { .. } => unreachable!("handled in new()"),
        }
    }

    pub fn delete(&self, request_id: &str) -> bool {
        match &self.backend {
            MailboxBackend::Memory { .. } => self.memory.lock().remove(request_id).is_some(),
            MailboxBackend::Postgres { .. } => unreachable!("handled in new()"),
        }
    }
}

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
