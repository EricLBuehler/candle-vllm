use crate::config::QueueBackendConfig;
use crate::state::queue_backends::{build_queue_backend, QueueBackend};
use candle_core::Error;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueRecord {
    pub id: String,
    pub model: String,
    pub queued_at: u64,
    pub request: serde_json::Value,
}

#[derive(Clone)]
pub struct QueueService {
    backend: QueueBackend,
    memory: Arc<Mutex<HashMap<String, VecDeque<QueueRecord>>>>,
    yaque_root: Option<PathBuf>,
    postgres_root: Option<PathBuf>,
    sqlite_root: Option<PathBuf>,
    surreal_root: Option<PathBuf>,
}

impl QueueService {
    pub fn new(config: &QueueBackendConfig) -> Result<Self, Error> {
        let backend = build_queue_backend(config)?;
        let yaque_root = match &backend {
            QueueBackend::Yaque { dir, .. } => {
                let root = PathBuf::from(dir);
                fs::create_dir_all(&root).map_err(|e| {
                    Error::Msg(format!(
                        "failed to create yaque dir {}: {e}",
                        root.display()
                    ))
                })?;
                Some(root)
            }
            _ => None,
        };
        let postgres_root = match &backend {
            QueueBackend::Postgres { url, .. } => {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                url.hash(&mut hasher);
                let hash = hasher.finish();
                let root = std::env::temp_dir()
                    .join("candle_vllm_queue_pg")
                    .join(format!("{hash:x}"));
                fs::create_dir_all(&root).map_err(|e| {
                    Error::Msg(format!(
                        "failed to create postgres queue emulation dir {}: {e}",
                        root.display()
                    ))
                })?;
                Some(root)
            }
            _ => None,
        };
        let sqlite_root = match &backend {
            QueueBackend::Sqlite { path, .. } => {
                let root = PathBuf::from(path);
                fs::create_dir_all(&root).map_err(|e| {
                    Error::Msg(format!(
                        "failed to create sqlite queue dir {}: {e}",
                        root.display()
                    ))
                })?;
                Some(root)
            }
            _ => None,
        };
        let surreal_root = match &backend {
            QueueBackend::Surreal { path, .. } => {
                let root = PathBuf::from(path);
                fs::create_dir_all(&root).map_err(|e| {
                    Error::Msg(format!(
                        "failed to create surrealdb queue dir {}: {e}",
                        root.display()
                    ))
                })?;
                Some(root)
            }
            _ => None,
        };
        Ok(Self {
            backend,
            memory: Arc::new(Mutex::new(HashMap::new())),
            yaque_root,
            postgres_root,
            sqlite_root,
            surreal_root,
        })
    }

    pub fn enqueue(&self, record: QueueRecord) -> Result<(), Error> {
        match &self.backend {
            QueueBackend::Memory { .. } => {
                let mut guard = self.memory.lock();
                guard
                    .entry(record.model.clone())
                    .or_insert_with(VecDeque::new)
                    .push_back(record);
                Ok(())
            }
            QueueBackend::Yaque { .. } => {
                let root = self
                    .yaque_root
                    .as_ref()
                    .ok_or_else(|| Error::Msg("yaque root not initialized".into()))?;
                self.write_record(root, &record)
            }
            QueueBackend::Postgres { .. } => {
                let root = self
                    .postgres_root
                    .as_ref()
                    .ok_or_else(|| Error::Msg("postgres queue root not initialized".into()))?;
                self.write_record(root, &record)
            }
            QueueBackend::Sqlite { .. } => {
                let root = self
                    .sqlite_root
                    .as_ref()
                    .ok_or_else(|| Error::Msg("sqlite queue root not initialized".into()))?;
                self.write_record(root, &record)
            }
            QueueBackend::Surreal { .. } => {
                let root = self
                    .surreal_root
                    .as_ref()
                    .ok_or_else(|| Error::Msg("surreal queue root not initialized".into()))?;
                self.write_record(root, &record)
            }
        }
    }

    pub fn dequeue(&self, model: &str, max: usize) -> Vec<QueueRecord> {
        match &self.backend {
            QueueBackend::Memory { .. } => {
                let mut guard = self.memory.lock();
                let deque = guard.entry(model.to_string()).or_insert_with(VecDeque::new);
                let mut drained = Vec::new();
                for _ in 0..max {
                    if let Some(r) = deque.pop_front() {
                        drained.push(r);
                    } else {
                        break;
                    }
                }
                drained
            }
            QueueBackend::Postgres { .. } => {
                if let Some(root) = &self.postgres_root {
                    let model_dir = root.join(model);
                    self.read_dir_and_drain(&model_dir, max, true)
                } else {
                    Vec::new()
                }
            }
            QueueBackend::Sqlite { .. } => {
                if let Some(root) = &self.sqlite_root {
                    let model_dir = root.join(model);
                    self.read_dir_and_drain(&model_dir, max, true)
                } else {
                    Vec::new()
                }
            }
            QueueBackend::Surreal { .. } => {
                if let Some(root) = &self.surreal_root {
                    let model_dir = root.join(model);
                    self.read_dir_and_drain(&model_dir, max, true)
                } else {
                    Vec::new()
                }
            }
            QueueBackend::Yaque { .. } => {
                if let Some(root) = &self.yaque_root {
                    let model_dir = root.join(model);
                    self.read_dir_and_drain(&model_dir, max, true)
                } else {
                    Vec::new()
                }
            }
        }
    }

    pub fn len(&self, model: &str) -> usize {
        match &self.backend {
            QueueBackend::Memory { .. } => {
                self.memory.lock().get(model).map(|q| q.len()).unwrap_or(0)
            }
            QueueBackend::Postgres { .. } => self.count_dir(model, self.postgres_root.as_ref()),
            QueueBackend::Sqlite { .. } => self.count_dir(model, self.sqlite_root.as_ref()),
            QueueBackend::Surreal { .. } => self.count_dir(model, self.surreal_root.as_ref()),
            QueueBackend::Yaque { .. } => self.count_dir(model, self.yaque_root.as_ref()),
        }
    }

    pub fn list(&self, model: Option<&str>) -> Vec<QueueRecord> {
        match &self.backend {
            QueueBackend::Memory { .. } => {
                let guard = self.memory.lock();
                if let Some(model) = model {
                    guard
                        .get(model)
                        .map(|q| q.iter().cloned().collect())
                        .unwrap_or_default()
                } else {
                    guard.values().flat_map(|q| q.iter().cloned()).collect()
                }
            }
            QueueBackend::Postgres { .. } => self.list_from_fs(model, self.postgres_root.as_ref()),
            QueueBackend::Sqlite { .. } => self.list_from_fs(model, self.sqlite_root.as_ref()),
            QueueBackend::Surreal { .. } => self.list_from_fs(model, self.surreal_root.as_ref()),
            QueueBackend::Yaque { .. } => self.list_from_fs(model, self.yaque_root.as_ref()),
        }
    }

    fn write_record(&self, root: &PathBuf, record: &QueueRecord) -> Result<(), Error> {
        let model_dir = root.join(&record.model);
        fs::create_dir_all(&model_dir).map_err(|e| {
            Error::Msg(format!(
                "failed to create queue dir {}: {e}",
                model_dir.display()
            ))
        })?;
        let file_path = model_dir.join(format!("{}-{}.json", record.queued_at, record.id));
        let payload = serde_json::to_vec(record)
            .map_err(|e| Error::Msg(format!("failed to serialize queue record: {e}")))?;
        fs::write(&file_path, payload).map_err(|e| {
            Error::Msg(format!(
                "failed to write queue file {}: {e}",
                file_path.display()
            ))
        })?;
        Ok(())
    }

    fn read_dir_and_drain(&self, dir: &PathBuf, max: usize, delete: bool) -> Vec<QueueRecord> {
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };
        let mut files: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .collect();
        files.sort();
        let mut records = Vec::new();
        for path in files.into_iter().take(max) {
            if let Ok(data) = fs::read(&path) {
                if let Ok(record) = serde_json::from_slice::<QueueRecord>(&data) {
                    records.push(record);
                }
            }
            if delete {
                let _ = fs::remove_file(&path);
            }
        }
        records
    }

    fn list_from_fs(&self, model: Option<&str>, root: Option<&PathBuf>) -> Vec<QueueRecord> {
        let root = match root {
            Some(r) => r,
            None => return Vec::new(),
        };
        if let Some(model) = model {
            return self.read_dir_and_drain(&root.join(model), usize::MAX, false);
        }
        let mut all = Vec::new();
        if let Ok(models) = fs::read_dir(root) {
            for entry in models.flatten() {
                if entry.path().is_dir() {
                    all.extend(self.read_dir_and_drain(&entry.path(), usize::MAX, false));
                }
            }
        }
        all
    }

    fn count_dir(&self, model: &str, root: Option<&PathBuf>) -> usize {
        let root = match root {
            Some(r) => r,
            None => return 0,
        };
        let model_dir = root.join(model);
        if let Ok(entries) = fs::read_dir(model_dir) {
            entries.flatten().filter(|e| e.path().is_file()).count()
        } else {
            0
        }
    }
}

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
