use crate::config::{MailboxBackendConfig, QueueBackendConfig};
use candle_core::Error;

/// Selected queue backend with validated parameters.
#[derive(Debug, Clone)]
pub enum QueueBackend {
    Memory { persistence: bool },
    Postgres { url: String, persistence: bool },
    Sqlite { path: String, persistence: bool },
    Yaque { dir: String, persistence: bool },
    Surreal { path: String, persistence: bool },
}

/// Selected mailbox backend with validated parameters.
#[derive(Debug, Clone)]
pub enum MailboxBackend {
    Memory { retention_secs: u64 },
    Postgres { url: String, retention_secs: u64 },
}

pub fn build_queue_backend(config: &QueueBackendConfig) -> candle_core::Result<QueueBackend> {
    match config.backend.to_lowercase().as_str() {
        "memory" => Ok(QueueBackend::Memory {
            persistence: config.persistence,
        }),
        "postgres" => {
            #[cfg(feature = "queue-postgres")]
            {
                let url = config.postgres_url.clone().ok_or_else(|| {
                    Error::Msg("queue.backend=postgres requires postgres_url".into())
                })?;
                Ok(QueueBackend::Postgres {
                    url,
                    persistence: config.persistence,
                })
            }
            #[cfg(not(feature = "queue-postgres"))]
            {
                Err(Error::Msg(
                    "queue.backend=postgres requires the `queue-postgres` feature".into(),
                ))
            }
        }
        "sqlite" => {
            #[cfg(feature = "queue-sqlite")]
            {
                let path = config
                    .yaque_dir
                    .clone()
                    .unwrap_or_else(|| "queue.sqlite".to_string());
                Ok(QueueBackend::Sqlite {
                    path,
                    persistence: config.persistence,
                })
            }
            #[cfg(not(feature = "queue-sqlite"))]
            {
                Err(Error::Msg(
                    "queue.backend=sqlite requires the `queue-sqlite` feature".into(),
                ))
            }
        }
        "yaque" => {
            let dir = config
                .yaque_dir
                .clone()
                .ok_or_else(|| Error::Msg("queue.backend=yaque requires yaque_dir".into()))?;
            Ok(QueueBackend::Yaque {
                dir,
                persistence: config.persistence,
            })
        }
        "surrealdb" => {
            #[cfg(feature = "queue-surreal")]
            {
                let path = config.yaque_dir.clone().ok_or_else(|| {
                    Error::Msg("queue.backend=surrealdb requires yaque_dir/path".into())
                })?;
                Ok(QueueBackend::Surreal {
                    path,
                    persistence: config.persistence,
                })
            }
            #[cfg(not(feature = "queue-surreal"))]
            {
                Err(Error::Msg(
                    "queue.backend=surrealdb requires the `queue-surreal` feature".into(),
                ))
            }
        }
        other => Err(Error::Msg(format!("unsupported queue backend '{}'", other))),
    }
}

pub fn build_mailbox_backend(config: &MailboxBackendConfig) -> candle_core::Result<MailboxBackend> {
    match config.backend.to_lowercase().as_str() {
        "memory" => Ok(MailboxBackend::Memory {
            retention_secs: config.retention_secs,
        }),
        "postgres" => {
            let url = config.postgres_url.clone().ok_or_else(|| {
                Error::Msg("mailbox.backend=postgres requires postgres_url".into())
            })?;
            Ok(MailboxBackend::Postgres {
                url,
                retention_secs: config.retention_secs,
            })
        }
        other => Err(Error::Msg(format!(
            "unsupported mailbox backend '{}'",
            other
        ))),
    }
}
