//! Webhook service for firing callbacks on mailbox events.
//!
//! This module provides webhook functionality for notifying external services
//! when inference requests complete, especially useful for:
//! - Async/fire-and-forget inference patterns
//! - Client disconnect recovery
//! - Integration with external systems (e.g., Supabase Edge Functions)

use crate::config::WebhookConfig;
use crate::state::backend_traits::MailboxRecord;
use std::collections::HashMap;
#[cfg(feature = "webhooks")]
use tracing::{debug, error, info, warn};
#[cfg(not(feature = "webhooks"))]
use tracing::warn;

/// Error type for webhook operations.
#[derive(Debug, Clone)]
pub struct WebhookError {
    pub message: String,
    pub kind: WebhookErrorKind,
    pub status_code: Option<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebhookErrorKind {
    /// Configuration error
    Configuration,
    /// Network/connection error
    Network,
    /// HTTP error response
    HttpError,
    /// Timeout
    Timeout,
    /// Serialization error
    Serialization,
    /// Authentication error
    Authentication,
}

impl std::fmt::Display for WebhookError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.status_code {
            Some(code) => write!(f, "{}: {} (HTTP {})", self.kind, self.message, code),
            None => write!(f, "{}: {}", self.kind, self.message),
        }
    }
}

impl std::fmt::Display for WebhookErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Configuration => write!(f, "configuration error"),
            Self::Network => write!(f, "network error"),
            Self::HttpError => write!(f, "HTTP error"),
            Self::Timeout => write!(f, "timeout"),
            Self::Serialization => write!(f, "serialization error"),
            Self::Authentication => write!(f, "authentication error"),
        }
    }
}

impl std::error::Error for WebhookError {}

impl WebhookError {
    pub fn config(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: WebhookErrorKind::Configuration,
            status_code: None,
        }
    }

    pub fn network(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: WebhookErrorKind::Network,
            status_code: None,
        }
    }

    pub fn http(message: impl Into<String>, status_code: u16) -> Self {
        Self {
            message: message.into(),
            kind: WebhookErrorKind::HttpError,
            status_code: Some(status_code),
        }
    }

    pub fn timeout(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: WebhookErrorKind::Timeout,
            status_code: None,
        }
    }
}

/// Webhook delivery mode for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WebhookMode {
    /// Only fire webhook if client disconnects before receiving response
    #[default]
    OnDisconnect,
    /// Always fire webhook when request completes (in addition to normal response)
    Always,
    /// Never fire webhook for this request
    Never,
}

impl std::str::FromStr for WebhookMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "on_disconnect" | "ondisconnect" | "disconnect" => Ok(Self::OnDisconnect),
            "always" | "all" => Ok(Self::Always),
            "never" | "none" | "disabled" => Ok(Self::Never),
            _ => Err(format!("unknown webhook mode: {}", s)),
        }
    }
}

/// Per-request webhook configuration that can override global settings.
#[derive(Debug, Clone, Default)]
pub struct RequestWebhookConfig {
    pub url: Option<String>,
    pub mode: Option<WebhookMode>,
    pub bearer_token: Option<String>,
    pub headers: Option<HashMap<String, String>>,
}

/// Service for managing webhook deliveries.
#[derive(Clone)]
pub struct WebhookService {
    config: Option<WebhookConfig>,
    #[cfg(feature = "webhooks")]
    client: reqwest::Client,
}

impl WebhookService {
    /// Create a new webhook service with the given configuration.
    pub fn new(config: Option<WebhookConfig>) -> Self {
        #[cfg(feature = "webhooks")]
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(
                config.as_ref().map(|c| c.timeout_secs).unwrap_or(30),
            ))
            .build()
            .unwrap_or_default();

        Self {
            config,
            #[cfg(feature = "webhooks")]
            client,
        }
    }

    /// Check if webhooks are enabled.
    pub fn is_enabled(&self) -> bool {
        self.config
            .as_ref()
            .map(|c| c.enabled && c.url.is_some())
            .unwrap_or(false)
    }

    /// Get the default webhook mode from config.
    pub fn default_mode(&self) -> WebhookMode {
        self.config
            .as_ref()
            .map(|c| {
                if c.on_complete {
                    WebhookMode::Always
                } else if c.on_disconnect {
                    WebhookMode::OnDisconnect
                } else {
                    WebhookMode::Never
                }
            })
            .unwrap_or(WebhookMode::OnDisconnect)
    }

    /// Fire a webhook for the given mailbox record.
    ///
    /// This method handles authentication, retries, and error logging.
    #[cfg(feature = "webhooks")]
    pub async fn fire(
        &self,
        record: &MailboxRecord,
        request_config: Option<&RequestWebhookConfig>,
    ) -> Result<(), WebhookError> {
        // Determine the URL to use
        let url = request_config
            .and_then(|c| c.url.as_deref())
            .or_else(|| self.config.as_ref().and_then(|c| c.url.as_deref()))
            .ok_or_else(|| WebhookError::config("no webhook URL configured"))?;

        info!(
            request_id = %record.request_id,
            url = %url,
            "Firing webhook"
        );

        // Build the request
        let mut request = self.client.post(url);

        // Serialize the payload
        let payload = serde_json::to_string(record).map_err(|e| WebhookError {
            message: format!("failed to serialize payload: {}", e),
            kind: WebhookErrorKind::Serialization,
            status_code: None,
        })?;

        // Apply authentication from config
        if let Some(ref config) = self.config {
            if let Some(ref auth) = config.auth {
                request = apply_auth(request, auth, &payload)?;
            }

            // Add custom headers from config
            for (key, value) in &config.headers {
                request = request.header(key.as_str(), value.as_str());
            }

            // Add payload signature if configured
            if config.sign_payload {
                if let Some(ref secret) = config.signing_secret {
                    let signature = compute_hmac_signature(secret, &payload)?;
                    request = request.header("X-Signature-256", format!("sha256={}", signature));
                }
            }
        }

        // Override with per-request config
        if let Some(req_config) = request_config {
            if let Some(ref token) = req_config.bearer_token {
                request = request.header("Authorization", format!("Bearer {}", token));
            }
            if let Some(ref headers) = req_config.headers {
                for (key, value) in headers {
                    request = request.header(key.as_str(), value.as_str());
                }
            }
        }

        // Set content type and body
        request = request
            .header("Content-Type", "application/json")
            .body(payload);

        // Send the request
        let response = request.send().await.map_err(|e| {
            if e.is_timeout() {
                WebhookError::timeout(format!("request timed out: {}", e))
            } else if e.is_connect() {
                WebhookError::network(format!("connection failed: {}", e))
            } else {
                WebhookError::network(format!("request failed: {}", e))
            }
        })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!(
                request_id = %record.request_id,
                status = %status,
                body = %body,
                "Webhook request failed"
            );
            return Err(WebhookError::http(
                format!("webhook returned error: {}", body),
                status.as_u16(),
            ));
        }

        info!(
            request_id = %record.request_id,
            status = %status,
            "Webhook delivered successfully"
        );
        Ok(())
    }

    /// Fire a webhook with retries.
    #[cfg(feature = "webhooks")]
    pub async fn fire_with_retry(
        &self,
        record: &MailboxRecord,
        request_config: Option<&RequestWebhookConfig>,
    ) -> Result<(), WebhookError> {
        let retry_count = self
            .config
            .as_ref()
            .map(|c| c.retry_count)
            .unwrap_or(3);
        let retry_delay_ms = self
            .config
            .as_ref()
            .map(|c| c.retry_delay_ms)
            .unwrap_or(1000);

        let mut last_error = None;
        for attempt in 0..=retry_count {
            if attempt > 0 {
                debug!(
                    request_id = %record.request_id,
                    attempt = attempt,
                    "Retrying webhook"
                );
                tokio::time::sleep(std::time::Duration::from_millis(
                    retry_delay_ms * (1 << (attempt - 1)), // Exponential backoff
                ))
                .await;
            }

            match self.fire(record, request_config).await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    warn!(
                        request_id = %record.request_id,
                        attempt = attempt,
                        error = %e,
                        "Webhook attempt failed"
                    );
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| WebhookError::config("no attempts made")))
    }

    /// Stub implementation when webhooks feature is not enabled.
    #[cfg(not(feature = "webhooks"))]
    pub async fn fire(
        &self,
        _record: &MailboxRecord,
        _request_config: Option<&RequestWebhookConfig>,
    ) -> Result<(), WebhookError> {
        warn!("Webhook feature not enabled, skipping webhook delivery");
        Err(WebhookError::config(
            "webhooks feature not enabled; rebuild with --features webhooks",
        ))
    }

    /// Stub implementation when webhooks feature is not enabled.
    #[cfg(not(feature = "webhooks"))]
    pub async fn fire_with_retry(
        &self,
        _record: &MailboxRecord,
        _request_config: Option<&RequestWebhookConfig>,
    ) -> Result<(), WebhookError> {
        self.fire(_record, _request_config).await
    }
}

/// Apply authentication to a request builder.
#[cfg(feature = "webhooks")]
fn apply_auth(
    mut request: reqwest::RequestBuilder,
    auth: &crate::config::WebhookAuth,
    payload: &str,
) -> Result<reqwest::RequestBuilder, WebhookError> {
    match auth {
        crate::config::WebhookAuth::Bearer { token } => {
            let token = interpolate_env(token);
            request = request.header("Authorization", format!("Bearer {}", token));
        }
        crate::config::WebhookAuth::ApiKey { header, key } => {
            let key = interpolate_env(key);
            request = request.header(header.as_str(), key);
        }
        crate::config::WebhookAuth::Hmac {
            secret,
            algorithm,
            header,
        } => {
            let secret = interpolate_env(secret);
            let signature = match algorithm.to_lowercase().as_str() {
                "sha256" => compute_hmac_signature(&secret, payload)?,
                "sha512" => compute_hmac_sha512_signature(&secret, payload)?,
                other => {
                    return Err(WebhookError::config(format!(
                        "unsupported HMAC algorithm: {}",
                        other
                    )))
                }
            };
            request = request.header(header.as_str(), format!("{}={}", algorithm, signature));
        }
        crate::config::WebhookAuth::None => {}
    }
    Ok(request)
}

/// Compute HMAC-SHA256 signature.
#[cfg(feature = "webhooks")]
fn compute_hmac_signature(secret: &str, payload: &str) -> Result<String, WebhookError> {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    type HmacSha256 = Hmac<Sha256>;
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
        .map_err(|e| WebhookError::config(format!("invalid HMAC key: {}", e)))?;
    mac.update(payload.as_bytes());
    let result = mac.finalize();
    Ok(hex::encode(result.into_bytes()))
}

/// Compute HMAC-SHA512 signature.
#[cfg(feature = "webhooks")]
fn compute_hmac_sha512_signature(secret: &str, payload: &str) -> Result<String, WebhookError> {
    use hmac::{Hmac, Mac};
    use sha2::Sha512;

    type HmacSha512 = Hmac<Sha512>;
    let mut mac = HmacSha512::new_from_slice(secret.as_bytes())
        .map_err(|e| WebhookError::config(format!("invalid HMAC key: {}", e)))?;
    mac.update(payload.as_bytes());
    let result = mac.finalize();
    Ok(hex::encode(result.into_bytes()))
}

/// Interpolate environment variables in a string.
/// Supports ${VAR_NAME} syntax.
#[cfg(feature = "webhooks")]
fn interpolate_env(s: &str) -> String {
    let mut result = s.to_string();
    
    // Find all ${...} patterns
    while let Some(start) = result.find("${") {
        if let Some(end) = result[start..].find('}') {
            let var_name = &result[start + 2..start + end];
            let replacement = std::env::var(var_name).unwrap_or_default();
            result = format!("{}{}{}", &result[..start], replacement, &result[start + end + 1..]);
        } else {
            break;
        }
    }
    
    result
}

impl Default for WebhookService {
    fn default() -> Self {
        Self::new(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webhook_mode_from_str() {
        assert_eq!(
            "on_disconnect".parse::<WebhookMode>().unwrap(),
            WebhookMode::OnDisconnect
        );
        assert_eq!(
            "always".parse::<WebhookMode>().unwrap(),
            WebhookMode::Always
        );
        assert_eq!(
            "never".parse::<WebhookMode>().unwrap(),
            WebhookMode::Never
        );
    }

    #[test]
    fn test_webhook_service_disabled_by_default() {
        let service = WebhookService::new(None);
        assert!(!service.is_enabled());
    }

    #[cfg(feature = "webhooks")]
    #[test]
    fn test_interpolate_env() {
        std::env::set_var("TEST_WEBHOOK_VAR", "secret123");
        let result = interpolate_env("Bearer ${TEST_WEBHOOK_VAR}");
        assert_eq!(result, "Bearer secret123");
        std::env::remove_var("TEST_WEBHOOK_VAR");
    }
}
