//! Tests for StreamingRegistry.

use crate::parking_lot::{StreamingRegistry, StreamingTokenResult};
use std::time::Duration;

#[test]
fn test_register_and_retrieve() {
    let registry = StreamingRegistry::with_default_retention();
    let (tx, rx) = flume::unbounded();

    let key = "test-key".to_string();
    let request_id = "test-request".to_string();

    registry.register(key.clone(), request_id, rx);
    assert_eq!(registry.len(), 1);
    assert!(!registry.is_empty());

    let retrieved = registry.retrieve(&key);
    assert!(retrieved.is_some());

    // Send a test token
    let token = StreamingTokenResult {
        text: "hello".to_string(),
        token_id: 123,
        is_finished: false,
        finish_reason: None,
        is_reasoning: false,
    };
    tx.send(Ok(token)).unwrap();

    // Receive through retrieved channel
    let received = retrieved.unwrap().recv().unwrap();
    assert!(received.is_ok());
    assert_eq!(received.unwrap().text, "hello");
}

#[test]
fn test_remove() {
    let registry = StreamingRegistry::with_default_retention();
    let (_tx, rx) = flume::unbounded();

    let key = "test-key".to_string();
    registry.register(key.clone(), "test-request".to_string(), rx);
    assert_eq!(registry.len(), 1);

    let removed = registry.remove(&key);
    assert!(removed);
    assert_eq!(registry.len(), 0);
    assert!(registry.is_empty());

    // Removing again should fail
    let removed_again = registry.remove(&key);
    assert!(!removed_again);
}

#[test]
fn test_retrieve_nonexistent() {
    let registry = StreamingRegistry::with_default_retention();
    let retrieved = registry.retrieve("nonexistent-key");
    assert!(retrieved.is_none());
}

#[test]
fn test_cleanup_expired() {
    let registry = StreamingRegistry::new(Duration::from_millis(50));
    let (_tx, rx) = flume::unbounded();

    registry.register("test-key".to_string(), "test-request".to_string(), rx);
    assert_eq!(registry.len(), 1);

    // Should not clean up immediately
    let removed = registry.cleanup_expired();
    assert_eq!(removed, 0);
    assert_eq!(registry.len(), 1);

    // Wait for expiration
    std::thread::sleep(Duration::from_millis(100));

    // Should clean up now
    let removed = registry.cleanup_expired();
    assert_eq!(removed, 1);
    assert_eq!(registry.len(), 0);
}

#[test]
fn test_multiple_channels() {
    let registry = StreamingRegistry::with_default_retention();

    for i in 0..5 {
        let (_tx, rx) = flume::unbounded();
        registry.register(format!("key-{}", i), format!("request-{}", i), rx);
    }

    assert_eq!(registry.len(), 5);

    // Retrieve all
    for i in 0..5 {
        let retrieved = registry.retrieve(&format!("key-{}", i));
        assert!(retrieved.is_some());
    }

    // Remove all
    for i in 0..5 {
        let removed = registry.remove(&format!("key-{}", i));
        assert!(removed);
    }

    assert_eq!(registry.len(), 0);
}

#[test]
fn test_streaming_registry_default() {
    let registry = StreamingRegistry::default();
    assert_eq!(registry.len(), 0);
    assert!(registry.is_empty());
}

#[test]
fn test_streaming_registry_new() {
    let registry = StreamingRegistry::new(Duration::from_secs(3600));
    assert_eq!(registry.len(), 0);
}

#[test]
fn test_streaming_registry_concurrent_access() {
    use std::sync::Arc;
    use std::thread;

    let registry = Arc::new(StreamingRegistry::with_default_retention());
    let mut handles = vec![];

    // Spawn multiple threads registering channels
    for i in 0..10 {
        let registry_clone = Arc::clone(&registry);
        let handle = thread::spawn(move || {
            let (_tx, rx) = flume::unbounded();
            registry_clone.register(format!("key-{}", i), format!("request-{}", i), rx);
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(registry.len(), 10);
}

#[test]
fn test_streaming_registry_cleanup_partial_expiration() {
    let registry = StreamingRegistry::new(Duration::from_millis(50));

    // Register two channels
    let (_tx1, rx1) = flume::unbounded();
    registry.register("key-1".to_string(), "request-1".to_string(), rx1);

    // Wait a bit, then register another
    std::thread::sleep(Duration::from_millis(30));
    let (_tx2, rx2) = flume::unbounded();
    registry.register("key-2".to_string(), "request-2".to_string(), rx2);

    // Wait for first to expire but not second
    std::thread::sleep(Duration::from_millis(30));

    let removed = registry.cleanup_expired();
    assert_eq!(removed, 1);
    assert_eq!(registry.len(), 1);

    // First should be gone, second should remain
    assert!(registry.retrieve("key-1").is_none());
    assert!(registry.retrieve("key-2").is_some());
}

#[test]
fn test_streaming_registry_cleanup_no_expired() {
    let registry = StreamingRegistry::new(Duration::from_secs(3600));
    let (_tx, rx) = flume::unbounded();

    registry.register("key-1".to_string(), "request-1".to_string(), rx);

    // Should not clean up immediately
    let removed = registry.cleanup_expired();
    assert_eq!(removed, 0);
    assert_eq!(registry.len(), 1);
}

#[test]
fn test_streaming_registry_empty_cleanup() {
    let registry = StreamingRegistry::with_default_retention();

    // Cleanup on empty registry should be safe
    let removed = registry.cleanup_expired();
    assert_eq!(removed, 0);
    assert_eq!(registry.len(), 0);
}

#[test]
fn test_streaming_registry_clone() {
    let registry = StreamingRegistry::with_default_retention();
    let (_tx, rx) = flume::unbounded();

    registry.register("key-1".to_string(), "request-1".to_string(), rx);

    // Clone should share the same underlying data
    let cloned = registry.clone();
    assert_eq!(cloned.len(), 1);
    assert_eq!(registry.len(), 1);

    // Removing from one should affect the other
    cloned.remove("key-1");
    assert_eq!(registry.len(), 0);
    assert_eq!(cloned.len(), 0);
}

#[test]
fn test_streaming_registry_retrieve_clone_receiver() {
    let registry = StreamingRegistry::with_default_retention();
    let (tx, rx) = flume::unbounded();

    registry.register("key-1".to_string(), "request-1".to_string(), rx);

    // Retrieve multiple times should get independent receivers
    let rx1 = registry.retrieve("key-1").unwrap();
    let rx2 = registry.retrieve("key-1").unwrap();

    // Both should receive the same messages
    tx.send(Ok(StreamingTokenResult {
        text: "hello".to_string(),
        token_id: 1,
        is_finished: false,
        finish_reason: None,
        is_reasoning: false,
    }))
    .unwrap();

    let msg1 = rx1.recv().unwrap();
    let msg2 = rx2.recv().unwrap();

    assert_eq!(msg1.unwrap().text, "hello");
    assert_eq!(msg2.unwrap().text, "hello");
}

#[tokio::test]
async fn test_streaming_registry_cleanup_task() {
    let registry = StreamingRegistry::new(Duration::from_millis(50));
    let (_tx, rx) = flume::unbounded();

    registry.register("key-1".to_string(), "request-1".to_string(), rx);

    // Start cleanup task
    let cleanup_handle = registry
        .clone()
        .start_cleanup_task(Duration::from_millis(100));

    // Wait for cleanup to run
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Channel should be cleaned up
    assert_eq!(registry.len(), 0);

    // Cleanup task should still be running
    cleanup_handle.abort();
}
