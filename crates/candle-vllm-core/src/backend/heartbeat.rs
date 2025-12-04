use crate::openai::communicator::DaemonManager;
use std::{process, thread, time};
use tracing::{debug, info, warn};
pub async fn heartbeat_worker(num_subprocess: Option<usize>) {
    let _ = thread::spawn(move || {
        let mut connect_retry_count = 0;
        let mut command_manager = if DaemonManager::is_daemon() {
            let mut manager = DaemonManager::new_command("heartbeat", None);
            loop {
                if manager.is_ok() {
                    break;
                } else if connect_retry_count < 120 {
                    connect_retry_count += 1;
                    warn!(
                        "Retry connect to main process' command channel ({:?})!",
                        manager
                    );
                    let _ = thread::sleep(time::Duration::from_millis(1000 as u64));
                    manager = DaemonManager::new_command("heartbeat", None);
                    continue;
                } else {
                    warn!("{:?}", manager);
                    break;
                }
            }
            manager
        } else {
            DaemonManager::new_command("heartbeat", num_subprocess)
        };
        let mut heartbeat_error_count = 0;
        info!("enter heartbeat processing loop ({:?})", command_manager);
        loop {
            let alive_result = command_manager.as_mut().unwrap().heartbeat();
            if alive_result.is_err() {
                warn!("{:?}", alive_result);
                if heartbeat_error_count > 10 {
                    warn!(
                        "heartbeat detection failed, exit the current process because of {:?}",
                        alive_result
                    );
                    process::abort();
                }
                heartbeat_error_count += 1;
            } else {
                debug!("paired processes still alive!");
            }
            let _ = thread::sleep(time::Duration::from_millis(1000 as u64));
        }
    });
}
