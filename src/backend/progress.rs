#[cfg(feature = "nccl")]
use crate::openai::communicator::DaemonManager;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::{thread, time};

pub trait ProgressLike: Send + Sync {
    fn get_progress(&self) -> (usize, usize);
    fn set_progress(&mut self, p: usize);
}

pub struct ProgressReporter {
    pub rank: usize,
    pub progress: usize,
}

impl ProgressLike for ProgressReporter {
    fn get_progress(&self) -> (usize, usize) {
        (self.rank, self.progress)
    }

    fn set_progress(&mut self, p: usize) {
        self.progress = p;
    }
}

impl ProgressReporter {
    pub fn new(rank: usize) -> Self {
        Self { rank, progress: 0 }
    }
}

unsafe impl Send for ProgressReporter {}
unsafe impl Sync for ProgressReporter {}

pub struct Progress {
    m: MultiProgress,
    bars: Vec<ProgressBar>,
    size: usize,
}

impl Progress {
    pub fn new(n: usize, size: usize) -> Progress {
        let m = MultiProgress::new();
        let sty = ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:60.cyan/blue} {pos:>4}/{len:4} {msg}",
        )
        .unwrap()
        .progress_chars("##-");

        let mut bars = Vec::<ProgressBar>::new();
        for i in 0..n {
            let pb = m.add(ProgressBar::new(size as u64));
            pb.set_style(sty.clone());
            if n > 1 {
                pb.set_message(format!("On Rank {i} Device"));
            }
            bars.push(pb);
        }

        if n > 1 {
            m.println(format!("Loading model in {n} ranks!")).unwrap();
        }
        Self { m, bars, size }
    }

    pub fn update(&self, idx: usize, progress: usize) {
        if idx < self.bars.len() && progress > 0 {
            let pos = self.bars[idx].position();
            self.bars[idx].inc(progress as u64 - pos);
            if self.bars.len() > 1 {
                if progress >= self.size {
                    self.bars[idx].set_message(format!("On Rank {idx} Device Finished"));
                } else {
                    self.bars[idx].set_message(format!("On Rank {idx} Device"));
                }
            }
        }
    }

    pub fn finish(&self) {
        for idx in 0..self.bars.len() {
            let pos = self.bars[idx].position();
            self.bars[idx].inc(self.size as u64 - pos);
            if self.bars.len() > 1 {
                self.bars[idx].set_message(format!("On Rank {idx} Device Finished"));
            }
        }
        self.m.clear().unwrap();
    }
}

#[allow(unused_variables)]
pub async fn progress_worker(
    num_subprocess: Option<usize>,
    length: usize,
    progress_reporter: Arc<RwLock<ProgressReporter>>,
) -> std::thread::JoinHandle<()> {
    #[cfg(feature = "nccl")]
    use tracing::{debug, warn};
    let mut finished_map = HashMap::<usize, usize>::new();
    let reporter = progress_reporter.clone();
    let handle = thread::spawn(move || {
        #[cfg(feature = "nccl")]
        let mut connect_retry_count = 0;
        #[cfg(feature = "nccl")]
        let (mut command_manager, progress_bar) = if DaemonManager::is_daemon() {
            let mut manager = DaemonManager::new_command("progress", None);
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
                    manager = DaemonManager::new_command("progress", None);
                    continue;
                } else {
                    warn!("{:?}", manager);
                    break;
                }
            }
            (manager, None)
        } else {
            (
                DaemonManager::new_command("progress", num_subprocess),
                Some(Progress::new(num_subprocess.unwrap_or(0) + 1, length)),
            )
        };
        #[cfg(not(feature = "nccl"))]
        let progress_bar = Some(Progress::new(1, length));

        let _ = thread::sleep(time::Duration::from_millis(1000_u64));

        loop {
            {
                let (rank, progress) = reporter.read().unwrap().get_progress();
                finished_map.insert(rank, progress);
                #[cfg(feature = "nccl")]
                if DaemonManager::is_daemon() {
                    //report progress to main process
                    let _ = command_manager
                        .as_mut()
                        .unwrap()
                        .progress(Some((rank, progress)));
                } else {
                    progress_bar.as_ref().unwrap().update(rank, progress); // for current rank
                                                                           //receive progress from daemon processes
                    if let Ok(Some(progresses)) = command_manager.as_mut().unwrap().progress(None) {
                        for (rank, progress) in progresses {
                            progress_bar.as_ref().unwrap().update(rank, progress); //for other ranks
                            finished_map.insert(rank, progress);
                            debug!("rank {} progress {}", rank, progress);
                        }
                    }
                }

                #[cfg(not(feature = "nccl"))]
                progress_bar.as_ref().unwrap().update(rank, progress);

                if progress >= length {
                    #[cfg(not(feature = "nccl"))]
                    {
                        progress_bar.as_ref().unwrap().finish();
                        break;
                    }
                    #[cfg(feature = "nccl")]
                    if !DaemonManager::is_daemon() {
                        if finished_map.len() < 2 || finished_map.values().all(|v| v == &length) {
                            progress_bar.as_ref().unwrap().finish();
                            warn!("all ranks finished model loading!");
                            break;
                        }
                    } else {
                        break;
                    }
                }
            }

            let _ = thread::sleep(time::Duration::from_millis(500_u64));
        }
    });
    handle
}
