use super::responses::{ChatCompletionChunk, EmbeddingResponse};
use crate::openai::logger::ChatCompletionLogger;
use axum::response::sse::Event;
use flume::Receiver;
use futures::Stream;
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

#[derive(PartialEq)]
pub enum StreamingStatus {
    Uninitialized,
    Started,
    Interrupted,
    Stopped,
}
pub enum ChatResponse {
    InternalError(String),
    ValidationError(String),
    ModelError(String),
    Chunk(ChatCompletionChunk),
    Embedding(EmbeddingResponse),
    Done, //finish flag
}

pub struct Streamer {
    pub rx: Receiver<ChatResponse>,
    pub status: StreamingStatus,
    pub logger: Option<Arc<ChatCompletionLogger>>,
}

impl Stream for Streamer {
    type Item = Result<Event, axum::Error>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.status == StreamingStatus::Stopped {
            return Poll::Ready(None);
        }
        match self.rx.try_recv() {
            Ok(resp) => match resp {
                ChatResponse::InternalError(e) => {
                    if let Some(logger) = &self.logger {
                        logger.log_error(&e);
                    }
                    Poll::Ready(Some(Ok(Event::default().data(e))))
                }
                ChatResponse::ValidationError(e) => {
                    if let Some(logger) = &self.logger {
                        logger.log_error(&e);
                    }
                    Poll::Ready(Some(Ok(Event::default().data(e))))
                }
                ChatResponse::ModelError(e) => {
                    if let Some(logger) = &self.logger {
                        logger.log_error(&e);
                    }
                    Poll::Ready(Some(Ok(Event::default().data(e))))
                }
                ChatResponse::Chunk(response) => {
                    if self.status != StreamingStatus::Started {
                        self.status = StreamingStatus::Started;
                    }
                    if let Some(logger) = &self.logger {
                        let mut should_log_final_chunk = false;
                        for choice in &response.choices {
                            if let Some(content) = choice.delta.content.as_deref() {
                                logger.log_stream_token(content);
                            }
                            if let Some(reasoning) = choice.delta.reasoning_content.as_deref() {
                                logger.log_stream_token(reasoning);
                            }
                            if let Some(tool_calls) = choice.delta.tool_calls.as_deref() {
                                logger.log_tool_calls("stream", tool_calls);
                            }
                            if choice.finish_reason.is_some() {
                                should_log_final_chunk = true;
                            }
                        }
                        if should_log_final_chunk {
                            logger.log_stream_end(&response);
                        }
                    }
                    Poll::Ready(Some(Event::default().json_data(response)))
                }
                ChatResponse::Embedding(response) => {
                    if self.status != StreamingStatus::Started {
                        self.status = StreamingStatus::Started;
                    }
                    Poll::Ready(Some(Event::default().json_data(response)))
                }
                ChatResponse::Done => {
                    self.status = StreamingStatus::Stopped;
                    Poll::Ready(Some(Ok(Event::default().data("[DONE]"))))
                }
            },
            Err(e) => {
                if self.status == StreamingStatus::Started && e == flume::TryRecvError::Disconnected
                {
                    self.status = StreamingStatus::Interrupted;
                    Poll::Ready(None)
                } else {
                    Poll::Pending
                }
            }
        }
    }
}
