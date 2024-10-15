use super::responses::ChatCompletionChunk;
use axum::response::sse::Event;
use flume::Receiver;
use futures::Stream;
use std::{
    pin::Pin,
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
    Done, //finish flag
}

pub struct Streamer {
    pub rx: Receiver<ChatResponse>,
    pub status: StreamingStatus,
}

impl Stream for Streamer {
    type Item = Result<Event, axum::Error>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.status == StreamingStatus::Stopped {
            return Poll::Ready(None);
        }
        match self.rx.try_recv() {
            Ok(resp) => match resp {
                ChatResponse::InternalError(e) => Poll::Ready(Some(Ok(Event::default().data(e)))),
                ChatResponse::ValidationError(e) => Poll::Ready(Some(Ok(Event::default().data(e)))),
                ChatResponse::ModelError(e) => Poll::Ready(Some(Ok(Event::default().data(e)))),
                ChatResponse::Chunk(response) => {
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
