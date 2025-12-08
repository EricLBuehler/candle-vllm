use super::responses::ChatCompletionChunk;
use axum::response::sse::Event;
use flume::Receiver;
use futures::{FutureExt, Stream};
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

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.status == StreamingStatus::Stopped {
            return Poll::Ready(None);
        }

        let recv_result = self.rx.recv_async().poll_unpin(cx);

        match recv_result {
            Poll::Ready(Ok(resp)) => match resp {
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
            Poll::Ready(Err(flume::RecvError::Disconnected)) => {
                if self.status == StreamingStatus::Started {
                    self.status = StreamingStatus::Interrupted;
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ChatResponse, Streamer, StreamingStatus};
    use crate::openai::responses::{ChatCompletionChunk, Choice, ChoiceData};
    use futures::StreamExt;
    use tokio::time::{timeout, Duration};

    #[tokio::test]
    async fn streamer_wakes_on_new_messages() {
        let (tx, rx) = flume::unbounded();

        let streamer = Streamer {
            rx,
            status: StreamingStatus::Uninitialized,
        };

        let handle = tokio::spawn(async move {
            futures::pin_mut!(streamer);
            streamer.next().await
        });

        // Allow the stream task to register its waker before sending
        tokio::task::yield_now().await;

        let chunk = ChatCompletionChunk {
            id: "test".to_string(),
            choices: vec![Choice {
                delta: ChoiceData {
                    role: Some("assistant".to_string()),
                    content: Some("hi".to_string()),
                    tool_calls: None,
                    reasoning: None,
                },
                finish_reason: None,
                index: 0,
            }],
            created: 0,
            model: "model".to_string(),
            object: "chat.completion.chunk",
            system_fingerprint: None,
            conversation_id: None,
            resource_id: None,
        };

        tx.send(ChatResponse::Chunk(chunk)).expect("send chunk");

        let event = timeout(Duration::from_secs(1), handle)
            .await
            .expect("stream task timed out")
            .expect("join stream task");

        assert!(event.is_some(), "stream did not yield an event");
    }
}
