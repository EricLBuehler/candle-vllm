use std::sync::Arc;

use flume::Sender;

use super::{
    BufferedFinalizeResult, ChatCompletionChunk, ChatCompletionUsageResponse, Choice, ChoiceData,
    DefaultPipeline, LLMEngine, Logprobs, ParserState, Sequence, SequenceGroup, StreamEmission,
    StreamResult, StreamToolParser,
};
use crate::openai::{streaming::ChatResponse, utils::get_created_time_secs};
use tracing::warn;

impl LLMEngine {
    pub fn get_stream_response(
        &self,
        request_id: String,
        created: u64,
        content: Option<String>,
        tool_calls: Option<Vec<crate::tools::ToolCall>>,
        finish_reason: Option<String>,
        usage: Option<ChatCompletionUsageResponse>,
        pipeline: &DefaultPipeline,
    ) -> ChatCompletionChunk {
        let mut choices = Vec::new();
        let choice = Choice {
            delta: ChoiceData {
                role: "assistant".to_string(),
                content,
                tool_calls,
            },
            finish_reason,
            index: 0,
        };
        choices.push(choice);

        ChatCompletionChunk {
            id: request_id,
            choices,
            created,
            model: pipeline.name().to_string(),
            object: "chat.completion.chunk",
            system_fingerprint: None,
            usage,
        }
    }

    pub fn collect_stream_emission_for_token(
        &self,
        rank: usize,
        group: &Arc<SequenceGroup>,
        seq: &Arc<Sequence>,
        logprobs: &Logprobs,
    ) -> StreamEmission {
        let (pipeline, _) = self.get_pipeline(rank).unwrap();
        let token_str = &logprobs.bytes;
        let should_parse_tools = group.sampling_params.mcp_mode.is_some();
        let mut content = None;

        {
            let outer = seq.deref();
            let mut data = outer.deref_mut();

            if should_parse_tools {
                if data.stream_tool_parser.is_none() {
                    data.stream_tool_parser = Some(StreamToolParser::new_with_config(
                        &pipeline.tool_model_type,
                        pipeline.tool_parser_model_id.clone(),
                        pipeline.tool_config.clone(),
                        group.tools.clone(),
                        pipeline.enforce_parser.clone(),
                    ));
                }
                if let Some(parser) = data.stream_tool_parser.as_mut() {
                    match parser.process_token(logprobs.token, token_str) {
                        StreamResult::Content(text) | StreamResult::FlushBuffer(text) => {
                            if !text.is_empty() {
                                content = Some(text);
                            }
                        }
                        StreamResult::Buffering => {}
                        StreamResult::ToolCalls(calls) => {
                            data.pending_tool_calls.extend(calls);
                        }
                    }
                }
            } else if !token_str.is_empty() {
                content = Some(token_str.clone());
            }
        }

        StreamEmission {
            content,
            tool_calls: None,
        }
    }

    pub fn apply_pending_finish_logprobs(
        &self,
        rank: usize,
        group: &Arc<SequenceGroup>,
        seq: &Arc<Sequence>,
    ) {
        let pending = {
            let outer = seq.deref();
            let mut data = outer.deref_mut();
            data.pending_finish_logprobs.take()
        };

        let Some(logprobs) = pending else {
            return;
        };

        if group.sampling_params.mcp_mode.is_some() {
            let (pipeline, _) = self.get_pipeline(rank).unwrap();
            let outer = seq.deref();
            let mut data = outer.deref_mut();
            if data.stream_tool_parser.is_none() {
                data.stream_tool_parser = Some(StreamToolParser::new_with_config(
                    &pipeline.tool_model_type,
                    pipeline.tool_parser_model_id.clone(),
                    pipeline.tool_config.clone(),
                    group.tools.clone(),
                    pipeline.enforce_parser.clone(),
                ));
            }
            if let Some(parser) = data.stream_tool_parser.as_mut() {
                if let StreamResult::ToolCalls(calls) =
                    parser.process_token(logprobs.token, &logprobs.bytes)
                {
                    data.pending_tool_calls.extend(calls);
                }
            }
        }

        seq.deref_mut().add_token(logprobs);
    }

    pub fn collect_stream_emission_on_finish(
        &self,
        group: &Arc<SequenceGroup>,
        seq: &Arc<Sequence>,
    ) -> StreamEmission {
        let mut content = None;
        let mut tool_calls = None;

        {
            let outer = seq.deref();
            let mut data = outer.deref_mut();
            let should_parse_tools = group.sampling_params.mcp_mode.is_some();

            if should_parse_tools {
                let pending_was_empty = data.pending_tool_calls.is_empty();
                if let Some(parser) = data.stream_tool_parser.as_mut() {
                    let (mut finalized_calls, buffered_content, mut reparsed_calls) = {
                        let mut buffered_content = None;
                        let mut finalized_calls = Vec::new();
                        let mut reparsed_calls = Vec::new();

                        if matches!(parser.state(), ParserState::Buffering) {
                            match parser.finalize_buffered_tool_calls() {
                                Some(BufferedFinalizeResult::ToolCalls(parsed)) => {
                                    finalized_calls = parsed;
                                }
                                Some(BufferedFinalizeResult::FlushBuffer(buffer)) => {
                                    if !buffer.is_empty() {
                                        buffered_content = Some(buffer);
                                    }
                                }
                                None => {}
                            }
                        }

                        if pending_was_empty && finalized_calls.is_empty() {
                            reparsed_calls = parser.reparse_accumulated_output();
                        }

                        (finalized_calls, buffered_content, reparsed_calls)
                    };

                    data.pending_tool_calls.append(&mut finalized_calls);
                    if !reparsed_calls.is_empty() {
                        warn!(
                            "Recovered {} tool call(s) from full-output fallback parse",
                            reparsed_calls.len()
                        );
                        data.pending_tool_calls.append(&mut reparsed_calls);
                    }

                    if data.pending_tool_calls.is_empty() {
                        content = buffered_content;
                    }
                }
            }

            let pending = std::mem::take(&mut data.pending_tool_calls);
            if !pending.is_empty() {
                tool_calls = Some(pending);
            }
        }

        StreamEmission {
            content,
            tool_calls,
        }
    }

    pub fn send_stream_emission(
        &self,
        rank: usize,
        sender: &Sender<ChatResponse>,
        group: &Arc<SequenceGroup>,
        seq: &Arc<Sequence>,
        emission: StreamEmission,
        finish_reason: Option<String>,
    ) {
        let should_log_free_blocks =
            finish_reason.is_none() && emission.tool_calls.is_none() && emission.content.is_some();
        let (pipeline, _) = self.get_pipeline(rank).unwrap();
        let chunk = if let Some(tool_calls) = emission.tool_calls {
            self.get_stream_response(
                group.request_id.clone(),
                get_created_time_secs(),
                None,
                Some(
                    tool_calls
                        .into_iter()
                        .enumerate()
                        .map(|(idx, call)| call.with_index(idx))
                        .collect(),
                ),
                Some("tool_calls".to_string()),
                None,
                pipeline,
            )
        } else {
            self.get_stream_response(
                group.request_id.clone(),
                group.arrival_time,
                emission.content,
                None,
                finish_reason,
                None,
                pipeline,
            )
        };

        if matches!(
            chunk.choices.first(),
            Some(choice) if choice.delta.tool_calls.is_some()
        ) {
            tracing::info!("Sending final tool call chunk: {:?}", chunk);
        }

        let ret = sender.send(ChatResponse::Chunk(chunk));
        if ret.is_err() {
            warn!(
                "Send stream response error! (sequence id {})",
                seq.deref().get_id()
            );
            seq.deref_mut().set_finish_reason("abort".to_string());
        } else if should_log_free_blocks && seq.deref_mut().get_len() % 1000 == 0 {
            self.scheduler.print_free_blocks();
        }
    }
}
