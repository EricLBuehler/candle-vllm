use std::sync::Arc;

use flume::Sender;

use super::{
    BufferedFinalizeResult, ChatCompletionChunk, ChatCompletionUsageResponse, Choice, ChoiceData,
    DefaultPipeline, LLMEngine, Logprobs, ParserState, Sequence, SequenceGroup, StreamEmission,
    StreamResult, StreamToolParser,
};
use crate::openai::{streaming::ChatResponse, utils::get_created_time_secs};
use crate::tools::stream_parser::strip_reasoning_markers;
use tracing::warn;

enum TokenAction {
    Content(String),
    ReasoningContent(String),
    SuppressToolMarkup(String),
    ToolCalls(Vec<crate::tools::ToolCall>),
    None,
}

impl LLMEngine {
    fn append_stream_text(slot: &mut Option<String>, text: String) {
        if text.is_empty() {
            return;
        }
        if let Some(existing) = slot.as_mut() {
            existing.push_str(&text);
        } else {
            *slot = Some(text);
        }
    }

    fn decode_prompt_replay_text(
        &self,
        pipeline: &DefaultPipeline,
        replay_ids: &[u32],
    ) -> Option<String> {
        if replay_ids.is_empty() {
            return None;
        }
        let text = pipeline
            .tokenizer()
            .decode(replay_ids, false)
            .unwrap_or_default();
        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    }

    pub fn get_stream_response(
        &self,
        request_id: String,
        created: u64,
        role: Option<String>,
        content: Option<String>,
        reasoning_content: Option<String>,
        tool_calls: Option<Vec<crate::tools::ToolCall>>,
        finish_reason: Option<String>,
        usage: Option<ChatCompletionUsageResponse>,
        pipeline: &DefaultPipeline,
    ) -> ChatCompletionChunk {
        let mut choices = Vec::new();
        let choice = Choice {
            delta: ChoiceData {
                role,
                content,
                reasoning_content,
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

    fn maybe_send_stream_role_start(
        &self,
        sender: &Sender<ChatResponse>,
        group: &Arc<SequenceGroup>,
        seq: &Arc<Sequence>,
        pipeline: &DefaultPipeline,
    ) -> bool {
        let should_send = {
            let outer = seq.deref();
            let mut data = outer.deref_mut();
            if data.stream_role_sent {
                false
            } else {
                data.stream_role_sent = true;
                true
            }
        };

        if !should_send {
            return true;
        }

        let chunk = self.get_stream_response(
            group.request_id.clone(),
            group.arrival_time,
            Some("assistant".to_string()),
            None,
            None,
            None,
            None,
            None,
            pipeline,
        );
        sender.send(ChatResponse::Chunk(chunk)).is_ok()
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
        let stream_reasoning = crate::stream_as_reasoning_content() && should_parse_tools;
        let mut content = None;
        let mut reasoning_content = None;

        {
            let outer = seq.deref();
            let mut data = outer.deref_mut();

            if should_parse_tools {
                if data.stream_tool_parser.is_none() {
                    let mut parser = StreamToolParser::new_with_config(
                        &pipeline.tool_model_type,
                        pipeline.tool_parser_model_id.clone(),
                        pipeline.tool_config.clone(),
                        group.tools.clone(),
                        pipeline.enforce_parser.clone(),
                    );
                    if group.active_reasoning_end.is_some() {
                        parser.set_initial_reasoning_end_marker(group.active_reasoning_end.clone());
                    }
                    if stream_reasoning {
                        parser.set_detect_tools_in_reasoning(true);
                    }
                    data.stream_tool_parser = Some(parser);
                }

                // Replay prompt suffix tokens (e.g. `<think>\n`) through the
                // parser before the first real decoded token so the parser
                // enters reasoning mode naturally.
                if !data.prompt_replay_consumed {
                    data.prompt_replay_consumed = true;
                    if let Some(replay_ids) = group.prompt_replay_token_ids.as_ref() {
                        if data.stream_tool_parser.is_some() {
                            for &token_id in replay_ids {
                                let token_text = pipeline
                                    .tokenizer()
                                    .decode(&[token_id], false)
                                    .unwrap_or_default();
                                let has_pending = !data.pending_tool_calls.is_empty();
                                let action = {
                                    let parser = data.stream_tool_parser.as_mut().unwrap();
                                    let was_in_reasoning = parser.in_reasoning();
                                    match parser.process_token(token_id, &token_text) {
                                        StreamResult::Content(text) => {
                                            if text.is_empty() {
                                                TokenAction::None
                                            } else if parser.contains_tool_markup(&text) {
                                                TokenAction::SuppressToolMarkup(text)
                                            } else if has_pending {
                                                TokenAction::None
                                            } else if stream_reasoning {
                                                let stripped = strip_reasoning_markers(&text);
                                                if stripped.is_empty() {
                                                    TokenAction::None
                                                } else if was_in_reasoning {
                                                    TokenAction::ReasoningContent(stripped)
                                                } else {
                                                    TokenAction::Content(stripped)
                                                }
                                            } else {
                                                TokenAction::Content(text)
                                            }
                                        }
                                        StreamResult::FlushBuffer(text) => {
                                            if text.is_empty() {
                                                TokenAction::None
                                            } else if parser.contains_tool_markup(&text) {
                                                TokenAction::SuppressToolMarkup(text)
                                            } else if has_pending {
                                                TokenAction::None
                                            } else {
                                                let safe_text =
                                                    parser.sanitize_tool_markup_for_display(&text);
                                                if safe_text.is_empty() {
                                                    TokenAction::None
                                                } else if stream_reasoning {
                                                    let stripped =
                                                        strip_reasoning_markers(&safe_text);
                                                    if stripped.is_empty() {
                                                        TokenAction::None
                                                    } else {
                                                        TokenAction::Content(stripped)
                                                    }
                                                } else {
                                                    TokenAction::Content(safe_text)
                                                }
                                            }
                                        }
                                        StreamResult::Buffering => TokenAction::None,
                                        StreamResult::ToolCalls(calls) => {
                                            TokenAction::ToolCalls(calls)
                                        }
                                    }
                                };
                                match action {
                                    TokenAction::Content(text) => {
                                        Self::append_stream_text(&mut content, text)
                                    }
                                    TokenAction::ReasoningContent(text) => {
                                        Self::append_stream_text(&mut reasoning_content, text)
                                    }
                                    TokenAction::SuppressToolMarkup(text) => {
                                        data.suppressed_tool_markup.push_str(&text);
                                    }
                                    TokenAction::ToolCalls(calls) => {
                                        data.pending_tool_calls.extend(calls);
                                    }
                                    TokenAction::None => {}
                                }
                            }
                            tracing::info!(
                                "Replayed {} prompt suffix token(s) through stream parser",
                                replay_ids.len()
                            );
                        }
                    }
                }

                let has_pending = !data.pending_tool_calls.is_empty();
                let action = if let Some(parser) = data.stream_tool_parser.as_mut() {
                    let was_in_reasoning = parser.in_reasoning();
                    match parser.process_token(logprobs.token, token_str) {
                        StreamResult::Content(text) => {
                            if text.is_empty() {
                                TokenAction::None
                            } else if parser.contains_tool_markup(&text) {
                                TokenAction::SuppressToolMarkup(text)
                            } else if has_pending {
                                TokenAction::None
                            } else if stream_reasoning {
                                let stripped = strip_reasoning_markers(&text);
                                if stripped.is_empty() {
                                    TokenAction::None
                                } else if was_in_reasoning {
                                    TokenAction::ReasoningContent(stripped)
                                } else {
                                    TokenAction::Content(stripped)
                                }
                            } else {
                                TokenAction::Content(text)
                            }
                        }
                        StreamResult::FlushBuffer(text) => {
                            if text.is_empty() {
                                TokenAction::None
                            } else if parser.contains_tool_markup(&text) {
                                TokenAction::SuppressToolMarkup(text)
                            } else if has_pending {
                                TokenAction::None
                            } else {
                                let safe_text = parser.sanitize_tool_markup_for_display(&text);
                                if safe_text.is_empty() {
                                    TokenAction::None
                                } else if stream_reasoning {
                                    let stripped = strip_reasoning_markers(&safe_text);
                                    if stripped.is_empty() {
                                        TokenAction::None
                                    } else {
                                        TokenAction::Content(stripped)
                                    }
                                } else {
                                    TokenAction::Content(safe_text)
                                }
                            }
                        }
                        StreamResult::Buffering => TokenAction::None,
                        StreamResult::ToolCalls(calls) => TokenAction::ToolCalls(calls),
                    }
                } else {
                    TokenAction::None
                };

                match action {
                    TokenAction::Content(text) => Self::append_stream_text(&mut content, text),
                    TokenAction::ReasoningContent(text) => {
                        Self::append_stream_text(&mut reasoning_content, text)
                    }
                    TokenAction::SuppressToolMarkup(text) => {
                        data.suppressed_tool_markup.push_str(&text);
                    }
                    TokenAction::ToolCalls(calls) => {
                        data.pending_tool_calls.extend(calls);
                    }
                    TokenAction::None => {}
                }
            } else if !data.prompt_replay_consumed {
                data.prompt_replay_consumed = true;
                if let Some(replay_ids) = group.prompt_replay_token_ids.as_ref() {
                    content = self.decode_prompt_replay_text(pipeline, replay_ids);
                }
                if let Some(existing) = content.as_mut() {
                    existing.push_str(token_str);
                } else if !token_str.is_empty() {
                    content = Some(token_str.clone());
                }
            } else if !token_str.is_empty() {
                content = Some(token_str.clone());
            }
        }

        StreamEmission {
            content,
            reasoning_content,
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

        let should_parse_tools = group.sampling_params.mcp_mode.is_some();
        let stream_reasoning = crate::stream_as_reasoning_content() && should_parse_tools;

        if should_parse_tools {
            let (pipeline, _) = self.get_pipeline(rank).unwrap();
            let outer = seq.deref();
            let mut data = outer.deref_mut();
            if data.stream_tool_parser.is_none() {
                let mut parser = StreamToolParser::new_with_config(
                    &pipeline.tool_model_type,
                    pipeline.tool_parser_model_id.clone(),
                    pipeline.tool_config.clone(),
                    group.tools.clone(),
                    pipeline.enforce_parser.clone(),
                );
                if group.active_reasoning_end.is_some() {
                    parser.set_initial_reasoning_end_marker(group.active_reasoning_end.clone());
                }
                if stream_reasoning {
                    parser.set_detect_tools_in_reasoning(true);
                }
                data.stream_tool_parser = Some(parser);
            }

            if !data.prompt_replay_consumed {
                data.prompt_replay_consumed = true;
                if let Some(replay_ids) = group.prompt_replay_token_ids.as_ref() {
                    if let Some(parser) = data.stream_tool_parser.as_mut() {
                        for &token_id in replay_ids {
                            let token_text = pipeline
                                .tokenizer()
                                .decode(&[token_id], false)
                                .unwrap_or_default();
                            let _ = parser.process_token(token_id, &token_text);
                        }
                        tracing::info!(
                            "Replayed {} prompt suffix token(s) through stream parser",
                            replay_ids.len()
                        );
                    }
                }
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
        rank: usize,
        group: &Arc<SequenceGroup>,
        seq: &Arc<Sequence>,
    ) -> StreamEmission {
        let mut content = None;
        let mut tool_calls = None;
        let should_parse_tools = group.sampling_params.mcp_mode.is_some();
        let stream_reasoning = crate::stream_as_reasoning_content() && should_parse_tools;
        let pipeline = if !should_parse_tools {
            Some(self.get_pipeline(rank).unwrap().0.as_ref())
        } else {
            None
        };

        {
            let outer = seq.deref();
            let mut data = outer.deref_mut();

            if should_parse_tools {
                let pending_was_empty = data.pending_tool_calls.is_empty();
                let suppressed_tool_markup = std::mem::take(&mut data.suppressed_tool_markup);

                let (finalized_calls, buffered_content, reparsed_calls, sanitized_suppressed) =
                    if let Some(parser) = data.stream_tool_parser.as_mut() {
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
                            if reparsed_calls.is_empty() {
                                let accumulated = parser.accumulated_output().to_string();
                                let stripped = parser.accumulated_output_without_reasoning();
                                if stripped != accumulated && !stripped.trim().is_empty() {
                                    let stripped_calls = futures::executor::block_on(
                                        parser.parse_complete_with_fallback(&stripped),
                                    );
                                    if !stripped_calls.is_empty() {
                                        warn!(
                                            "Recovered {} tool call(s) from reasoning-stripped fallback parse",
                                            stripped_calls.len()
                                        );
                                        reparsed_calls = stripped_calls;
                                    }
                                }
                            }
                        }

                        let sanitized = if !suppressed_tool_markup.is_empty() {
                            let safe =
                                parser.sanitize_tool_markup_for_display(&suppressed_tool_markup);
                            if safe.is_empty() {
                                None
                            } else {
                                Some(safe)
                            }
                        } else {
                            None
                        };

                        (finalized_calls, buffered_content, reparsed_calls, sanitized)
                    } else {
                        (Vec::new(), None, Vec::new(), None)
                    };

                let mut finalized_calls = finalized_calls;
                let mut reparsed_calls = reparsed_calls;

                data.pending_tool_calls.append(&mut finalized_calls);
                if !reparsed_calls.is_empty() {
                    warn!(
                        "Recovered {} tool call(s) from full-output fallback parse",
                        reparsed_calls.len()
                    );
                    data.pending_tool_calls.append(&mut reparsed_calls);
                }

                if data.pending_tool_calls.is_empty() {
                    if let Some(safe) = sanitized_suppressed {
                        content = Some(safe);
                    } else {
                        content = buffered_content;
                    }
                }
            }
            if !should_parse_tools && !data.prompt_replay_consumed {
                data.prompt_replay_consumed = true;
                if let (Some(replay_ids), Some(pipeline)) =
                    (group.prompt_replay_token_ids.as_ref(), pipeline)
                {
                    content = self.decode_prompt_replay_text(pipeline, replay_ids);
                }
            }

            let pending = std::mem::take(&mut data.pending_tool_calls);
            if !pending.is_empty() {
                tool_calls = Some(pending);
            }
        }

        if stream_reasoning {
            if let Some(text) = content.take() {
                let stripped = strip_reasoning_markers(&text);
                if !stripped.is_empty() {
                    content = Some(stripped);
                }
            }
        }

        StreamEmission {
            content,
            reasoning_content: None,
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
        let should_log_free_blocks = finish_reason.is_none()
            && emission.tool_calls.is_none()
            && (emission.content.is_some() || emission.reasoning_content.is_some());
        let (pipeline, _) = self.get_pipeline(rank).unwrap();
        let has_payload = emission.tool_calls.is_some()
            || emission.content.is_some()
            || emission.reasoning_content.is_some()
            || finish_reason.is_some();

        if has_payload && !self.maybe_send_stream_role_start(sender, group, seq, pipeline) {
            warn!(
                "Send stream role chunk error! (sequence id {})",
                seq.deref().get_id()
            );
            seq.deref_mut().set_finish_reason("abort".to_string());
            return;
        }

        if let Some(tool_calls) = emission.tool_calls {
            // OpenAI streaming spec: tool calls require TWO separate chunks:
            //   1) delta.tool_calls=[...], finish_reason=null
            //   2) delta={}, finish_reason="tool_calls"
            let tool_chunk = self.get_stream_response(
                group.request_id.clone(),
                get_created_time_secs(),
                None,
                None,
                None,
                Some(
                    tool_calls
                        .into_iter()
                        .enumerate()
                        .map(|(idx, call)| call.with_index(idx))
                        .collect(),
                ),
                None,
                None,
                pipeline,
            );
            tracing::info!("Sending tool call delta chunk: {:?}", tool_chunk);
            if sender.send(ChatResponse::Chunk(tool_chunk)).is_err() {
                warn!(
                    "Send stream response error! (sequence id {})",
                    seq.deref().get_id()
                );
                seq.deref_mut().set_finish_reason("abort".to_string());
                return;
            }

            let finish_chunk = self.get_stream_response(
                group.request_id.clone(),
                get_created_time_secs(),
                None,
                None,
                None,
                None,
                Some("tool_calls".to_string()),
                None,
                pipeline,
            );
            tracing::info!("Sending tool call finish chunk: {:?}", finish_chunk);
            if sender.send(ChatResponse::Chunk(finish_chunk)).is_err() {
                warn!(
                    "Send stream response error! (sequence id {})",
                    seq.deref().get_id()
                );
                seq.deref_mut().set_finish_reason("abort".to_string());
            }
        } else {
            let chunk = self.get_stream_response(
                group.request_id.clone(),
                group.arrival_time,
                None,
                emission.content,
                emission.reasoning_content,
                None,
                finish_reason,
                None,
                pipeline,
            );

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
}
