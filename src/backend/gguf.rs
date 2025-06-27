use anyhow::Result;
use candle_core::quantized::gguf_file::{self, Value};
use itertools::Itertools;
use std::collections::HashMap;
use tokenizers::pre_tokenizers::{
    sequence::Sequence,
    split::{Split, SplitPattern},
    PreTokenizerWrapper,
};
use tokenizers::tokenizer::normalizer::SplitDelimiterBehavior;
use tokenizers::{
    decoders::{
        self, byte_fallback::ByteFallback, byte_level::ByteLevel, fuse::Fuse, strip::Strip,
    },
    models::{bpe::BpeBuilder, unigram::Unigram},
    normalizers::{self, Prepend, Replace},
    processors, AddedToken, DecoderWrapper, ModelWrapper, NormalizerWrapper, Tokenizer,
};
use tracing::info;
// use indexmap::IndexMap;

fn parse_gguf_value(value: &Value) -> String {
    match value {
        Value::Array(vs) => vs
            .iter()
            .map(parse_gguf_value)
            .collect::<Vec<String>>()
            .join(", "),
        Value::Bool(b) => b.to_string(),
        Value::F32(x) => x.to_string(),
        Value::F64(x) => x.to_string(),
        Value::I8(x) => x.to_string(),
        Value::I16(x) => x.to_string(),
        Value::I32(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::String(x) => x.to_string(),
        Value::U8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
    }
}

// Internal invariant: contents and readers must be paired.
/// This abstracts the files for a GGUF model and enables multiple files to be used.
#[allow(dead_code)]
pub struct Content<'a, R: std::io::Seek + std::io::Read> {
    contents: Vec<gguf_file::Content>,
    readers: &'a mut [&'a mut R],
    all_metadata: HashMap<String, Value>,
}

impl<'a, R: std::io::Seek + std::io::Read> Content<'a, R> {
    /// Create a `Content` from a set of file readers.
    pub fn from_readers(readers: &'a mut [&'a mut R]) -> candle_core::Result<Self> {
        let mut contents = Vec::new();
        let n_readers = readers.len();
        for reader in readers.iter_mut() {
            contents.push(gguf_file::Content::read(reader)?);
        }
        let n_splits = contents
            .iter()
            .filter_map(|ct| {
                ct.metadata
                    .get("split.count")
                    .map(|val| val.to_u64().unwrap())
            })
            .fold(Vec::new(), |mut accum, x| {
                if !accum.contains(&x) {
                    accum.push(x);
                }
                accum
            });

        #[allow(clippy::cast_possible_truncation)]
        if !n_splits.is_empty() && n_splits[0] > 0 && n_readers != n_splits[0] as usize {
            candle_core::bail!(
                "Number of GGUF files does not match the number of splits, expected {} files.",
                n_splits[0]
            );
        } else if n_splits.len() == 1 {
            info!("GGUF file has been split into {} shards", n_splits[0]);
        }

        let mut all_metadata = HashMap::new();
        for content in &contents {
            all_metadata.extend(content.metadata.clone())
        }

        Ok(Self {
            contents,
            readers,
            all_metadata,
        })
    }

    /// Get all metadatas
    pub fn get_metadata(&self) -> &HashMap<String, Value> {
        &self.all_metadata
    }
}

pub struct ContentMetadata<'a> {
    pub path_prefix: &'a str,
    pub metadata: &'a HashMap<String, gguf_file::Value>,
}

impl ContentMetadata<'_> {
    // Retrieve a prop the struct needs by querying the metadata content:
    pub fn get_value<T: TryFromValue>(&self, field_name: &str) -> Result<T, anyhow::Error> {
        let prop_key = format!("{prefix}.{field_name}", prefix = self.path_prefix);
        let value = self.metadata.get(&prop_key).cloned();

        // Unwrap the inner value of the `Value` enum via trait method,
        // otherwise format error with prop key as context:
        value
            .try_value_into()
            .or_else(|e| anyhow::bail!("`{prop_key}` `{e}`"))
    }

    // Retrieve a prop the struct needs by querying the metadata content:
    pub fn get_option_value<T: TryFromValue>(
        &self,
        field_name: &str,
    ) -> Result<Option<T>, anyhow::Error> {
        let prop_key = format!("{prefix}.{field_name}", prefix = self.path_prefix);
        let value = self.metadata.get(&prop_key).cloned();

        // Unwrap the inner value of the `Value` enum via trait method,
        // otherwise format error with prop key as context:
        value
            .map(|v| {
                v.try_value_into()
                    .or_else(|e| anyhow::bail!("`{prop_key}` `{e}`"))
            })
            .map_or(Ok(None), |res| res.map(Some))
    }

    // Fail early - Catch all missing mandatory keys upfront:
    pub fn has_required_keys(&self, fields: &[&str]) -> Result<()> {
        let mut all_props_are_present = true;

        for field_name in fields {
            let prop_key = format!("{prefix}.{field_name}", prefix = self.path_prefix);

            if !self.metadata.contains_key(&prop_key) {
                all_props_are_present = false;
                tracing::warn!("Expected GGUF metadata to have key: `{prop_key}`");
            }
        }

        anyhow::ensure!(all_props_are_present, "Tokenizer is missing required props");
        Ok(())
    }
}

// These traits below are a workaround for converting candles GGUF `Value` enum type wrapper.
// A better upstream approach would instead be to provide serialize/deserialize support?
pub trait TryFromValue {
    fn try_from_value(value: gguf_file::Value) -> Result<Self, candle_core::Error>
    where
        Self: Sized;
}

use akin::akin;

// Value wrapped types, each has a different conversion method:
// NOTE: Type conversion methods internally bail with "not a <into type> <input value>"
// https://docs.rs/candle-core/latest/candle_core/quantized/gguf_file/enum.Value.html#variants
akin! {
    let &types = [String, bool, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64];
    let &to_type = [
        value.to_string().cloned(),
        value.to_bool(),
        value.to_f32(),
        value.to_f64(),
        value.to_i8(),
        value.to_i16(),
        value.to_i32(),
        value.to_i64(),
        value.to_u8(),
        value.to_u16(),
        value.to_u32(),
        value.to_u64(),
    ];

    impl TryFromValue for *types {
        fn try_from_value(value: gguf_file::Value) -> Result<Self, candle_core::Error> {
            *to_type.or_else(|_| candle_core::bail!("value is not a `*types`"))
        }
    }
}

// Vec<Value> to Vec<T> from above types:
impl<T: TryFromValue> TryFromValue for Vec<T> {
    fn try_from_value(value_vec: gguf_file::Value) -> Result<Self, candle_core::Error> {
        value_vec
            .to_vec()
            .or_else(|_| candle_core::bail!("value is not a `Vec`"))?
            .clone()
            .into_iter()
            .map(|item| T::try_from_value(item))
            .collect()
    }
}

pub trait TryValueInto<T>: Sized {
    fn try_value_into(self) -> Result<T, candle_core::Error>;
}

impl<T: TryFromValue> TryValueInto<T> for gguf_file::Value {
    fn try_value_into(self) -> Result<T, candle_core::Error> {
        T::try_from_value(self)
    }
}

impl<T: TryFromValue> TryValueInto<T> for Option<gguf_file::Value> {
    fn try_value_into(self) -> Result<T, candle_core::Error> {
        match self {
            Some(value) => value.try_value_into(),
            None => candle_core::bail!("Expected `Option<gguf_file::Value>` to contain a value"),
        }
    }
}

struct PropsGGUFTemplate {
    chat_template: Option<String>,
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUFTemplate {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> Result<Self, Self::Error> {
        // No required keys

        let props = Self {
            chat_template: c.get_option_value("chat_template")?,
        };

        Ok(props)
    }
}

// Get chat template from GGUF metadata if it exists
pub fn get_gguf_chat_template<R: std::io::Seek + std::io::Read>(
    content: &Content<'_, R>,
) -> Result<Option<String>> {
    let metadata = ContentMetadata {
        path_prefix: "tokenizer",
        metadata: content.get_metadata(),
    };
    let props = PropsGGUFTemplate::try_from(metadata)?;
    Ok(props.chat_template)
}

pub struct GGUFInfo {
    pub tokenizer: Tokenizer,
    pub bos: Option<String>,
    pub eos: Option<String>,
    pub unk: Option<String>,
    pub context_length: Option<usize>,
    pub chat_template: Option<String>,
}

struct PropsGGUF {
    model: String,
    tokens: Vec<String>,
    added_tokens: Option<Vec<String>>,
    scores: Option<Vec<f32>>,
    merges: Option<Vec<String>>,
    unk: Option<u32>,
    eos: Option<u32>,
    bos: Option<u32>,
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> Result<Self, Self::Error> {
        let required = ["model", "tokens", "eos_token_id"];
        c.has_required_keys(&required)?;

        let props = Self {
            model: c.get_value("model")?,
            tokens: c.get_value("tokens")?,
            added_tokens: c.get_value("added_tokens").ok(),
            scores: c.get_value("scores").ok(),
            merges: c.get_value("merges").ok(),
            unk: c.get_value("unknown_token_id").ok(),
            eos: c.get_value("eos_token_id").ok(),
            bos: c.get_value("bos_token_id").ok(),
        };

        Ok(props)
    }
}

pub fn get_gguf_info<R: std::io::Seek + std::io::Read>(
    content: &Content<'_, R>,
) -> Result<GGUFInfo> {
    let chat_template = {
        match get_gguf_chat_template(content) {
            Ok(c) => c,
            _ => None,
        }
    };

    let metadata = ContentMetadata {
        path_prefix: "tokenizer.ggml",
        metadata: content.get_metadata(),
    };
    let md_get = |s: &str| match metadata.metadata.get(s) {
        None => candle_core::bail!("cannot find {s} in metadata"),
        Some(v) => Ok(v),
    };

    let mut context_length = 4096;
    let mut token_types = Vec::<i32>::new();
    for key in metadata.metadata.keys() {
        if key.contains(".context_length") {
            context_length = md_get(key).unwrap().to_u32().unwrap();
        }
        if !key.contains("tokenizer.") {
            tracing::info!("{key} : {:?}", parse_gguf_value(&md_get(key).unwrap()));
        }

        if key.contains("tokenizer.ggml.token_type") {
            let vtypes: &Vec<Value> = md_get(key).unwrap().to_vec().unwrap();
            let v: Vec<i32> = vtypes.iter().map(|v| v.to_i32().unwrap()).collect();
            token_types.extend(v);
        }
    }
    let props = PropsGGUF::try_from(metadata)?;

    let (mut tokenizer, kind) = match props.model.as_str() {
        "llama" | "replit" => unigram_tokenizer(&props)?,
        "gpt2" => bpe_tokenizer(&props)?,
        other => {
            anyhow::bail!("Tokenizer model `{other}` not supported.");
        }
    };

    //token type other than 1 treated as special token
    let mut num_special_tokens = 0;
    if token_types.len() == props.tokens.len() {
        for i in 0..props.tokens.len() {
            if token_types[i] != 1i32 {
                let tk = props.tokens[i].clone();
                tokenizer.add_special_tokens(&[AddedToken::from(tk.to_string(), true)]);
                num_special_tokens += 1;
            }
        }
    }

    info!(
        "GGUF tokenizer model is `{model}`, kind: `{kind:?}`, num tokens: {}, num special tokens: {}, num added tokens: {}, num merges: {}, num scores: {}",
        tokenizer.get_vocab_size(true),
        num_special_tokens,
        props.added_tokens.as_ref().map(|x| x.len()).unwrap_or(0),
        props.merges.as_ref().map(|x| x.len()).unwrap_or(0),
        props.scores.as_ref().map(|x| x.len()).unwrap_or(0),
        model = props.model,
    );

    let bos = match props.bos {
        Some(u) => Some(props.tokens[u as usize].clone()),
        _ => None,
    };

    let eos = match props.eos {
        Some(u) => Some(props.tokens[u as usize].clone()),
        _ => None,
    };

    let unk = match props.unk {
        Some(u) => Some(props.tokens[u as usize].clone()),
        _ => None,
    };

    Ok(GGUFInfo {
        tokenizer,
        bos,
        eos,
        unk,
        context_length: Some(context_length as usize),
        chat_template,
    })
}

// TODO: Add support for additional tokenizer models: WordPiece, WordLevel
// https://docs.rs/tokenizers/latest/tokenizers/models/enum.ModelWrapper.html
#[derive(Debug)]
enum TokenizerKind {
    Unigram,
    Bpe,
}

fn bpe_tokenizer(p: &PropsGGUF) -> Result<(Tokenizer, TokenizerKind)> {
    // BPE merges have each string item as a space-delimited pair:
    // https://github.com/EricLBuehler/mistral.rs/pull/397#discussion_r1631988370
    let merges = p
        .merges
        .as_ref()
        .ok_or(anyhow::Error::msg("BPE tokenizer must include merges"))?
        .iter()
        .map(|merge| {
            let split: (&str, &str) = merge
                .splitn(2, ' ')
                .collect_tuple()
                .expect("Failed to convert split into 2-tuple");
            (split.0.to_string(), split.1.to_string())
        })
        .collect::<Vec<_>>();

    let mut vocab = ahash::AHashMap::new();
    for (i, token) in p.tokens.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        vocab.insert(token.clone(), i as u32);
    }

    let PropsGGUF { eos, bos, unk, .. } = *p;

    let mut bpe = BpeBuilder::new().vocab_and_merges(vocab, merges);
    if let Some(unk) = unk {
        bpe = bpe.unk_token(p.tokens[unk as usize].to_string());
    };

    let bpe = bpe.build().map_err(anyhow::Error::msg)?;

    let mut tokenizer = TokenizerX::new(
        ModelWrapper::BPE(bpe),
        Some(Decoder::ByteLevel(true, true, true)),
        None,
    )?;

    let split = Split::new(
        SplitPattern::Regex("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    ).unwrap();

    // example:
    // "type": "ByteLevel",
    // "add_prefix_space": false,
    // "trim_offsets": false,
    // "use_regex": false
    let pre_tokenizer = Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
    ]);

    tokenizer.with_pre_tokenizer(Some(pre_tokenizer));

    tokenizer.with_decoder(Some(decoders::byte_level::ByteLevel::new(
        false, false, false,
    )));
    tokenizer.with_post_processor(Some(processors::byte_level::ByteLevel::new(
        false, false, false,
    )));

    for i in [bos, eos, unk] {
        if i.is_some() {
            let tk = p.tokens[i.unwrap() as usize].clone();
            tokenizer.add_special_tokens(&[AddedToken::from(tk.to_string(), true)]);
        }
    }

    Ok((tokenizer, TokenizerKind::Bpe))
}

fn unigram_tokenizer(p: &PropsGGUF) -> Result<(Tokenizer, TokenizerKind)> {
    let PropsGGUF { unk, eos, bos, .. } = *p;
    // Unigram (SentencePiece) default UNK is 0
    let unk = unk.unwrap_or(0);

    // Create the Tokenizer model:
    let model = {
        let vocab: Vec<(String, f64)> = {
            let Some(s) = p.scores.as_ref() else {
                anyhow::bail!(
                    "`llama` unigram tokenizer is missing required metadata `tokenizer.ggml.scores`"
                );
            };
            let scores = s.iter().cloned().map(|f_32| f_32 as f64);

            p.tokens.iter().cloned().zip(scores).collect()
        };

        Unigram::from(vocab, Some(unk as usize), true).map_err(anyhow::Error::msg)?
    };

    // Decoder + Normalizer config reference:
    // https://github.com/EricLBuehler/mistral.rs/pull/389#discussion_r1630620763
    let decoder = Decoder::Sequence(vec![
        Decoder::Replace("▁", " "),
        Decoder::ByteFallback,
        Decoder::Fuse,
        Decoder::Strip(' ', 1, 0),
    ]);

    let normalizer = Normalizer::Sequence(vec![
        Normalizer::Prepend("▁"),
        Normalizer::Replace(" ", "▁"),
    ]);

    let mut tokenizer: Tokenizer = TokenizerX::new(
        ModelWrapper::Unigram(model),
        Some(decoder),
        Some(normalizer),
    )?;

    // Add special tokens (bos, eos, unk):
    for i in [bos, eos, Some(unk)] {
        if i.is_some() {
            let tk = p.tokens[i.unwrap() as usize].clone();
            tokenizer.add_special_tokens(&[AddedToken::from(tk.to_string(), true)]);
        }
    }

    Ok((tokenizer, TokenizerKind::Unigram))
}

// This is a workaround to have a better builder API.
// Upstream `TokenizerBuilder` is difficult to work with:
// https://github.com/huggingface/tokenizers/issues/1549
struct TokenizerX;

impl TokenizerX {
    #[allow(clippy::new_ret_no_self)]
    fn new<'a>(
        model: ModelWrapper,
        decoder: Option<Decoder<'a>>,
        normalizer: Option<Normalizer<'a>>,
    ) -> Result<Tokenizer> {
        let mut tokenizer = Tokenizer::new(model);

        // Handle local enum to remote enum type:
        if let Some(decoder) = decoder {
            let d = DecoderWrapper::try_from(decoder)?;
            tokenizer.with_decoder(Some(d));
        }
        if let Some(normalizer) = normalizer {
            let n: NormalizerWrapper = NormalizerWrapper::try_from(normalizer)?;
            tokenizer.with_normalizer(Some(n));
        }

        Ok(tokenizer)
    }
}

// Convenient alternative to upstream:
// https://docs.rs/tokenizers/latest/tokenizers/decoders/enum.DecoderWrapper.html
enum Decoder<'a> {
    ByteFallback,
    Fuse,
    Replace(&'a str, &'a str),
    Strip(char, usize, usize),
    Sequence(Vec<Self>),
    ByteLevel(bool, bool, bool),
}

// Convert into upstream type wrapped enum variants:
impl TryFrom<Decoder<'_>> for DecoderWrapper {
    type Error = anyhow::Error;

    fn try_from(variant: Decoder) -> Result<Self, Self::Error> {
        let value: DecoderWrapper = match variant {
            Decoder::ByteFallback => ByteFallback::default().into(),
            Decoder::Fuse => Fuse::default().into(),
            Decoder::Replace(pattern, content) => Replace::new(pattern, content)
                .map_err(anyhow::Error::msg)?
                .into(),
            Decoder::Strip(content, start, stop) => Strip::new(content, start, stop).into(),
            Decoder::Sequence(decoders) => {
                let seq = decoders
                    .into_iter()
                    .map(DecoderWrapper::try_from)
                    .collect::<Result<Vec<DecoderWrapper>>>()?;

                decoders::sequence::Sequence::new(seq).into()
            }
            Decoder::ByteLevel(add_prefix_space, trim_offsets, use_regex) => {
                ByteLevel::new(add_prefix_space, trim_offsets, use_regex).into()
            }
        };

        Ok(value)
    }
}

// Convenient alternative to upstream:
// https://docs.rs/tokenizers/latest/tokenizers/normalizers/enum.NormalizerWrapper.html
enum Normalizer<'a> {
    Prepend(&'a str),
    Replace(&'a str, &'a str),
    Sequence(Vec<Self>),
}

impl TryFrom<Normalizer<'_>> for NormalizerWrapper {
    type Error = anyhow::Error;

    fn try_from(variant: Normalizer) -> Result<Self, Self::Error> {
        let value: NormalizerWrapper = match variant {
            Normalizer::Prepend(prepend) => Prepend::new(prepend.to_owned()).into(),
            Normalizer::Replace(pattern, content) => Replace::new(pattern, content)
                .map_err(anyhow::Error::msg)?
                .into(),
            Normalizer::Sequence(decoders) => {
                let seq = decoders
                    .into_iter()
                    .map(NormalizerWrapper::try_from)
                    .collect::<Result<Vec<NormalizerWrapper>>>()?;

                normalizers::Sequence::new(seq).into()
            }
        };

        Ok(value)
    }
}
