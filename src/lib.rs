use std::fs;
use serde::{Deserialize, Serialize};
use tokenizers::tokenizer::{Result, Tokenizer};

#[derive(Debug, Serialize, Deserialize)]
pub struct Context {
    // tokens: str,
    pub embedding: Vec<u32>,
}

pub fn chunk_text(text: &str) -> Vec<&str> {
    // Implement your chunking logic here
    text.split('\n').collect()
}

pub fn create_context(tokenizer: &Tokenizer, text: &str) -> Result<Context> {
    let encoding = tokenizer.encode(text, false)?;
    let embedding = encoding.get_ids().to_vec();
    let tokens = encoding.get_tokens().to_vec();
    let context = Context { embedding };
    Ok(context)
}