use tokenizers::tokenizer::{Result, Tokenizer};
use serde::{Deserialize, Serialize};
use polodb_core::{Collection, Database};
use std::sync::Arc;
use tokio::sync::Mutex;

use axum::{http::StatusCode,Json,Router, routing::get, extract::State};

// STRUCTS
#[derive(Debug, Serialize, Deserialize)]
pub struct Context {
    pub text: String,
    pub tokens: Vec<String>,
    pub embedding: Vec<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Query {
    pub prompt: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Resp {
    pub body: String,
}

pub struct AppState {
    pub tokenizer: Tokenizer,
    pub collection: Collection<Context>,
}

// HELPER FXNS
pub fn chunk_text(text: &str) -> Vec<&str> {
    text.split("\n").collect()
}

pub fn create_context(tokenizer: &Tokenizer, text: &str) -> Result<Context> {
    let encoding = tokenizer.encode(text, false)?;
    let embedding = encoding.get_ids().to_vec();
    let tokens = encoding.get_tokens().to_vec();
    let context = Context {
        text: text.to_string(),
        tokens,
        embedding
    };
    Ok(context)
}

// ROUTES
// GET /
pub async fn root() -> &'static str {
    "Rusty RAG Application Launched!"
}

// GET /query
pub async fn query_handler(State(state): State<Arc<AppState>>, Json(payload): Json<Query>) -> (StatusCode, Json<Resp>) {
    let prompt = payload.prompt;
    println!("Received prompt {:?}", prompt);

    // Access tokenizer and collection from the shared state
    let _tokenizer = &state.tokenizer;
    let collection = &state.collection;

    let chunks = collection.find(None).unwrap();
    for chunk in chunks {
        let context = chunk.unwrap();
        let tokens = &context.tokens;
        // let embeddings = &context.embedding;
        println!("#: {:?}, tokens: {:?}", tokens.len(), tokens);
    }

    // Example response
    let resp = Resp {
        body: "Response".to_string()
    };

    (StatusCode::OK, Json(resp))
}
