use tokenizers::tokenizer::{Result, Tokenizer};
use serde::{Deserialize, Serialize};
use polodb_core::{Collection, Database};
use std::sync::Arc;
use tokio::sync::Mutex;
use ndarray::{Array, ArrayView};
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

pub fn build_embedding_matrix(collection: &Collection<Context>) {
    println!("Building RAG matrix...");
    let mut rag_matrix = Array::zeros((0, 60));
    let entries = collection.find(None).unwrap();
    for entry in entries {
        let embedding = entry.unwrap().embedding;
        let _ = rag_matrix.push_row((&Array::from_vec(embedding)).into());
    }
    println!("{:?}", rag_matrix.shape())
}

// ROUTES
// GET /
pub async fn root() -> &'static str {
    "Rusty RAG Application Launched!"
}

// GET /query
pub async fn query_handler(State(state): State<Arc<AppState>>, Json(payload): Json<Query>) -> (StatusCode, Json<Resp>) {
    let prompt = payload.prompt;
    println!("Prompt {:?}", prompt);
    let tokenizer = &state.tokenizer;
    let collection = &state.collection;
    // Prompt embedding
    // let prompt_encoding = tokenizer.encode(prompt, false)?;
    // let prompt_embedding = prompt_encoding.get_ids().to_vec();
    // Search for closest context match
    let entries = collection.find(None).unwrap();
    for entry in entries {
        let context = entry.unwrap();
        let embedding = &context.embedding;

        // println!("#: {:?}, tokens: {:?}", tokens.len(), tokens);
    }

    // Example response
    let resp = Resp {
        body: "Response".to_string()
    };

    (StatusCode::OK, Json(resp))
}
