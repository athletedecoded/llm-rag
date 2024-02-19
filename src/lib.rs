use tokenizers::tokenizer::{Result, Tokenizer};
use serde::{Deserialize, Serialize};
use polodb_core::{Collection, Database};
use std::sync::Arc;
use tokio::sync::Mutex;
use ndarray::{Array, Axis, ArrayView, ArrayBase, Dim, OwnedRepr};
use axum::{http::StatusCode,Json,Router, routing::get, extract::State};
use ndarray_stats::QuantileExt;

// STRUCTS
#[derive(Debug, Serialize, Deserialize)]
pub struct Context {
    pub text: String,
    pub tokens: Vec<String>,
    pub embedding: Vec<f64>,
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
    pub rag_matrix: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>
}

// HELPER FXNS
pub fn chunk_text(text: &str) -> Vec<&str> {
    text.split("\n").collect()
}

pub fn create_context(tokenizer: &Tokenizer, text: &str) -> Result<Context> {
    let encoding = tokenizer.encode(text, false)?;
    let tokens = encoding.get_tokens().to_vec();
    let embedding = encoding.get_ids().iter().map(|&x| x as f64).collect();
    let context = Context {
        text: text.to_string(),
        tokens,
        embedding
    };
    Ok(context)
}

// pub fn normalize_embedding(raw_embedding: Vec<f64>) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> {
//     let embedding = Array::from_vec(raw_embedding);
//     let norm = embedding.iter().fold(0.0, |acc, &x| acc + x * x).sqrt();
//     embedding.mapv(|x| x / norm)
// }

pub fn build_rag_matrix(collection: &Collection<Context>, context_window: usize) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>{
    println!("Building RAG matrix...");
    let mut rag_matrix = Array::zeros((0, context_window));
    let entries = collection.find(None).unwrap();
    for entry in entries {
        let embedding = Array::from_vec(entry.unwrap().embedding);
        let _ = rag_matrix.push(Axis(0), (&embedding).into());
    }
    rag_matrix
}

// ROUTES
// GET /
pub async fn root() -> &'static str {
    "Rusty RAG Application Launched!"
}

// GET /query
pub async fn query_handler(State(state): State<Arc<AppState>>, Json(payload): Json<Query>) -> (StatusCode, Json<Resp>) {
    let prompt = payload.prompt;
    println!("Prompt: {:?}", prompt);
    let tokenizer = &state.tokenizer;
    let rag_matrix = &state.rag_matrix; // N x context_window
    println!("RAG Matrix shape: {:?}", rag_matrix.shape());
    // Prompt embedding
    let prompt_encoding = tokenizer.encode(prompt, false).unwrap();
    let prompt_tokens = prompt_encoding.get_tokens().to_vec();
    let raw_embedding = prompt_encoding.get_ids().iter().map(|&x| x as f64).collect();
    let prompt_embedding = Array::from_vec(raw_embedding); // 1 x context_window
    // Similarity
    let similarities = rag_matrix.dot(&prompt_embedding); // N x 1
    println!("Similarities: {:?}", similarities);
    // Find most similar RAG entry
    let max_index = similarities.argmax().unwrap();
    let max_vector = rag_matrix.index_axis(Axis(0), max_index);
    // Decode
    // let decoded = tokenizer.decode(max_vector, true);
    // println!("Text with highest similarity: {:?}", decoded);


    // Example response
    let resp = Resp {
        body: "Response".to_string()
    };

    (StatusCode::OK, Json(resp))
}
