use std::sync::Arc;
use serde_json::json;
use ollama_rs::Ollama;
use tokio::sync::Mutex;
use ndarray_stats::QuantileExt;
use serde::{Deserialize, Serialize};
use polodb_core::{Collection, Database};
use tokenizers::tokenizer::{Result, Tokenizer};
use ndarray::{Array, Axis, ArrayView, ArrayBase, Dim, OwnedRepr};
use axum::{http::StatusCode, Json, Router, response::Html, routing::get, extract::State};
use ollama_rs::generation::completion::request::GenerationRequest;

// STRUCTS
#[derive(Debug, Serialize, Deserialize)]
pub struct Context {
    pub text: String,
    pub tokens: Vec<String>,
    pub embedding: Vec<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RagQuery {
    pub prompt: String,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct RagResp {
    pub body: String,
}

pub struct AppState {
    pub tokenizer: Tokenizer,
    pub rag_matrix: ArrayBase<OwnedRepr<u32>, Dim<[usize; 2]>>
}

// HELPER FXNS
pub fn chunk_text(text: &str) -> Vec<&str> {
    text.split("\n").collect()
}

pub fn create_context(tokenizer: &Tokenizer, text: &str) -> Result<Context> {
    let encoding = tokenizer.encode(text, false)?;
    let tokens = encoding.get_tokens().to_vec();
    let embedding = encoding.get_ids().to_vec();
    // let embedding = encoding.get_ids().iter().map(|&x| x as f64).collect();
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

pub fn build_rag_matrix(collection: &Collection<Context>, context_window: usize) -> ArrayBase<OwnedRepr<u32>, Dim<[usize; 2]>>{
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
// pub async fn root() -> &'static str {
//     "Rusty RAG Application Launched!"
// }
// Include utf-8 file at **compile** time.
pub async fn index() -> Html<&'static str> {
    Html(std::include_str!("index.html"))
}

// GET /query
pub async fn query_handler(State(state): State<Arc<AppState>>, Json(payload): Json<RagQuery>) -> (StatusCode, Json<RagResp>) {
    let prompt = payload.prompt;
    println!("Prompt: {:?}", prompt);
    let tokenizer = &state.tokenizer;
    let rag_matrix = &state.rag_matrix; // N x context_window
    println!("RAG Matrix shape: {:?}", rag_matrix.shape());
    // Prompt embedding
    let prompt_encoding = tokenizer.encode(prompt.clone(), false).unwrap();
    let prompt_tokens = prompt_encoding.get_tokens().to_vec();
    let raw_embedding = prompt_encoding.get_ids().to_vec();
    let prompt_embedding = Array::from_vec(raw_embedding); // 1 x context_window
    // Similarity
    let similarities = rag_matrix.dot(&prompt_embedding); // N x 1
    println!("Similarities: {:?}", similarities);
    // Find most similar RAG entry
    let max_index = similarities.argmax().unwrap();
    let max_vector = rag_matrix.index_axis(Axis(0), max_index);
    // Decode
    let decoded = tokenizer.decode(max_vector.to_slice().unwrap(), true).unwrap();
    println!("Text with highest similarity: {:?}", &decoded);
    // Query LLM
    let query_string = format!("Generate a response to the following question. Use the provided context only if it is useful. \n Question: {} \n Context: {}.", &prompt, &decoded);
    println!("Query string: {:?}", &query_string);
    let ollama = Ollama::default();
    let model = "mistral:latest".to_string();
    let res = ollama.generate(GenerationRequest::new(model, query_string)).await;
    match res {
        // 200 OK
        Ok(res) => {
            println!("Response: {:?}", res.response);
            (StatusCode::OK, Json(RagResp {
                body: res.response
            }))
        }
        // 500 Error
        Err(e) => {
            println!("Error: {:?}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(RagResp {
                body: e
            }))
        }
    }
}
