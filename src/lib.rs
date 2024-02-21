use std::sync::Arc;
use serde_json::json;
use ollama_rs::Ollama;
use tokio::sync::Mutex;
use ndarray_linalg::norm::{normalize, NormalizeAxis};
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
pub fn chunk_text(text: &str, max_words: usize) -> Vec<String> {
    let sentences: Vec<&str> = text.split_terminator(|c| c == '.' || c == '?' || c == '!').collect();

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_word_count = 0;

    for sentence in sentences {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        let sentence_word_count = words.len();

        if current_word_count + sentence_word_count > max_words && !current_chunk.is_empty() {
            chunks.push(current_chunk.clone());
            current_chunk.clear();
            current_word_count = 0;
        }

        if !current_chunk.is_empty() {
            current_chunk.push_str(" ");
        }
        current_chunk.push_str(sentence.trim());
        current_word_count += sentence_word_count;
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    chunks
}

pub fn create_context(tokenizer: &Tokenizer, text: String) -> Result<Context> {
    let encoding = tokenizer.encode(text.clone(), false)?;
    let tokens = encoding.get_tokens().to_vec();
    let embedding = encoding.get_ids().to_vec();
    let context = Context {
        text,
        tokens,
        embedding
    };
    Ok(context)
}

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

fn normalize_embedding(raw_embedding: Vec<u32>) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>> {
    // Convert u32 to f32 for normalization
    let f32_vec: Vec<f32> = raw_embedding.iter().map(|&x| x as f32).collect();
    let mut embedding = Array::from_vec(f32_vec);
    // L2 norm
    let norm = embedding.iter().fold(0.0, |acc, &x| acc + x * x).sqrt();
    embedding.mapv_inplace(|x| x / norm);

    embedding
}

fn normalize_matrix(matrix: &ArrayBase<OwnedRepr<u32>, Dim<[usize; 2]>>) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    // Convert u32 to f32 for normalization
    let f32_matrix: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> = matrix.map(|&x| x as f32);
    // L2 Norm
    let (norm_matrix, _) = normalize(f32_matrix, NormalizeAxis::Row);

    norm_matrix
}

// ROUTES
// GET /
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
    // let prompt_embedding = Array::from_vec(raw_embedding); // 1 x context_window
    // Normalize
    let prompt_norm = normalize_embedding(raw_embedding);
    let rag_norm = normalize_matrix(rag_matrix);
    // Similarity
    let similarities = rag_norm.dot(&prompt_norm); // N x 1
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
