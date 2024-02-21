use axum::{extract::State, http::StatusCode, response::Html, Json};
use ndarray::{s, Array, ArrayBase, Axis, Dim, OwnedRepr};
use ndarray_linalg::norm::{normalize, Norm, NormalizeAxis};
use ndarray_stats::QuantileExt;
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::Ollama;
use polodb_core::Collection;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use regex::Regex;
use tokenizers::tokenizer::{Result, Tokenizer};

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
    pub model: String,
    pub rag_matrix: ArrayBase<OwnedRepr<u32>, Dim<[usize; 2]>>,
}

// HELPER FXNS
pub fn chunk_text(tokenizer_pth: &str, text: &str, max_tokens: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_token_count = 0;

    // Preprocess
    let cleaned_text = text.replace("\n", " ").replace("\r", " ");
    let sentences: Vec<&str> = cleaned_text
        .split_terminator(|c| c == '.' || c == '?' || c == '!')
        .collect();

    let tokenizer = Tokenizer::from_file(&tokenizer_pth).unwrap();

    for sentence in sentences {
        let encoding = tokenizer.encode(sentence, false).unwrap();
        let num_tokens = encoding.get_tokens().to_vec().len();
        // Skip cases where the number of tokens is zero
        if num_tokens == 0 {
            continue;
        }
        // If adding sentence will exceed max tokens, start fresh chunk
        if current_token_count + num_tokens > max_tokens {
            chunks.push(current_chunk.clone());
            current_chunk.clear();
            current_token_count = 0;
        }
        // Add sentence to chunk
        if !current_chunk.is_empty() {
            // If current chunk not empty add a space
            current_chunk.push_str(" ");
        }
        current_chunk.push_str(sentence.trim());
        current_chunk.push_str(".");
        // Increment token count
        current_token_count += num_tokens + 1;
    }
    // Return chunks
    chunks
}

pub fn create_context(tokenizer: &Tokenizer, text: String) -> Result<Context> {
    let encoding = tokenizer.encode(text.clone(), false)?;
    let tokens = encoding.get_tokens().to_vec();
    let embedding = encoding.get_ids().to_vec();
    let context = Context {
        text,
        tokens,
        embedding,
    };
    Ok(context)
}

pub fn build_rag_matrix(
    collection: &Collection<Context>,
    context_window: usize,
) -> ArrayBase<OwnedRepr<u32>, Dim<[usize; 2]>> {
    println!("Building RAG matrix...");
    let mut rag_matrix = Array::zeros((0, context_window));
    let entries = collection.find(None).unwrap();
    for entry in entries {
        let embedding = Array::from_vec(entry.unwrap().embedding);
        let _ = rag_matrix.push(Axis(0), (&embedding).into());
    }
    println!("RAG matrix shape: {:?}", rag_matrix.shape());
    rag_matrix
}

fn normalize_embedding(raw_embedding: Vec<u32>) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>> {
    // Convert u32 to f32 for normalization
    let f32_vec: Vec<f32> = raw_embedding.iter().map(|&x| x as f32).collect();
    let mut embedding = Array::from_vec(f32_vec);
    // L2 norm
    let norm = embedding.norm_l2();
    if norm == 0.0 {
        return embedding;
    }
    // Normalize the embedding
    embedding.mapv_inplace(|x| x / norm);

    // Validate normalized vector length
    // let length = (embedding.mapv(|x| x.powi(2)).sum()).sqrt();
    // println!("Normalized Vector Length: {}", length);

    embedding
}

fn normalize_matrix(
    matrix: &ArrayBase<OwnedRepr<u32>, Dim<[usize; 2]>>,
) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    // Convert u32 to f32 for normalization
    let f32_matrix: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> = matrix.map(|&x| x as f32);
    // L2 Norm
    let (norm_matrix, _) = normalize(f32_matrix, NormalizeAxis::Row);

    // Validate the length of the normalized vector for each row
    // for i in 0..norm_matrix.shape()[0] {
    //     let row_length = (norm_matrix.slice(s![i, ..]).mapv(|x| x.powi(2)).sum()).sqrt();
    //     println!("Length of normalized vector for row {}: {}", i, row_length);
    // }

    norm_matrix
}

fn cosine_similarity(
    raw_embedding: Vec<u32>, rag_matrix: &ArrayBase<OwnedRepr<u32>, Dim<[usize; 2]>>
    ) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>> {
    // Normalize
    let prompt_norm = normalize_embedding(raw_embedding);
    let rag_norm = normalize_matrix(rag_matrix);
    // cosine similarity: dot product or normalized vectors
    rag_norm.dot(&prompt_norm)
}

// ROUTES
// GET /
pub async fn index() -> Html<&'static str> {
    Html(std::include_str!("index.html"))
}

// GET /query
pub async fn query_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<RagQuery>,
) -> (StatusCode, Json<RagResp>) {
    let prompt = payload.prompt;
    println!("Prompt: {:?}", prompt);
    // Extract AppState
    let tokenizer = &state.tokenizer;
    let model = &state.model;
    let rag_matrix = &state.rag_matrix; // N x context_window
    // Prompt embedding
    let prompt_encoding = tokenizer.encode(prompt.clone(), false).unwrap();
    let raw_embedding = prompt_encoding.get_ids().to_vec();
    println!("Prompt embedding length: {:?}", raw_embedding.len());
    // Cosine Similarity
    let similarities = cosine_similarity(raw_embedding, &rag_matrix);
    // Find most similar RAG entry
    let max_index = similarities.argmax().unwrap();
    let rag_match = rag_matrix.index_axis(Axis(0), max_index);
    // Extract context
    let context = tokenizer
        .decode(rag_match.to_slice().unwrap(), true)
        .unwrap();
    println!("Context: {:?}", &context);
    // Query LLM
    let query_string = format!("Generate a response to the following question. Use the provided context only if it is useful. \n Question: {} \n Context: {}.", &prompt, &context);
    let ollama = Ollama::default();
    let res = ollama
        .generate(GenerationRequest::new(model.to_string(), query_string))
        .await;
    match res {
        // 200 OK
        Ok(res) => {
            println!("Response: {:?}", res.response);
            (StatusCode::OK, Json(RagResp { body: res.response }))
        }
        // 500 Error
        Err(e) => {
            println!("Error: {:?}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(RagResp { body: e }))
        }
    }
}
