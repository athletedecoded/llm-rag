use std::sync::Arc;
use std::process::Command;
use env_logger::Env;
use polodb_core::{Collection, Database};
use tokenizers::tokenizer::Tokenizer;
use tokenizers::utils::truncation::{TruncationParams, TruncationStrategy};
use tokenizers::utils::padding::{PaddingParams, PaddingStrategy, PaddingDirection};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use axum::{Router, routing::{get, post}, extract::State};

use llm_rag::{Context, RagQuery, RagResp, AppState, build_rag_matrix};
#[tokio::main]
async fn main() {
    // Initialize logger
    env_logger::init_from_env(Env::default().default_filter_or("info"));
    // Load env variables
    dotenv::dotenv().ok();
    let db_pth = dotenv::var("DB_PTH").expect("ERROR: Invalid DB_PTH");
    let tokenizer_pth = dotenv::var("TOKENIZER_PTH").expect("ERROR: Invalid TOKENIZER_PTH");
    let context_window = dotenv::var("CONTEXT_WINDOW").expect("ERROR: Invalid CONTEXT_WINDOW").parse::<usize>().unwrap();
    // Init DB
    let db = Database::open_file(&db_pth).unwrap();
    let collection: Collection<Context> = db.collection("context");
    // Init tokenizer
    let mut tokenizer = Tokenizer::from_file(&tokenizer_pth).unwrap();
    let padding_params = PaddingParams {
        strategy: PaddingStrategy::Fixed(context_window),
        direction: PaddingDirection::Right,
        ..Default::default()
    };
    let truncation_params = TruncationParams {
        max_length: context_window,
        strategy: TruncationStrategy::LongestFirst,
        stride: 0,
        ..Default::default()
    };
    let _ = tokenizer.with_padding(Some(padding_params)).with_truncation(Some(truncation_params));
    // Build RAG matrix
    let rag_matrix = build_rag_matrix(&collection, context_window);
    // Init app state
    let app_state = Arc::new(AppState{tokenizer, rag_matrix});
    // Build app routes
    let app = Router::new()
        .route("/", get(llm_rag::index))
        .route("/query", post(llm_rag::query_handler))
        .with_state(app_state);
    // Launch
    let listener = TcpListener::bind("127.0.0.1:8000")
        .await
        .unwrap();
    println!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}