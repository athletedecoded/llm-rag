use std::fs;
use polodb_core::{Collection, Database};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tokenizers::tokenizer::{Result, Tokenizer};
use tokenizers::utils::truncation::{TruncationParams, TruncationStrategy};
use tokenizers::utils::padding::{PaddingStrategy, PaddingDirection, PaddingParams};
use llm_rag::{Context, chunk_text, create_context};

// Seed db with corpus embeddings
fn main() -> Result<()> {
    // Load env variables
    dotenv::dotenv().ok();
    let db_pth = dotenv::var("DB_PTH").expect("ERROR: Invalid DB_PTH");
    let corpus_pth = dotenv::var("CORPUS_PTH").expect("ERROR: Invalid CORPUS_PTH");
    let tokenizer_pth = dotenv::var("TOKENIZER_PTH").expect("ERROR: Invalid TOKENIZER_PTH");
    let context_window = dotenv::var("CONTEXT_WINDOW").expect("ERROR: Invalid CONTEXT_WINDOW");
    // Start with fresh db
    if fs::metadata(&db_pth).is_ok() {
        // Delete existing db file if exists
        match fs::remove_file(&db_pth) {
            Ok(_) => println!("Success: Existing db removed."),
            Err(err) => eprintln!("Error: Failed to delete existing database file: {:?}", err),
        }
    }
    // Init new db
    let db = Database::open_file(&db_pth)?;
    let collection: Collection<Context> = db.collection("context");
    println!("Success: New db created.");
    // Init tokenizer
    let mut tokenizer = Tokenizer::from_file(&tokenizer_pth)?;
    let padding_params = PaddingParams {
        strategy: PaddingStrategy::Fixed(60),
        // direction: PaddingDirection::Right,
        ..Default::default()
    };
    let truncation_params = TruncationParams {
        max_length: 60,
        strategy: TruncationStrategy::LongestFirst,
        stride: 0,
        ..Default::default()
    };
    let _ = tokenizer.with_padding(Some(padding_params)).with_truncation(Some(truncation_params));
    // Scan corpus
    println!("Scanning corpus...");
    let docs: Vec<_> = fs::read_dir(corpus_pth)?
        .filter_map(|doc| {
            if let Ok(doc) = doc {
                if doc.path().is_file() {
                    Some(doc.path())
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();
    println!("Found {:?} corpus docs: {:?}", docs.len(), docs);
    // Process corpus docs in parallel
    println!("Seeding DB...");
    docs.par_iter().for_each(|file_path| {
        // Chunk the file content
        let content = fs::read_to_string(&file_path).unwrap();
        let chunks = chunk_text(&content);
        // Embed and insert into DB
        for chunk in chunks {
            match create_context(&tokenizer, chunk) {
                Ok(context) => {
                    let resp = collection.insert_one(context);
                }
                Err(err) => {
                    eprintln!("Error creating context: {:?}", err);
                }
            }
        }
    });

    Ok(())
}
