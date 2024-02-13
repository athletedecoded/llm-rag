use polodb_core::Database;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
// use rust_bert::pipelines::common::Config;
// use rust_bert::pipelines::text::TextEmbeddingModel;

#[derive(Debug, Serialize, Deserialize)]
struct Context {
    chunk: String,
    embedding: Vec<f64>, // Change the type to store the embedding as a vector of floats
}

// Seed db with corpus embeddings
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load env variables
    dotenv::dotenv().ok();
    let db_pth = dotenv::var("DB_PTH").expect("ERROR: Invalid DB_PTH");
    let corpus_pth = dotenv::var("CORPUS_PTH").expect("ERROR: Invalid CORPUS_PTH");
    // DB config
    let db = Database::open_file(&db_pth)?;
    let collection = db.collection("context");

    // Initialize the BERT model for text embedding
    // let config = Config::new();
    // let text_embedding_model = TextEmbeddingModel::new(config);

    // Process corpus in parallel
    let entries: Vec<_> = fs::read_dir(corpus_pth)?
        .filter_map(|entry| {
            if let Ok(entry) = entry {
                if entry.path().is_file() {
                    Some(entry.path())
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    println!("{:?}", entries);

    entries.par_iter().for_each(|file_path| {
        // Read the file content
        let content = fs::read_to_string(&file_path).unwrap();
        // Chunk
        let chunks = chunk_text(&content);
        // Embed and insert into DB
        for chunk in chunks {
            // let embedding = embed_text(&text_embedding_model, &chunk);
            let embedding = generate_random_embedding();
            // Insert into DB
            collection.insert_one(Context {
                chunk: chunk.to_string(),
                embedding,
            });
        }
    });

    Ok(())
}

fn chunk_text(text: &str) -> Vec<&str> {
    // Implement your chunking logic here
    text.split('\n').collect()
}

// fn embed_text(model: &TextEmbeddingModel, text: &str) -> Vec<f64> {
//     model.encode(text)
// }

fn generate_random_embedding() -> Vec<f64> {
    // Placeholder for a random sample embedding
    // Modify this as needed for your testing
    (0..100).map(|_| rand::random::<f64>()).collect()
}
