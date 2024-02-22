use env_logger::Env;
use llm_rag::Context;
use serde::{Deserialize, Serialize};
use polodb_core::{Database, Error, Collection};

fn main() {
    // Load env variables
    dotenv::dotenv().ok();
    let db_pth = dotenv::var("DB_PTH").expect("ERROR: Invalid DB_PTH");
    // Scan DB
    let db = Database::open_file(&db_pth).unwrap();
    let collection: Collection<Context> = db.collection("context");
    let entries = collection.find(None).unwrap();
    for entry in entries {
        println!("item: {:?}", entry.unwrap().text);
    }
}