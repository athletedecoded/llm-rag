use actix_web::{web, App, HttpServer};
use env_logger::Env;
use serde::{Deserialize, Serialize};
use polodb_core::{Database, Error, Collection};

#[derive(Debug, Serialize, Deserialize)]
struct Context {
    chunk: String,
    embedding: Vec<f64>, // Change the type to store the embedding as a vector of floats
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize logger
    env_logger::init_from_env(Env::default().default_filter_or("info"));
    // Load env variables
    dotenv::dotenv().ok();
    let db_pth = dotenv::var("DB_PTH").expect("ERROR: Invalid DB_PTH");
    // Scan DB
    let db = Database::open_file(&db_pth).unwrap();
    let collection: Collection<Context> = db.collection("context");
    let chunks = collection.find(None).unwrap();
    for chunk in chunks {
        println!("item: {:?}", chunk);
    }

    // Launch service
    HttpServer::new(|| {
        App::new().service(web::resource("/query").route(web::get().to(query_handler)))
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}

async fn query_handler() -> &'static str {
    // Implement your query handling logic here
    "Hello, world!"
}
