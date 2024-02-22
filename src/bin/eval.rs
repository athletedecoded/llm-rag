use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::Ollama;
use tokio;

#[derive(Debug, Serialize, Deserialize)]
struct GroundTruth {
    question: String,
    answer: String,
}
#[derive(Debug, Serialize, Deserialize)]
struct TestResults {
    question: String,
    response: TestResponse,
}
#[derive(Debug, Serialize, Deserialize)]
struct TestResponse {
    ollama: String,
    rag: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct RagQuery {
    pub prompt: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct RagResp {
    pub body: String,
}

#[tokio::main]
async fn main() {
    // Load env vars
    dotenv::dotenv().ok();
    let model = dotenv::var("MODEL").expect("ERROR: Invalid MODEL");
    // Load test set
    let test_set: Vec<GroundTruth> = read_test_set("src/bin/eval.json");
    // Initialize clients
    let ollama_client = Ollama::default();
    let rag_client = reqwest::Client::new();
    // Store responses
    let mut test_responses: Vec<TestResults> = Vec::new();
    // Iterate through each question in the test set
    for entry in test_set.iter() {
        println!("Question: {}", entry.question);
        // Query Ollama model (w/out RAG)
        let test_query = format!("Answer the following question as concisely as possible: {}", entry.question);
        let ollama_response = ollama_client.generate(GenerationRequest::new(model.to_string(), test_query.clone())).await.unwrap().response;
        // Query RAG service
        let rag_query = RagQuery{prompt: test_query.clone()};
        let rag_response = rag_client
            .post("http://127.0.0.1:8000/query")
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .json(&rag_query)
            .send()
            .await
            .unwrap();
        // Store results
        let test_result = TestResults {
            question: entry.question.clone(),
            response: TestResponse {
                ollama: ollama_response,
                rag: "rag_response".to_string(), // Placeholder for rag_response
            },
        };

        test_responses.push(test_result);
    }

    // Ask strong learner to evaluate performance
    println!("Evaluating reposnses...");
    // let mut ollama_score = 0;
    // let mut rag_score = 0;
    // let mut errors = 0;
    let eval_model = "llama2";
    for entry in test_responses.iter() {
        let eval_query = format!("For the given question and two possible answers 'O' or 'R'. Which gives the best answer to the question? STRICTLY respond with the letter 'O' or 'R' only. Question: {}. Answer O: {}. Answer R: {}", entry.question, entry.response.ollama, entry.response.rag);
        let eval_reponse = ollama_client.generate(GenerationRequest::new(eval_model.to_string(), eval_query.clone())).await.unwrap().response;
        println!("Eval Response: {}", eval_reponse.trim());
        // Match results
        // if eval_reponse.trim().contains("O") {
        //     ollama_score += 1
        // } else if eval_reponse.trim().contains("R") {
        //     rag_score += 1
        // } else {
        //     errors += 1
        // }
    }
    // Print scores
    // let total = ollama_score + rag_score;
    // println!("Ollama Score: {}", ollama_score/total);
    // println!("RAG Score: {}", rag_score/total);
    // println!("Errors: {}", errors);

}

fn read_test_set(file_path: &str) -> Vec<GroundTruth> {
    // Read the test set from the file
    let mut file = File::open(file_path).expect("Failed to open file");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Failed to read file");
    // Parse the JSON contents
    let test_set: Vec<GroundTruth> = serde_json::from_str(&contents).expect("Failed to parse JSON");

    test_set
}

async fn query_ollama(client: &Ollama, model: &str, prompt: String) -> String {
    let res = client.generate(GenerationRequest::new(model.to_string(), prompt)).await;
    match res {
        Ok(res) => {
            println!("Ollama Response: {:?}", res.response);
            res.response
        }
        Err(e) => {
            println!("Error: {:?}", e);
            e
        }
    }
}
