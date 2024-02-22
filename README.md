# Rusty RAG

LLM w/ RAG from scratch using Rust + [Ollama](https://github.com/ollama/ollama/tree/main) + [PoloDB](https://github.com/PoloDB/PoloDB)

**Setup**

```
# Install ollama
$ make ollama

# Get model and tokenizer files
$ make models
```

**Seed RAG Database**

```
$ cargo run --bin seed
```

**Scan RAG Database**

```
$ cargo run --bin scan
```

**Launch**

```
$ cargo run
```

Navigate to http://127.0.0.1:8000/

**Evaluation**

```
$ cargo run --bin eval
```

**Run Binaries**

```
$ make release
$ cd target/release
$ ./seed # seed DB
$ ./scan # scan DB
$ ./llm-rag # run RAG service @ http://127.0.0.1:8000/
$ ./eval # evaluate
```

## ToDos:

- [ ] Debug RAG retrieval
- [ ] Refactor for binary build
- [ ] Error handling and refactoring