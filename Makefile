ollama:
	sudo curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/bin/ollama && \
    sudo chmod +x /usr/bin/ollama

models:
	mkdir tokenizers && \
	wget https://huggingface.co/microsoft/phi-2/resolve/main/tokenizer.json?download=true -O ./tokenizers/phi.json && \
	wget https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer.json?download=true -O ./tokenizers/mistral.json && \
#	wget https://huggingface.co/google/gemma-2b/resolve/main/tokenizer.json?download=true -O ./tokenizers/gemma.json && \
	ollama pull mistral && \
	ollama pull phi && \
	ollama pull gemma

format:
	cargo fmt --quiet

lint:
	cargo clippy --quiet

test:
	cargo test --quiet

run:
	cargo run

release:
	cargo build --release