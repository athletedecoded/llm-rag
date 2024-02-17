install:
	wget https://huggingface.co/microsoft/phi-2/resolve/main/tokenizer.json?download=true -O ./tokenizers/phi.json && \
	wget https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer.json?download=true -O ./tokenizers/mistral.json

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