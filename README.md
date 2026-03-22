# TinyLlama Fancy CLI (Local)

A polished terminal chat app powered by a **local** model:
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `transformers` + `torch`
- Rich terminal UI

## Features

- Fancy terminal panels, markdown rendering, and status spinner
- Local inference from `models/TinyLlama-1.1B-Chat-v1.0`
- Runtime prompt-aware tuning for better TinyLlama responses
- Built-in commands: `/help`, `/settings`, `/clear`, `/save`, `/exit`
- Transcript saving to `transcripts/`
- Automatic training-data export to `training_data/tinyllama_sft.jsonl` on save/exit

## Quick Start

```bash
cd /Path/to/tinyllama-cli
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN="your_huggingface_token_here"
python download_model.py
# or: python download_model.py --model tinyllama|smollm2|smollm3
python chat.py
```

Use the bootstrap script to install dependencies and launch the CLI:

```bash
./tinyllama.sh
```

After download, your model files are in:

```text
models/TinyLlama-1.1B-Chat-v1.0/
```

## Tuning Strategy

The CLI auto-optimizes generation settings per prompt:
- Factual questions: lower temperature/top-p for stability
- Code/debug prompts: lower randomness + longer max tokens
- Creative prompts: higher temperature/top-p for variety
- Long chats: tighter responses and stronger repetition penalty

You can inspect the latest active settings with `/settings`.

## Training Data Output

Every time you use `/save` or exit the app, it appends examples to:

```text
training_data/tinyllama_sft.jsonl
```

Each JSONL record includes:
- `id`
- `source_transcript`
- `created_at`
- `messages` (`system`, `user`, `assistant`)
