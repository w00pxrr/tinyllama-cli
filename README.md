# TinyLlama GUI

A modern GUI chat application powered by **local** language models with buttons for rating, mode selection, and settings.

## Table of Contents

1. [Features](#features)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
   - [Quick Start](#quick-start-1)
   - [Manual Setup](#manual-setup)
   - [Bootstrap Script](#bootstrap-script)
4. [Usage](docs/usage.md)
5. [Model Download](docs/model-download.md)
   - [Available Models](#available-models)
   - [Custom Models](#custom-models)
   - [HuggingFace Token](#huggingface-token)
6. [Configuration](docs/configuration.md)
7. [Advanced Features](docs/advanced-features.md)
   - [Prompt Tuning](#prompt-tuning)
   - [Training Data Export](#training-data-export)
   - [Transcript Saving](#transcript-saving)
8. [API Reference](docs/api-reference.md)
9. [Troubleshooting](docs/troubleshooting.md)

---

## Features

- **GUI Interface**: Modern native GUI with clickable buttons
- **Rate Responses**: Like/Dislike buttons to provide feedback
- **Mode Selection**: Toggle between questions, code, academic, math modes
- **Settings Panel**: Adjust temperature, top_p, top_k, max_tokens
- **Local Inference**: Run models locally on your machine
- **Training Data Export**: Automatic export to JSONL format for fine-tuning
- **Transcript Saving**: Save and reload chat histories
- **Multiple Model Support**: Download and use various models from HuggingFace
- **Web Search Integration**: Automatic search for recent information

### GUI Buttons

- **Like/Dislike**: Rate AI responses after they appear
- **Mode**: Click to select AI mode (questions, code, academic, math)
- **Settings**: Configure generation parameters
- **Clear Chat**: Reset the conversation
- **Save Chat**: Export conversation to JSON

---

## Quick Start

```bash
cd /Path/to/tinyllama-cli
./tinyllama.sh
```

The bootstrap script will automatically:
1. Check for Python and install if needed
2. Create a virtual environment (`.venv`)
3. Install dependencies from `requirements.txt`
4. Prompt for HuggingFace token (optional)
5. Download a model (if none is installed)
6. Launch the GUI

After download, your model files are in:

```
models/
├── TinyLlama-1.1B-Chat-v1.0/
└── ...
```

---

## Installation

```bash
# Create virtual environment
python3 -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Set HuggingFace token (optional, for gated models)
export HF_TOKEN="your_huggingface_token_here"

# Download a model
python download_model.py

# Run the GUI
```bash
./tinyllama_gui/target/release/tinyllama_gui
```

Or build and run:
```bash
cd tinyllama_gui
cargo run --release
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download a model
python download_model.py

# Run the GUI
./tinyllama_gui/target/release/tinyllama_gui
```

---

## Available Models

| Key | Model ID | Size | Description |
|-----|----------|------|-------------|
| `tinyllama` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | ~1GB | Lightweight, fast |
| `smollm2` | HuggingFaceTB/SmolLM2-135M | ~270MB | Very small, efficient |
| `qwen` | Qwen/Qwen2.5-0.5B-Instruct | ~1GB | Multilingual |
| `nvidia_nemotron` | nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF | ~4GB | NVIDIA efficient |
| `choose more` | Any HuggingFace model | Varies | Custom model |

### Download Models

```bash
# Interactive picker
python download_model.py

# Direct download
python download_model.py --model tinyllama
python download_model.py --model nvidia_nemotron

# Any HuggingFace model
python download_model.py --model meta/Llama-3-8B
```

---

## Tuning Strategy

The CLI auto-optimizes generation settings per prompt:

| Prompt Type | Temperature | Top-P | Max Tokens |
|------------|-------------|-------|------------|
| Factual | 0.45 | 0.82 | 220 |
| Code | 0.40 | 0.85 | 280 |
| Creative | 0.88 | 0.95 | 320 |
| Math | 0.00 | 1.00 | 96 |

You can inspect settings with `/settings`.

---

## Training Data Export

Every time you use `/save` or exit the app, it exports:

```json
{
  "id": "conv_2024-01-15_14-30-22",
  "source_transcript": "2024-01-15_143022.json",
  "created_at": "2024-01-15T14:30:22Z",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Saved to: `training_data/tinyllama_sft.jsonl`

---

## Documentation

For detailed documentation, see the [docs/](docs/) folder:

- [Installation Guide](docs/installation.md) - Full setup instructions
- [Usage Guide](docs/usage.md) - Chat CLI commands
- [Model Download](docs/model-download.md) - Download options
- [Configuration](docs/configuration.md) - Environment setup
- [Advanced Features](docs/advanced-features.md) - Prompt tuning, export
- [API Reference](docs/api-reference.md) - CLI and Python API
- [Troubleshooting](docs/troubleshooting.md) - Common issues

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | macOS, Linux, Windows (WSL) | macOS 12+, Ubuntu 20.04+ |
| Python | 3.10+ | 3.11+ |
| RAM | 8GB | 12GB+ |
| Storage | 2GB | 10GB+ |
| GPU | Optional | NVIDIA with CUDA |

---

## License

See [LICENSE](LICENSE) file for details.
