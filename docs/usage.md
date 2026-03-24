# Usage Guide

This guide explains how to use the TinyLlama CLI for chatting with local language models.

## Starting the CLI

### Quick Start

After installation, simply run:

```bash
./tinyllama.sh
```

This will start the interactive chat interface.

### Manual Start

```bash
# Activate virtual environment (if not using tinyllama.sh)
source .venv/bin/activate

# Start the CLI
python ai_cli.py
```

---

## Model Selection

When you start the CLI, it will show you a list of installed models:

```
╔════════════════════════════════════════════════════════════════╗
║                      Model Picker                                ║
╠════════════════════════════════════════════════════════════════╣
║  Choose an installed model to launch                            ║
║  Each folder under ./models with a config.json is listed below. ║
║  Use [A] for Auto mode - selects based on your query.           ║
║                                                                    ║
║  #   Installed Model              Path                          ║
║  ─────────────────────────────────────────────────────────────  ║
║  A   Auto                       Smart selection based on task    ║
║  1   TinyLlama-1.1B-Chat-v1.0   ./models/TinyLlama-1.1B...      ║
║  2   NVIDIA-Nemotron-3-Nano     ./models/NVIDIA-Nemotron...    ║
╚════════════════════════════════════════════════════════════════╝
```

### Auto Mode

Press `A` to enable Auto mode. The CLI will automatically select the best model based on your query:
- Factual questions → smaller, factual models
- Code generation → models good at code
- Creative tasks → models with better creativity

### Direct Model Selection

```bash
# Select a specific model when starting
python ai_cli.py --model tinyllama
python ai_cli.py --model NVIDIA-Nemotron-3-Nano-4B-GGUF
python ai_cli.py --model auto  # Smart selection
```

---

## Chat Interface

Once a model is selected, you'll see the chat interface:

```
╔════════════════════════════════════════════════════════════════╗
║                      TinyLlama CLI                              ║
║                     Local Chat                                  ║
╠════════════════════════════════════════════════════════════════╣
║  Try /help  /settings  /save  /exit                              ║
║                                                                    ║
║  Model: TinyLlama-1.1B-Chat-v1.0 (GPU)                          ║
╚════════════════════════════════════════════════════════════════╝
```

### Sending Messages

Simply type your message and press Enter. The model will respond with generated text.

Example:

```
You: What is Python?

TinyLlama: Python is a high-level, interpreted programming language...
```

### Markdown Support

The CLI supports markdown rendering for model responses:

- **Bold text** renders as bold
- *Italic text* renders as italic
- `code` renders in monospace
- Lists render properly
- Code blocks are highlighted

---

## Chat Commands

The CLI provides several built-in commands:

### /help

Shows all available commands and their descriptions.

```
/help
```

### /settings

Display and modify generation parameters:

```
/settings
```

Shows a panel with current settings:

| Setting | Value |
|---------|-------|
| temperature | 0.65 |
| top_p | 0.9 |
| top_k | 40 |
| repetition_penalty | 1.1 |
| max_new_tokens | 256 |
| do_sample | True |

### /clear

Clears the chat history (starts a fresh conversation).

```
/clear
```

### /save

Saves the current transcript to `transcripts/` and exports training data.

```
/save
```

### /exit

Exits the CLI and saves the transcript automatically.

```
/exit
```

---

## Settings

The CLI automatically tunes generation settings based on your prompts:

### Automatic Tuning

| Prompt Type | Temperature | Top-P | Max Tokens |
|------------|-------------|-------|------------|
| Factual | 0.45 | 0.82 | 220 |
| Code | 0.40 | 0.85 | 280 |
| Creative | 0.88 | 0.95 | 320 |
| Math | 0.00 | 1.00 | 96 |
| Long Context | 0.55 | 0.86 | 192 |

### Manual Settings

You can modify settings by editing the code in `ai_cli.py`:

```python
@dataclass
class GenerationConfig:
    temperature: float = 0.65
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    max_new_tokens: int = 256
    do_sample: bool = True
```

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Enter | Send message |
| Ctrl+C | Interrupt generation |
| Ctrl+L | Clear screen |
| Ctrl+D | Exit (same as /exit) |

---

## Transcript and Training Data

### Transcript Saving

Chat transcripts are saved to:

```
transcripts/
├── 2024-01-15_143022.json
├── 2024-01-16_091545.json
└── ...
```

Each transcript contains:
- Full chat history
- Model used
- Timestamp
- Generation settings

### Training Data Export

Every time you use `/save` or `/exit`, the CLI exports training data:

```
training_data/tinyllama_sft.jsonl
```

Each line is a JSON object with:
- `id`: Unique identifier
- `source_transcript`: Transcript file name
- `created_at`: ISO timestamp
- `messages`: Chat messages array

---

## Tips for Better Responses

1. **Be specific**: "Explain how Python decorators work with examples" works better than "Explain decorators"

2. **Use prefixes**: The model responds better to clear role definitions:
   - "As a Python expert, explain..."
   - "Write a haiku about..."

3. **Break down complex requests**: Split into multiple messages for better results

4. **Check settings**: Use `/settings` to verify generation parameters

---

## Next Steps

- [Model Download](model-download.md) - Download more models
- [Configuration](configuration.md) - Customize your setup
- [Advanced Features](advanced-features.md) - Learn about prompt tuning
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
