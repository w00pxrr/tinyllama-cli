# API Reference

This document provides complete reference for all CLI commands, scripts, and Python APIs.

## CLI Commands

### download_model.py

Downloads models from HuggingFace Hub.

```bash
python download_model.py [OPTIONS]
```

#### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--model KEY` | Model key or custom model ID | `--model tinyllama` |

#### Model Keys

- `tinyllama` - TinyLlama/TinyLlama-1.1B-Chat-v1.0
- `smollm2` - HuggingFaceTB/SmolLM2-135M
- `qwen` - Qwen/Qwen2.5-0.5B-Instruct
- `nvidia_nemotron` - nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF
- Custom: Any HuggingFace model ID (e.g., `meta/Llama-3-8B`)

#### Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | Error (invalid model, download failed, etc.) |

---

### ai_cli.py

The main chat CLI.

```bash
python ai_cli.py [OPTIONS]
```

#### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--model MODEL` | Model folder name, path, or 'auto' | `--model tinyllama` |

#### Examples

```bash
# Auto-select model based on query
python ai_cli.py --model auto

# Use specific model
python ai_cli.py --model TinyLlama-1.1B-Chat-v1.0

# Use local path
python ai_cli.py --model ./models/my-model/
```

---

### tinyllama.sh

Bootstrap script for automated setup.

```bash
./tinyllama.sh [OPTIONS]
```

#### Options

| Option | Description |
|--------|-------------|
| `--bootstrap-only` | Download deps and model, don't start CLI |
| `--model MODEL` | Auto-download specific model |

#### Examples

```bash
# Full bootstrap + launch
./tinyllama.sh

# Download only
./tinyllama.sh --bootstrap-only

# Download specific model
./tinyllama.sh --model nvidia_nemotron
```

---

## Python API

### TinyLlamaCLI

Main chat interface class.

```python
from ai_cli import TinyLlamaCLI
from pathlib import Path

# Initialize
cli = TinyLlamaCLI(
    model_dir=Path("models/TinyLlama-1.1B-Chat-v1.0"),
    model_label="TinyLlama"
)

# Run the CLI
cli.run()
```

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_dir` | Path | Path to model directory |
| `model_label` | str | Display label for the model |

#### Methods

##### run()

Starts the interactive chat loop.

```python
cli.run()
```

##### _generate_response(user_input: str) -> str

Generates a response to user input.

```python
response = cli._generate_response("Hello!")
```

##### _save_transcript() -> None

Saves the current chat transcript.

```python
cli._save_transcript()
```

##### _export_training_data() -> None

Exports training data in JSONL format.

```python
cli._export_training_data()
```

---

### GenerationConfig

Configuration for text generation.

```python
from ai_cli import GenerationConfig

cfg = GenerationConfig(
    temperature=0.65,
    top_p=0.9,
    top_k=40,
    repetition_penalty=1.1,
    max_new_tokens=256,
    do_sample=True
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 0.65 | Sampling temperature (0=deterministic) |
| `top_p` | float | 0.9 | Nucleus sampling threshold |
| `top_k` | int | 40 | Top-k sampling |
| `repetition_penalty` | float | 1.1 | Repetition penalty |
| `max_new_tokens` | int | 256 | Maximum tokens to generate |
| `do_sample` | bool | True | Use sampling vs greedy |

---

### TinyLlamaOptimizer

Automatic tuning for generation settings.

```python
from ai_cli import TinyLlamaOptimizer

cfg = TinyLlamaOptimizer.tune(
    user_input="Explain Python decorators",
    turns=1
)
```

#### Methods

##### tune(user_input: str, turns: int) -> GenerationConfig

Automatically tunes settings based on input.

```python
cfg = TinyLlamaOptimizer.tune("Write a poem", 0)
```

---

### download_model Functions

#### MODEL_CHOICES

Dictionary of pre-configured models.

```python
from download_model import MODEL_CHOICES

print(MODEL_CHOICES)
# {'tinyllama': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', ...}
```

#### model_dir_for(model_id: str) -> Path

Get the local directory for a model.

```python
from download_model import model_dir_for

path = model_dir_for("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# Returns: Path('models/TinyLlama-1.1B-Chat-v1.0')
```

---

## Module Functions

### parse_args()

Parse command line arguments.

```python
from ai_cli import parse_args

args = parse_args()
# args.model contains the --model value
```

---

## Constants

### DEFAULT_MODEL_ID

```python
from ai_cli import DEFAULT_MODEL_ID
# "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### DEFAULT_MODEL_DIR

```python
from ai_cli import DEFAULT_MODEL_DIR
# Path("models/TinyLlama-1.1B-Chat-v1.0")
```

### SYSTEM_PROMPT

```python
from ai_cli import SYSTEM_PROMPT
# "You are a helpful, concise AI assistant..."
```

---

## Web Search API

### search_web(query: str) -> list[WebResult]

Search the web for information.

```python
from web_search import search_web

results = search_web("Python decorators tutorial")
for result in results:
    print(result.title, result.url)
```

#### Returns

List of WebResult objects:

```python
@dataclass
class WebResult:
    title: str
    url: str
    snippet: str
```

### should_search_web(query: str) -> bool

Determine if a query needs web search.

```python
from web_search import should_search_web

if should_search_web("latest AI news"):
    # Search the web
    pass
```

---

## Data Structures

### ChatMessage

```python
@dataclass
class ChatMessage:
    role: str      # "system", "user", "assistant"
    content: str
    timestamp: str  # ISO 8601
```

### Transcript

```python
@dataclass
class Transcript:
    model: str
    model_path: str
    started_at: str
    settings: dict
    messages: list[ChatMessage]
```

### TrainingDataRecord

```python
@dataclass
class TrainingDataRecord:
    id: str
    source_transcript: str
    created_at: str
    messages: list[dict]  # [{"role": "...", "content": "..."}]
```

---

## Next Steps

- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [Advanced Features](advanced-features.md) - Deep dive into features
- [Model Download](model-download.md) - Model management
