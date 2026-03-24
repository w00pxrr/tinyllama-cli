# Configuration Guide

This guide covers all configuration options for TinyLlama CLI.

## Environment Variables

### Required Variables

#### HF_TOKEN

Your HuggingFace API token (required for gated models):

```bash
# Linux/macOS
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Windows
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Alternative variable name:

```bash
export HUGGINGFACEHUB_API_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Optional Variables

#### MODEL_DIR

Custom directory for storing models (default: `./models`):

```bash
export MODEL_DIR="/path/to/models"
```

#### TRANSCRIPTS_DIR

Custom directory for chat transcripts (default: `./transcripts`):

```bash
export TRANSCRIPTS_DIR="/path/to/transcripts"
```

#### TRAINING_DATA_DIR

Custom directory for training data export (default: `./training_data`):

```bash
export TRAINING_DATA_DIR="/path/to/training_data"
```

---

## HuggingFace Token Setup

### Getting Your Token

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token (if needed)
3. Copy the token

### Token Permissions

For most models, you'll need at least:
- Read access to model repositories

For gated models (like Llama), you'll also need:
- Accepted the model's license on HuggingFace
- Token with appropriate permissions

### Testing Your Token

```bash
# Test authentication
python -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"
```

---

## Model Directory Configuration

### Default Structure

```
tinyllama-cli/
├── models/                    # Default model directory
│   ├── TinyLlama-1.1B-Chat-v1.0/
│   ├── NVIDIA-Nemotron-3-Nano-4B-GGUF/
│   └── ...
├── transcripts/               # Chat transcripts
├── training_data/             # Exported training data
└── ...
```

### Custom Directories

Modify the code to use custom directories:

```python
# In download_model.py
def model_dir_for(model_id: str) -> Path:
    folder_name = model_id.split("/")[-1]
    custom_dir = os.getenv("MODEL_DIR", "models")
    return Path(custom_dir) / folder_name
```

---

## Runtime Configuration

### Generation Settings

The CLI uses automatic tuning, but you can modify defaults in `ai_cli.py`:

```python
@dataclass
class GenerationConfig:
    temperature: float = 0.65      # Randomness (0=deterministic, 1=random)
    top_p: float = 0.9            # Nucleus sampling threshold
    top_k: int = 40                # Top-k sampling
    repetition_penalty: float = 1.1 # Penalize repetition
    max_new_tokens: int = 256      # Maximum tokens to generate
    do_sample: bool = True        # Use sampling vs greedy
```

### Prompt Templates

The CLI uses different prompt templates for different models:

#### TinyLlama Template

```xml
<|system|>
{system_prompt}</s>
<|user|>
{user_message}</s>
<|assistant|>
```

#### Custom Templates

You can modify the prompt template in `ai_cli.py`:

```python
def _prompt_template(self, extra_system: str | None = None) -> str:
    # Custom template logic here
    pass
```

---

## System Prompt

The default system prompt is:

```
You are a helpful, concise AI assistant. Keep answers clear and practical. 
When unsure, say what you are uncertain about.
```

### Customizing the System Prompt

Modify in `ai_cli.py`:

```python
SYSTEM_PROMPT = (
    "Your custom system prompt here. "
    "Be specific about the assistant's role and capabilities."
)
```

Or pass it at runtime (requires code modification).

---

## GPU Configuration

### Automatic Detection

The CLI automatically detects CUDA GPUs:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
```

### Manual GPU Selection

Force CPU mode:

```python
# In ai_cli.py, modify the device selection
device = "cpu"  # Force CPU
```

### Mixed Precision

For better performance on modern GPUs:

```python
# Uses float16 on GPU, float32 on CPU
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
```

---

## Web Search Configuration

### API Keys

For web search features, you may need API keys:

```bash
# Optional: Serper API for web search
export SERPER_API_KEY="your_serper_api_key"

# Optional: Tavily API
export TAVILY_API_KEY="your_tavily_api_key"
```

### Search Behavior

The CLI automatically decides when to search:

```python
def should_search_web(query: str) -> bool:
    # Search for recent information
    # Skip for factual/educational queries
    pass
```

---

## .env File Configuration

### Creating .env File

```bash
# Copy example
cp .env.example .env
```

### Example .env Content

```bash
# HuggingFace Token (required for gated models)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional: Custom directories
MODEL_DIR=./models
TRANSCRIPTS_DIR=./transcripts
TRAINING_DATA_DIR=./training_data

# Optional: API Keys
# SERPER_API_KEY=your_key
# TAVILY_API_KEY=your_key
```

---

## Performance Tuning

### Memory Optimization

For systems with limited RAM:

```python
# Use quantization (if supported)
model = AutoModelForCausalLM.from_pretrained(
    str(model_dir),
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,  # Reduces peak memory
)
```

### Batch Processing

For better throughput (requires code changes):

```python
# Process multiple inputs in batches
# Useful for batch inference scenarios
```

---

## Logging Configuration

### Debug Mode

Enable verbose logging (requires code changes):

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Custom Logs

Logs are written to:
- Console (terminal output)
- Optional: Log files

---

## Next Steps

- [Usage Guide](usage.md) - Using the chat CLI
- [Model Download](model-download.md) - Downloading models
- [Advanced Features](advanced-features.md) - Prompt tuning
- [Troubleshooting](troubleshooting.md) - Common issues
