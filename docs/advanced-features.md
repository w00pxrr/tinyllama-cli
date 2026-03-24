# Advanced Features

This guide covers advanced features of TinyLlama CLI including prompt tuning, training data export, and transcript management.

## Prompt Tuning

### How It Works

TinyLlama CLI automatically optimizes generation settings based on your input:

```python
# Automatic tuning based on prompt analysis
def tune(user_input: str, turns: int) -> GenerationConfig:
    # Analyzes the prompt and adjusts settings
    pass
```

### Tuning Triggers

| Trigger Keywords | Settings Adjusted |
|-----------------|-------------------|
| explain, what is, why, how does, difference, compare | Temperature: 0.45, Top-P: 0.82, Max: 220 |
| code, python, bug, error, function, class, api | Temperature: 0.40, Top-P: 0.85, Max: 280 |
| story, poem, creative, brainstorm, ideas, fiction | Temperature: 0.88, Top-P: 0.95, Max: 320 |
| calculate, solve, math, equation, evaluate | Temperature: 0.00, Top-P: 1.00, Max: 96 |

### Manual Tuning

You can modify the automatic tuning in `ai_cli.py`:

```python
class TinyLlamaOptimizer:
    FACTUAL_HINTS = ("explain", "what is", "why", "how does", "difference", "compare")
    CREATIVE_HINTS = ("story", "poem", "creative", "brainstorm", "ideas", "fiction")
    CODE_HINTS = ("code", "python", "bug", "error", "function", "class", "api")
    MATH_HINTS = ("calculate", "solve", "math", "equation", "evaluate", "multiply", "divide")
```

### Adding Custom Triggers

Add your own keywords:

```python
class TinyLlamaOptimizer:
    # Add custom triggers
    MY_HINTS = ("custom", "keywords", "here")
    
    @staticmethod
    def tune(user_input: str, turns: int) -> GenerationConfig:
        cfg = GenerationConfig()
        
        if any(hint in user_input.lower() for hint in TinyLlamaOptimizer.MY_HINTS):
            cfg.temperature = 0.5
            # ... more custom settings
        
        return cfg
```

---

## Training Data Export

### Automatic Export

Every time you use `/save` or exit the CLI, training data is exported:

```
training_data/tinyllama_sft.jsonl
```

### Export Format

Each line is a JSON object:

```json
{
  "id": "conv_2024-01-15_14-30-22",
  "source_transcript": "2024-01-15_143022.json",
  "created_at": "2024-01-15T14:30:22Z",
  "messages": [
    {"role": "system", "content": "You are a helpful AI assistant..."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a high-level..."}
  ]
}
```

### Training Data Fields

| Field | Description |
|-------|-------------|
| `id` | Unique conversation ID |
| `source_transcript` | Original transcript filename |
| `created_at` | ISO 8601 timestamp |
| `messages` | Array of message objects |

### Customizing Export

Modify export behavior in `ai_cli.py`:

```python
def _export_training_data(self) -> None:
    # Custom export logic here
    pass
```

---

## Transcript Saving

### Automatic Saving

Transcripts are saved:

1. When you type `/save`
2. When you type `/exit`
3. When you press Ctrl+D

### Transcript Location

```
transcripts/
├── 2024-01-15_143022.json
├── 2024-01-16_091545.json
└── ...
```

### Transcript Format

```json
{
  "model": "TinyLlama-1.1B-Chat-v1.0",
  "model_path": "./models/TinyLlama-1.1B-Chat-v1.0",
  "started_at": "2024-01-15T14:30:22Z",
  "settings": {
    "temperature": 0.65,
    "top_p": 0.9,
    "max_new_tokens": 256
  },
  "messages": [
    {
      "role": "user",
      "content": "Hello!",
      "timestamp": "2024-01-15T14:30:25Z"
    },
    {
      "role": "assistant", 
      "content": "Hello! How can I help you today?",
      "timestamp": "2024-01-15T14:30:28Z"
    }
  ]
}
```

### Loading Transcripts

To load a previous transcript (requires code modification):

```python
# Example: Load transcript at startup
python ai_cli.py --load-transcript transcripts/2024-01-15_143022.json
```

---

## Web Search Integration

### When to Use

The CLI automatically decides when to search based on:

```python
def should_search_web(query: str) -> bool:
    # Search for recent events, news, weather
    # Skip for general knowledge
    pass
```

### Search Providers

The CLI supports multiple search backends:

1. **Serper** (default) - Google's search results
2. **Tavily** - AI-powered search
3. **DuckDuckGo** - Free, no API key needed

### Setting Up Search

```bash
# Serper API
export SERPER_API_KEY="your_api_key"

# Tavily API  
export TAVILY_API_KEY="your_api_key"
```

### Using Web Search

Just ask about current events:

```
You: What's the latest news about AI?

[Searching the web...]

TinyLlama: Here's the latest news about AI...
```

---

## Custom Prompt Templates

### TinyLlama Template

```xml
<|system|>
{system_prompt}</s>
<|user|>
{user_message}</s>
<|assistant|>
{response}</s>
```

### Custom Templates

Add your own template in `ai_cli.py`:

```python
def _prompt_template(self, extra_system: str | None = None) -> str:
    # Your custom template
    template = f"""<|system|>
{self.history[0]['content']}</s>
<|user|>
{self.history[-1]['content']}</s>
<|assistant|>
"""
    return template
```

### HuggingFace Chat Templates

Many models use standard chat templates:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_dir)
chat = [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello"}
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False)
```

---

## Memory Management

### Context Window

Each model has a maximum context length:

| Model | Context Length |
|-------|---------------|
| TinyLlama | 2048 tokens |
| SmolLM2 | 2048 tokens |
| Qwen2.5 | 8192 tokens |
| Nemotron | 4096 tokens |

### Dynamic Context

The CLI automatically manages context:

```python
def _dynamic_max_new_tokens(self, prompt_tokens: int, requested_tokens: int) -> int:
    # Calculates safe token limits
    # Adjusts based on available RAM
    pass
```

### RAM-Based Limits

| Available RAM | Max Tokens |
|--------------|------------|
| < 8 GB | 96 |
| < 12 GB | 160 |
| < 16 GB | 224 |
| < 24 GB | 320 |
| < 32 GB | 448 |
| > 32 GB | 640 |

---

## Performance Optimization

### GPU Optimization

```python
# Use float16 for faster inference on GPU
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
```

### CPU Optimization

```python
# Use quantized weights (if supported)
# Check your model's quantization support
```

### Inference Speed

Tips for faster inference:

1. Use GPU if available
2. Use smaller models (TinyLlama, SmolLM2)
3. Reduce `max_new_tokens`
4. Use greedy decoding (`do_sample=False`)

---

## Custom Commands

### Adding Commands

Add custom commands in `ai_cli.py`:

```python
def _handle_custom_command(self, cmd: str) -> bool:
    if cmd == "/mycommand":
        # Custom logic
        return True
    return False
```

### Example: Context Stats

```python
def _show_context_stats(self) -> None:
    """Show current context usage"""
    console.print(f"Messages: {len(self.history)}")
    console.print(f"Prompt tokens: {self.last_prompt_tokens}")
    console.print(f"Response tokens: {self.last_response_tokens}")
```

---

## Next Steps

- [Usage Guide](usage.md) - Basic usage
- [Model Download](model-download.md) - Download models
- [Configuration](configuration.md) - Environment setup
- [API Reference](api-reference.md) - Full API documentation
- [Troubleshooting](troubleshooting.md) - Common issues
