# Model Download Guide

This guide explains how to download and manage models for TinyLlama CLI.

## Available Models

TinyLlama CLI comes with several pre-configured models:

### Model Comparison Table

| Key | Model ID | Size | Description |
|-----|----------|------|-------------|
| tinyllama | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | ~1GB | Lightweight chat model, fast inference |
| smollm2 | HuggingFaceTB/SmolLM2-135M | ~270MB | Very small, resource-efficient |
| qwen | Qwen/Qwen2.5-0.5B-Instruct | ~1GB | Good quality, multilingual support |
| nvidia_nemotron | nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF | ~4GB | NVIDIA's efficient nano model |

### Model Recommendations

- **Beginners**: Start with `tinyllama` - it's fast and works well on most machines
- **Limited RAM**: Use `smollm2` - runs smoothly on systems with 8GB+ RAM
- **Better Quality**: Try `qwen` or `nvidia_nemotron` - higher quality outputs
- **Research**: Mix different models for different tasks

---

## Downloading a Model

### Interactive Mode

Run the downloader without arguments to see the interactive picker:

```bash
python download_model.py
```

You'll see:

```
╔════════════════════════════════════════════════════════════════╗
║                      Model Downloader                           ║
╠════════════════════════════════════════════════════════════════╣
║  Available: tinyllama  smollm2  qwen  nvidia_nemotron  choose more ║
╚════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════╗
║                      Model Picker                                ║
╠════════════════════════════════════════════════════════════════╣
║  #   Key               Model                                    ║
║  ─────────────────────────────────────────────────────────────  ║
║  1   tinyllama         TinyLlama/TinyLlama-1.1B-Chat-v1.0       ║
║  2   smollm2          HuggingFaceTB/SmolLM2-135M               ║
║  3   qwen             Qwen/Qwen2.5-0.5B-Instruct               ║
║  4   nvidia_nemotron  nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF     ║
║  5   choose more      Type any HuggingFace model ID            ║
╚════════════════════════════════════════════════════════════════╝
```

Select a number (1-5) to download that model.

### Direct Download

Download a specific model directly:

```bash
# Using the short key
python download_model.py --model tinyllama
python download_model.py --model smollm2
python download_model.py --model qwen
python download_model.py --model nvidia_nemotron
```

---

## Custom Models

### Download Any HuggingFace Model

Select option "5" (choose more) in the interactive picker, or use a custom model ID:

```bash
# Any model from HuggingFace Hub
python download_model.py --model meta/Llama-3-8B
python download_model.py --model mistralai/Mistral-7B-Instruct-v0.2
python download_model.py --modelEleutherAI/gpt-neo-2.7B
```

### Model ID Format

The model ID should be in the format `organization/model-name`:

| Organization | Example Model IDs |
|--------------|-------------------|
| meta | Llama-3-8B, Llama-2-13B-chat |
| mistralai | Mistral-7B-Instruct-v0.2, Mixtral-8x7B |
| EleutherAI | gpt-neo-2.7B, gpt-j-6B |
| TinyLlama | TinyLlama-1.1B-Chat-v1.0 |
| Qwen | Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B |

### Important: Gated Models

Some models require permission to download:

1. **Llama models** (Meta): Need HuggingFace token with Llama license
2. **Gemma models** (Google): Need permission
3. **Some Mistral variants**: May need acceptance

For these, you must:
1. Get a HuggingFace account
2. Request access to the model
3. Set your token via environment variable or prompt

---

## HuggingFace Token

### Why You Need a Token

Some models are "gated" and require:
- Account on HuggingFace
- Acceptance of the model's license
- Authentication via token

### Setting Up Your Token

#### Option 1: Environment Variable (Recommended)

```bash
# Linux/macOS
export HF_TOKEN="your_huggingface_token_here"

# Windows (PowerShell)
$env:HF_TOKEN = "your_huggingface_token_here"
```

Or use the alternative variable:

```bash
export HUGGINGFACEHUB_API_TOKEN="your_huggingface_token_here"
```

#### Option 2: Prompt (Automatic)

If no token is found in environment variables, the downloader will prompt you:

```
HuggingFace Token (optional)
Press Enter to skip (anonymous download may fail for gated models): 
```

Enter your token or press Enter to skip.

#### Option 3: .env File

```bash
# Copy example
cp .env.example .env

# Edit .env and add:
HF_TOKEN=your_token_here
```

---

## Download Options

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--model` | Model key or custom model ID | `--model tinyllama` |
| `--model` | Any HuggingFace model | `--model meta/Llama-3-8B` |

### Resume Downloads

The downloader automatically resumes interrupted downloads:

```bash
# Just run again - it will pick up where left off
python download_model.py --model tinyllama
```

### Model Storage Location

By default, models are stored in:

```
models/
├── TinyLlama-1.1B-Chat-v1.0/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── NVIDIA-Nemotron-3-Nano-4B-GGUF/
│   └── ...
└── ...
```

You can change the location by modifying the code or using symbolic links.

---

## Managing Downloaded Models

### Listing Models

Models are stored in the `./models` directory:

```bash
ls -la models/
```

### Deleting a Model

To remove a downloaded model:

```bash
# Delete a specific model folder
rm -rf models/TinyLlama-1.1B-Chat-v1.0

# Or use the tinyllama.sh helper (if implemented)
./tinyllama.sh --remove-model tinyllama
```

### Selecting a Model at Runtime

When you run `ai_cli.py`, it will show installed models:

```
#   Installed Model              Path
─────────────────────────────────────────
1   TinyLlama-1.1B-Chat-v1.0   ./models/TinyLlama-1.1B-Chat-v1.0
2   NVIDIA-Nemotron-3-Nano     ./models/NVIDIA-Nemotron-3-Nano-4B-GGUF
```

---

## Troubleshooting Download Issues

### "Authentication Required" Error

```
Error: HfApiInvalidUsername: 
```

**Solution**: Set your HuggingFace token (see HuggingFace Token section)

### "Model Not Found" Error

```
Error: Repository not found: https://huggingface.co/api/models/bad/model/id
```

**Solution**: Check the model ID is correct. Some models are renamed or moved.

### Disk Space Issues

```
Error: No space left on device
```

**Solution**: Free up space or download smaller models (smollm2, tinyllama)

### Slow Downloads

The downloader uses `resume_download=True` to continue interrupted downloads. For slow connections, consider:
- Using a wired connection
- Downloading during off-peak hours
- Using a mirror (if available)

---

## Next Steps

- [Usage Guide](usage.md) - How to use the chat CLI
- [Configuration](configuration.md) - Environment setup
- [Advanced Features](advanced-features.md) - Prompt tuning and export
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
