# Troubleshooting Guide

This guide covers common issues and their solutions for TinyLlama CLI.

## Common Issues

### Installation Issues

#### Python Not Found

**Error:**
```
bash: python: command not found
```

**Solution:**
```bash
# Install Python via Homebrew (macOS)
brew install python

# Or via apt (Ubuntu/Debian)
sudo apt update
sudo apt install python3 python3-venv python3-pip

# Or download from python.org
```

#### pip Not Found

**Error:**
```
bash: pip: command not found
```

**Solution:**
```bash
# Install pip
python3 -m ensurepip --upgrade

# Or
python3 -m pip install --upgrade pip
```

#### Virtual Environment Issues

**Error:**
```
Error: venv module not found
```

**Solution:**
```bash
# Ubuntu/Debian
sudo apt install python3-venv

# macOS with Homebrew (already included)
brew install python
```

---

### Model Download Issues

#### "Authentication Required" Error

**Error:**
```
HuggingFaceAuthenticationError: 
Invalid username or password
```

**Cause:** Gated model requires authentication

**Solution:**
1. Get a HuggingFace account
2. Request access to the model
3. Set your token:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

#### "Model Not Found" Error

**Error:**
```
RepositoryNotFoundError: 
404 Client Error: Not Found
```

**Cause:** Invalid model ID

**Solution:**
1. Check the model ID is correct
2. Verify the model exists on HuggingFace
3. Some models are renamed - check the latest version

```bash
# Check model exists
python -c "from huggingface_hub import HfApi; api = HfApi(); print(api.list_models('meta', limit=5))"
```

#### "No Space Left on Device"

**Error:**
```
OSError: [Errno 28] No space left on device
```

**Solution:**
```bash
# Check disk space
df -h

# Free up space by removing old models
rm -rf models/old-model-name/

# Or use a smaller model
python download_model.py --model smollm2
```

#### Slow Download

**Solution:**
```bash
# Use resume_download=True (already enabled)
# Just run the download again
python download_model.py --model tinyllama

# Or try during off-peak hours
# Or use a wired connection
```

---

### Runtime Issues

#### "Local model not found"

**Error:**
```
Local model not found. 
Run `python download_model.py` first.
```

**Solution:**
```bash
# Download a model first
python download_model.py

# Or verify models exist
ls -la models/
```

#### CUDA/GPU Issues

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Free up GPU memory
# Close other GPU applications

# Or use a smaller model
python download_model.py --model smollm2

# Or force CPU mode in ai_cli.py:
device = "cpu"
```

**Error:**
```
RuntimeError: Expected all tensors to be on the same device
```

**Cause:** Mixed CPU/GPU tensors

**Solution:**
```python
# In ai_cli.py, ensure device consistency
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

#### Out of Memory (RAM)

**Error:**
```
RuntimeError: CPU out of memory
```

**Solution:**
```python
# In ai_cli.py, use lower memory settings
model = AutoModelForCausalLM.from_pretrained(
    str(self.model_dir),
    low_cpu_mem_usage=True,  # Add this
)
```

#### Model Loading Takes Forever

**Cause:** Large model, slow disk, or not enough RAM

**Solution:**
```bash
# Check available RAM
free -h  # Linux
 Activity Monitor  # macOS

# Use a smaller model
python download_model.py --model smollm2
```

---

### Chat/Generation Issues

#### Garbage or Unexpected Output

**Cause:** Wrong prompt template or settings

**Solution:**
```python
# Adjust temperature in ai_cli.py
temperature = 0.4  # Lower for more coherent output
```

#### Repetitive Output

**Cause:** Low repetition penalty

**Solution:**
```python
# Increase repetition penalty
repetition_penalty = 1.15  # Default is 1.1
```

#### Response Too Short

**Cause:** Max tokens limit

**Solution:**
```python
# Increase max tokens
max_new_tokens = 512  # Default is 256
```

#### Response Too Long

**Cause:** High max tokens or temperature

**Solution:**
```python
# Lower max tokens
max_new_tokens = 128

# Or lower temperature
temperature = 0.4
```

---

### CLI Interface Issues

#### Terminal Display Issues

**Problem:** Text not displaying correctly

**Solution:**
```bash
# Check terminal supports Unicode
export LC_ALL=en_US.UTF-8

# Or use a different terminal
# iTerm2 (macOS), Windows Terminal
```

#### Command Not Working

**Problem:** /help, /settings, etc. not responding

**Solution:**
```bash
# Make sure to type the full command with /
# Not just "help"

# Check for typos
/help    # Correct
help     # Wrong
```

---

### Web Search Issues

#### "API Key Required" Error

**Error:**
```
APIError: API key required
```

**Solution:**
```bash
# Get API key from Serper or Tavily
export SERPER_API_KEY="your_api_key"
# or
export TAVILY_API_KEY="your_api_key"
```

#### Search Returns No Results

**Cause:** API issues or rate limiting

**Solution:**
```bash
# Wait and try again
# Check your API key is correct
# Try a different search provider
```

---

## Error Messages

### Python Errors

| Error | Meaning | Solution |
|-------|---------|----------|
| `ModuleNotFoundError` | Missing package | `pip install package-name` |
| `ImportError` | Import failed | Check Python path and dependencies |
| `SyntaxError` | Code error | Check the code syntax |

### HuggingFace Errors

| Error | Meaning | Solution |
|-------|---------|----------|
| `HfHubHTTPError 401` | Unauthorized | Set HF_TOKEN |
| `HfHubHTTPError 403` | Forbidden | Request model access |
| `HfHubHTTPError 404` | Not found | Check model ID |
| `HfHubHTTPError 429` | Rate limited | Wait and retry |

### PyTorch Errors

| Error | Meaning | Solution |
|-------|---------|----------|
| `CUDA not available` | No GPU | Install CUDA or use CPU |
| `OutOfMemoryError` | OOM | Use smaller model/settings |
| `RuntimeError` | Various | Check GPU drivers |

---

## Getting Help

### Debug Mode

Enable verbose output:

```python
# Add to your code
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Environment

```bash
# Check Python version
python --version

# Check installed packages
pip list

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Check HuggingFace token
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"
```

### Report Issues

When reporting an issue, include:

1. Operating system and version
2. Python version
3. Error message (full traceback)
4. Steps to reproduce
5. What you've tried

---

## Known Limitations

### Model-Specific Issues

| Model | Limitation |
|-------|------------|
| TinyLlama | Limited context (2048) |
| SmolLM2 | Smaller capacity |
| Qwen | May need more RAM |
| Nemotron | GGUF format may differ |

### Platform-Specific

| Platform | Issue |
|----------|-------|
| WSL | May need CUDA setup |
| Docker | Needs GPU pass-through |
| ARM Mac | Limited GPU support |

---

## Performance Tips

### Speed Up Inference

1. **Use GPU** - Much faster than CPU
2. **Use smaller models** - TinyLlama or SmolLM2
3. **Lower max tokens** - Less generation time
4. **Use greedy decoding** - `do_sample=False`

### Reduce Memory Usage

1. **Use float16** - Half the memory
2. **Load only needed files** - Use `low_cpu_mem_usage=True`
3. **Clear cache** - Restart between sessions

---

## Next Steps

- [Installation Guide](installation.md) - Setup help
- [Usage Guide](usage.md) - How to use CLI
- [Model Download](model-download.md) - Model management
- [Configuration](configuration.md) - Customize settings
