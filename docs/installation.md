# Installation Guide

This guide covers all the ways to install and set up TinyLlama CLI on your system.

## Quick Start

The fastest way to get started is using the bootstrap script:

```bash
cd /Path/to/tinyllama-cli
./tinyllama.sh
```

The bootstrap script will automatically:
1. Check for Python and install if needed
2. Create a virtual environment (`.venv`)
3. Install dependencies from `requirements.txt`
4. Download a model (if none is installed)
5. Launch the chat CLI

---

## Manual Setup

If you prefer to set up manually or need more control, follow these steps:

### Prerequisites

- Python 3.10 or later
- pip (Python package manager)
- git (optional, for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-repo/tinyllama-cli.git
cd tinyllama-cli
```

### Step 2: Create a Virtual Environment

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate it (Linux/macOS)
source .venv/bin/activate

# Activate it (Windows)
.venv\Scripts\activate.bat
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Environment Variables (Optional)

```bash
# Set HuggingFace token for gated models
export HF_TOKEN="your_huggingface_token_here"

# Or use the alternative variable name
export HUGGINGFACEHUB_API_TOKEN="your_huggingface_token_here"
```

You can also copy the example environment file:

```bash
cp .env.example .env
# Edit .env and add your token
```

### Step 5: Download a Model

```bash
python download_model.py
```

This will prompt you to select a model. See [Model Download](model-download.md) for more options.

### Step 6: Start the Chat CLI

```bash
python ai_cli.py
```

Or use the bootstrap script:

```bash
./tinyllama.sh
```

---

## Bootstrap Script

The `tinyllama.sh` script automates the entire setup process. Here's what it does:

### What It Does

1. **Python Check**: Verifies Python is installed, installs via Homebrew/apt if needed
2. **Virtual Environment**: Creates `.venv` in the project directory
3. **Dependency Installation**: Installs all packages from `requirements.txt`
4. **Model Download**: Downloads a model if none is installed
5. **CLI Launch**: Starts the chat interface

### Usage

```bash
# Basic usage - walks through setup and launches CLI
./tinyllama.sh

# Bootstrap only (download model but don't start CLI)
./tinyllama.sh --bootstrap-only

# Auto-download specific model
./tinyllama.sh --model tinyllama
```

### Options

| Option | Description |
|--------|-------------|
| `--bootstrap-only` | Download dependencies and model, but don't start CLI |
| `--model MODEL` | Auto-download a specific model (tinyllama, smollm2, qwen, nvidia_nemotron) |

---

## Environment Variables

TinyLlama CLI uses the following environment variables:

### Required for Gated Models

```bash
# HuggingFace token (required for some models like Llama)
export HF_TOKEN="your_token_here"

# Alternative variable name
export HUGGINGFACEHUB_API_TOKEN="your_token_here"
```

### Optional Variables

```bash
# Custom model directory (default: ./models)
export MODEL_DIR="./custom_models"

# Custom transcripts directory (default: ./transcripts)
export TRANSCRIPTS_DIR="./custom_transcripts"
```

### Setting Up Environment Variables

#### Linux/macOS (Bash/Zsh)

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
export HF_TOKEN="your_huggingface_token_here"
```

Then reload:

```bash
source ~/.bashrc
```

#### Windows (PowerShell)

```powershell
$env:HF_TOKEN = "your_huggingface_token_here"
```

To make it permanent, add to your PowerShell profile.

---

## System Requirements

### Minimum Requirements

- **OS**: macOS, Linux, or Windows (WSL supported)
- **Python**: 3.10+
- **RAM**: 8GB (for smaller models like TinyLlama)
- **Storage**: 2GB+ for model files

### Recommended Requirements

- **OS**: macOS 12+, Ubuntu 20.04+, or Windows 11 with WSL2
- **Python**: 3.11+
- **RAM**: 12GB+ (for larger models like Nemotron)
- **Storage**: 10GB+ for multiple models
- **GPU**: NVIDIA GPU with CUDA (optional, for faster inference)

### GPU Support

For GPU-accelerated inference:

1. Install CUDA Toolkit (11.8+)
2. Install PyTorch with CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

The CLI will automatically use CUDA if available.

---

## Building C++ Extensions from Source

TinyLlama CLI includes optional C++ extensions for performance optimization. These extensions provide:

- **MathExtractor**: Fast math expression extraction and evaluation
- **RamDetector**: Cross-platform RAM detection
- **StringUtils**: Optimized string operations (cleaning, trimming, etc.)
- **TokenCounter**: Fast token estimation

### Prerequisites

```bash
# Install pybind11
pip install pybind11

# For macOS, you may need Xcode command line tools
xcode-select --install
```

### Building the Extension

```bash
# Navigate to the cpp_extensions directory
cd cpp_extensions

# Build the extension in-place
python setup.py build_ext --inplace
```

This will create `tinyllama_cpp.cpython-XXX-darwin.so` (or `.so` on Linux, `.pyd` on Windows).

### Installing for Use

Copy the built extension to your virtual environment:

```bash
# For macOS
cp tinyllama_cpp.cpython-314-darwin.so .venv/lib/python3.14/site-packages/

# For Linux
cp tinyllama_cpp.cpython-310-x86_64-linux-gnu.so .venv/lib/python3.10/site-packages/
```

### Verifying the Installation

```bash
.venv/bin/python -c "import tinyllama_cpp; print(tinyllama_cpp.VERSION)"
```

### Troubleshooting

**Error: unsupported option '-fopenmp'**

On macOS with clang, OpenMP may not be available. Edit `cpp_extensions/setup.py` and comment out the `-fopenmp` flag:

```python
# extra_compile_args = [
#     "-O3",
#     "-march=native",
#     "-ffast-math",
#     "-fopenmp",  # Comment this out on macOS
#     "-std=c++17",
# ]
```

**Error: character too large for enclosing character literal**

This is a known issue with Unicode characters in C++ on some compilers. The source code has been updated to use `static_cast<char>()` for Unicode characters like × (0xD7) and ÷ (0xF7).

---

## Next Steps

- [Usage Guide](usage.md) - Learn how to use the chat CLI
- [Model Download](model-download.md) - Download different models
- [Configuration](configuration.md) - Customize your setup
- [Advanced Features](advanced-features.md) - Learn about prompt tuning and export features
