# TinyLlama CLI - Documentation

## Table of Contents

1. [Installation](installation.md)
   - [Quick Start](installation.md#quick-start)
   - [Manual Setup](installation.md#manual-setup)
   - [Bootstrap Script](installation.md#bootstrap-script)
   - [Environment Variables](installation.md#environment-variables)

2. [Usage](usage.md)
   - [Starting the CLI](usage.md#starting-the-cli)
   - [Model Selection](usage.md#model-selection)
   - [Chat Commands](usage.md#chat-commands)
   - [Settings](usage.md#settings)

3. [Model Download](model-download.md)
   - [Available Models](model-download.md#available-models)
   - [Custom Models](model-download.md#custom-models)
   - [Download Options](model-download.md#download-options)

4. [Configuration](configuration.md)
   - [Environment Variables](configuration.md#environment-variables)
   - [HuggingFace Token](configuration.md#huggingface-token)
   - [Model Directory](configuration.md#model-directory)

5. [Advanced Features](advanced-features.md)
   - [Prompt Tuning](advanced-features.md#prompt-tuning)
   - [Training Data Export](advanced-features.md#training-data-export)
   - [Transcript Saving](advanced-features.md#transcript-saving)

6. [API Reference](api-reference.md)
   - [CLI Commands](api-reference.md#cli-commands)
   - [Python API](api-reference.md#python-api)

7. [Troubleshooting](troubleshooting.md)
   - [Common Issues](troubleshooting.md#common-issues)
   - [Error Messages](troubleshooting.md#error-messages)

---

## Overview

TinyLlama CLI is a polished terminal chat application powered by local language models. It provides a rich terminal UI with markdown rendering, automatic prompt tuning, and training data export capabilities.

### Key Features

- **Local Inference**: Run models locally on your machine
- **Rich Terminal UI**: Fancy panels, markdown rendering, status spinners
- **Auto-Tuning**: Runtime prompt-aware optimization for better responses
- **Training Data Export**: Automatic export to JSONL format
- **Multiple Model Support**: Download and use various models from HuggingFace
