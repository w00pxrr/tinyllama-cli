#!/usr/bin/env python3
"""TinyLlama GUI Backend - Provides model loading and generation for the C++ Qt GUI."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

# Default settings
DEFAULT_SETTINGS = {
    "temperature": 0.7,
    "maxTokens": 512,
    "topP": 0.9,
    "systemPrompt": "You are a helpful AI assistant."
}

# Mode configurations
MODE_CONFIGS = {
    "chat": {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 512,
    },
    "code": {
        "temperature": 0.4,
        "top_p": 0.85,
        "max_new_tokens": 1024,
    },
    "creative": {
        "temperature": 0.9,
        "top_p": 0.95,
        "max_new_tokens": 768,
    },
    "analysis": {
        "temperature": 0.5,
        "top_p": 0.85,
        "max_new_tokens": 1024,
    },
}

# Global model and tokenizer
model: Optional[AutoModelForCausalLM] = None
tokenizer: Optional[AutoTokenizer] = None
current_model_path: Optional[str] = None


def load_model(model_path: str) -> bool:
    """Load a model from the given path."""
    global model, tokenizer, current_model_path
    
    try:
        model_path = Path(model_path)
        
        # Check if it's a directory (HuggingFace format) or a single file
        if model_path.is_dir():
            # Load from directory
            print(f"Loading model from {model_path}...", file=sys.stderr)
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            # Single file - need to handle GGUF or other formats
            print(f"Loading model from {model_path}...", file=sys.stderr)
            # For now, just try to load as pretrained
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        current_model_path = model_path
        print(f"Model loaded successfully!", file=sys.stderr)
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return False


def generate_response(
    message: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 0.9,
    mode: str = "chat",
    system_prompt: str = "You are a helpful AI assistant."
) -> str:
    """Generate a response from the model."""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return json.dumps({"error": "No model loaded"})
    
    try:
        # Get mode-specific settings
        mode_settings = MODE_CONFIGS.get(mode, MODE_CONFIGS["chat"])
        
        # Build the prompt in TinyLlama chat format
        prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{message}</s>\n<|assistant|>\n"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        return json.dumps({"response": response})
        
    except Exception as e:
        return json.dumps({"error": str(e)})


def main():
    """Main entry point for the GUI backend."""
    parser = argparse.ArgumentParser(description="TinyLlama GUI Backend")
    parser.add_argument("--load-model", type=str, help="Path to model to load")
    parser.add_argument("--model", type=str, help="Model identifier or path")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--system-prompt", type=str, default="You are a helpful AI assistant.")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Load model if specified
    if args.load_model:
        success = load_model(args.load_model)
        if not success:
            print(json.dumps({"error": "Failed to load model"}))
            sys.exit(1)
        print(json.dumps({"status": "loaded"}))
    elif args.model:
        # Try to find model in models directory
        model_path = MODEL_DIR / args.model
        if model_path.exists():
            success = load_model(str(model_path))
            if not success:
                print(json.dumps({"error": "Failed to load model"}))
                sys.exit(1)
            print(json.dumps({"status": "loaded"}))
        else:
            # Try loading directly
            success = load_model(args.model)
            if not success:
                print(json.dumps({"error": "Failed to load model"}))
                sys.exit(1)
            print(json.dumps({"status": "loaded"}))
    elif args.interactive:
        # Interactive mode - read from stdin
        if model is None:
            print(json.dumps({"error": "No model loaded"}))
            sys.exit(1)
        
        # Read input from stdin
        message = sys.stdin.read().strip()
        if message:
            response = generate_response(
                message,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                mode=args.mode,
                system_prompt=args.system_prompt
            )
            print(response)
    else:
        # No arguments, just print status
        print(json.dumps({
            "loaded": model is not None,
            "model": str(current_model_path) if current_model_path else None
        }))


if __name__ == "__main__":
    main()
