#!/usr/bin/env python3
"""Backend server for TinyLlama GUI - handles IPC communication with Electron app."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
SETTINGS_FILE = PROJECT_ROOT / "tinyllama_gui_settings.json"

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


class TinyLlamaBackend:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_mode = "chat"
        self.settings = self._load_settings()
        self.chat_history = []

    def _load_settings(self) -> dict:
        """Load settings from file or return defaults."""
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE) as f:
                    return {**DEFAULT_SETTINGS, **json.load(f)}
            except Exception:
                pass
        return DEFAULT_SETTINGS.copy()

    def _save_settings(self) -> None:
        """Save settings to file."""
        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(self.settings, f)
        except Exception as e:
            print(f"Error saving settings: {e}", file=sys.stderr)

    def load_model(self) -> bool:
        """Load the language model if not already loaded."""
        if self.model is not None:
            return True

        # Find installed model
        if not MODEL_DIR.exists():
            return False

        model_paths = [d for d in MODEL_DIR.iterdir() if d.is_dir()]
        if not model_paths:
            return False

        # Use first available model
        model_path = str(model_paths[0])

        try:
            print(f"Loading model from {model_path}...", file=sys.stderr)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(device)
            print(f"Model loaded on {device}", file=sys.stderr)
            return True
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            return False

    def _build_prompt(self, user_message: str) -> str:
        """Build the prompt for the model."""
        system_prompt = self.settings.get("systemPrompt", DEFAULT_SETTINGS["systemPrompt"])
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent chat history (last 5 turns)
        for msg in self.chat_history[-10:]:
            messages.append(msg)
        
        messages.append({"role": "user", "content": user_message})

        # Format for TinyLlama chat template
        chunks = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"].strip()
            if role == "system":
                chunks.append(f"<|system|>\n{content}</s>")
            elif role == "user":
                chunks.append(f"<|user|>\n{content}</s>")
            elif role == "assistant":
                chunks.append(f"<|assistant|>\n{content}</s>")
        
        chunks.append("<|assistant|>\n")
        return "\n".join(chunks)

    def generate_response(self, user_message: str) -> str:
        """Generate a response from the model."""
        if not self.load_model():
            return "Error: No model available. Please download a model first."

        prompt = self._build_prompt(user_message)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)

            # Get mode-specific config
            mode_config = MODE_CONFIGS.get(self.current_mode, MODE_CONFIGS["chat"])
            
            generation_cfg = {
                "max_new_tokens": self.settings.get("maxTokens", 512),
                "temperature": mode_config.get("temperature", self.settings.get("temperature", 0.7)),
                "top_p": mode_config.get("top_p", self.settings.get("topP", 0.9)),
                "do_sample": True,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.eos_token_id,
            }

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_cfg
                )

            new_tokens = output_ids[0, input_ids.shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean up response
            response = response.split("</s>")[0].strip()
            
            # Update chat history
            self.chat_history.append({"role": "user", "content": user_message})
            self.chat_history.append({"role": "assistant", "content": response})
            
            return response

        except Exception as e:
            return f"Error generating response: {str(e)}"

    def handle_message(self, msg: dict) -> dict:
        """Handle an incoming message and return a response."""
        msg_type = msg.get("type")
        
        if msg_type == "chat":
            message = msg.get("message", "")
            response = self.generate_response(message)
            return {"response": response}
        
        elif msg_type == "set-mode":
            mode = msg.get("mode", "chat")
            if mode in MODE_CONFIGS:
                self.current_mode = mode
                return {"success": True, "mode": mode}
            return {"error": f"Unknown mode: {mode}"}
        
        elif msg_type == "get-settings":
            return {"settings": self.settings}
        
        elif msg_type == "save-settings":
            new_settings = msg.get("settings", {})
            self.settings = {**self.settings, **new_settings}
            self._save_settings()
            return {"success": True}
        
        elif msg_type == "download-model":
            return {"success": True, "message": "Please use python download_model.py"}
        
        elif msg_type == "get-modes":
            return {
                "modes": [
                    {"id": "chat", "name": "Chat", "description": "General conversation mode"},
                    {"id": "code", "name": "Code", "description": "Software development assistance"},
                    {"id": "creative", "name": "Creative", "description": "Writing and brainstorming"},
                    {"id": "analysis", "name": "Analysis", "description": "Logical reasoning and problem solving"},
                ]
            }
        
        return {"error": f"Unknown message type: {msg_type}"}


def main():
    """Main entry point - read messages from stdin and write responses to stdout."""
    backend = TinyLlamaBackend()
    
    print("TinyLlama backend ready", file=sys.stderr)
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            
            line = line.strip()
            if not line:
                continue
            
            msg = json.loads(line)
            response = backend.handle_message(msg)
            print(json.dumps(response), flush=True)
            
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON: {e}"}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()