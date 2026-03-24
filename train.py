"""
TinyLlama Training Module

This module handles automatic fine-tuning of the model from conversation data.
It uses LoRA (Low-Rank Adaptation) for memory-efficient training.

Usage:
    from train import FineTuner
    
    # Initialize with model path
    trainer = FineTuner(model_path="models/TinyLlama-1.1B-Chat-v1.0")
    
    # Train on exported data
    trainer.train_from_jsonl("training_data/tinyllama_sft.jsonl")
    
    # Or train on latest conversation
    trainer.train_latest()

Options:
    --epochs N        Number of training epochs (default: 1)
    --batch-size N   Batch size (default: 1)
    --lr N           Learning rate (default: 2e-4)
    --rank N         LoRA rank (default: 8)
    --device cpu     Device to use (cpu/cuda)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Try to import training dependencies
TRAINING_DEPS_AVAILABLE = False
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollator
    from datasets import Dataset
    TRAINING_DEPS_AVAILABLE = True
except ImportError as e:
    console.print(Panel(
        f"[yellow]Training dependencies not installed:[/yellow]\n{e}\n\n"
        "[green]Install with:[/green]\n"
        "pip install torch transformers datasets peft",
        title="[yellow]Warning[/yellow]",
        border_style="yellow",
        box=box.ROUNDED,
    ))


class FineTuner:
    """Fine-tune TinyLlama on conversation data using LoRA."""
    
    DEFAULT_CONFIG = {
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    
    def __init__(
        self,
        model_path: str = "models/TinyLlama-1.1B-Chat-v1.0",
        output_dir: str = "checkpoints",
        device: str = None,
    ):
        if not TRAINING_DEPS_AVAILABLE:
            raise ImportError("Training dependencies not available. Install with: pip install torch transformers datasets peft")
        
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        console.print(f"[cyan]Fine-tuner initialized for:[/cyan] {self.model_path}")
        console.print(f"[cyan]Device:[/cyan] {self.device}")
    
    def load_jsonl_data(self, jsonl_path: str) -> list[dict]:
        """Load training data from JSONL file."""
        data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        console.print(f"[green]Loaded {len(data)} training examples[/green]")
        return data
    
    def prepare_dataset(self, data: list[dict]) -> Dataset:
        """Prepare dataset for training."""
        def format_example(ex):
            messages = ex.get("messages", [])
            text = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    text += f"<|system|>\n{content}</s>\n"
                elif role == "user":
                    text += f"<|user|>\n{content}</s>\n"
                elif role == "assistant":
                    text += f"<|assistant|>\n{content}</s>\n"
            return {"text": text}
        
        formatted = [format_example(ex) for ex in data]
        return Dataset.from_list(formatted)
    
    def train_from_jsonl(
        self,
        jsonl_path: str,
        epochs: int = 1,
        batch_size: int = 1,
        learning_rate: float = 2e-4,
        lora_rank: int = 8,
        warmup_steps: int = 10,
        save_steps: int = 50,
    ):
        """Fine-tune model from JSONL data."""
        if not self.model_path.exists():
            console.print(f"[red]Model not found at: {self.model_path}[/red]")
            return
        
        # Load data
        console.print(Panel(
            "[bold cyan]Loading training data...[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        ))
        data = self.load_jsonl_data(jsonl_path)
        
        if len(data) == 0:
            console.print("[yellow]No training data found![/yellow]")
            return
        
        dataset = self.prepare_dataset(data)
        
        # Load model and tokenizer
        with console.status("[bold cyan]Loading model...[/bold cyan]", spinner="dots"):
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
            )
        
        # Setup LoRA
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            console.print("[red]PEFT not installed. Install with: pip install peft[/red]")
            return
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=self.DEFAULT_CONFIG["target_modules"],
            lora_dropout=self.DEFAULT_CONFIG["lora_dropout"],
            bias=self.DEFAULT_CONFIG["bias"],
            task_type=self.DEFAULT_CONFIG["task_type"],
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Tokenize
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=512)
        
        dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
            save_total_limit=3,
            logging_steps=10,
            logging_dir=f"{self.output_dir}/logs",
            report_to="none",
            fp16=self.device == "cuda",
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # For causal LM
        )
        
        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        console.print(Panel(
            f"[bold green]Starting training...[/bold green]\n"
            f"Epochs: {epochs}\n"
            f"Batch size: {batch_size}\n"
            f"Learning rate: {learning_rate}\n"
            f"LoRA rank: {lora_rank}",
            border_style="green",
            box=box.ROUNDED,
        ))
        
        trainer.train()
        
        # Save
        model_save_path = self.output_dir / f"lora_finetuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        
        console.print(Panel(
            f"[bold green]Training complete![/bold green]\n"
            f"Model saved to: {model_save_path}",
            border_style="green",
            box=box.ROUNDED,
        ))
    
    def train_latest(
        self,
        epochs: int = 1,
        batch_size: int = 1,
        learning_rate: float = 2e-4,
        lora_rank: int = 8,
    ):
        """Train on the latest conversation transcript."""
        transcripts_dir = Path("transcripts")
        
        if not transcripts_dir.exists():
            console.print("[red]No transcripts directory found![/red]")
            return
        
        # Find latest transcript
        transcripts = sorted(transcripts_dir.glob("*.json"))
        
        if not transcripts:
            console.print("[red]No transcripts found![/red]")
            return
        
        latest = transcripts[-1]
        console.print(f"[cyan]Training on latest transcript:[/cyan] {latest}")
        
        # Convert transcript to training format
        with open(latest, "r") as f:
            transcript_data = json.load(f)
        
        # Convert to JSONL format
        training_data = []
        messages = transcript_data.get("messages", [])
        
        system_msg = "You are a helpful, concise AI assistant."
        
        # Create examples from user-assistant pairs
        pending_user = None
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                pending_user = content
            elif role == "assistant" and pending_user:
                training_data.append({
                    "id": f"{latest.stem}-{len(training_data)}",
                    "source_transcript": str(latest),
                    "created_at": datetime.now().isoformat(),
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": pending_user},
                        {"role": "assistant", "content": content},
                    ]
                })
                pending_user = None
        
        # Save temporary JSONL
        temp_jsonl = Path("training_data") / f"{latest.stem}_temp.jsonl"
        temp_jsonl.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_jsonl, "w") as f:
            for ex in training_data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        
        # Train
        self.train_from_jsonl(
            str(temp_jsonl),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lora_rank=lora_rank,
        )
        
        # Cleanup
        temp_jsonl.unlink()
    
    def merge_and_save(self, checkpoint_path: str, output_name: str = "merged"):
        """Merge LoRA weights with base model and save."""
        from peft import PeftModel
        
        console.print(Panel(
            "[bold cyan]Merging LoRA weights...[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        ))
        
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        base_model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        merged = model.merge_and_unload()
        
        output_path = self.output_dir / output_name
        merged.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        
        console.print(Panel(
            f"[bold green]Merged model saved to:[/bold green] {output_path}",
            border_style="green",
            box=box.ROUNDED,
        ))


def DataCollatorForLanguageModeling(tokenizer, mlm=False):
    """Simple data collator for language modeling."""
    def collate_fn(batch):
        texts = [item["input_ids"] for item in batch]
        return tokenizer.pad(
            {"input_ids": texts},
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
    return collate_fn


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama on conversation data")
    parser.add_argument("--jsonl", type=str, default="training_data/tinyllama_sft.jsonl",
                        help="Path to JSONL training data")
    parser.add_argument("--model", type=str, default="models/TinyLlama-1.1B-Chat-v1.0",
                        help="Path to model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--latest", action="store_true", 
                        help="Train on latest transcript instead of JSONL")
    parser.add_argument("--merge", type=str, default=None,
                        help="Merge checkpoint and save")
    parser.add_argument("--output", type=str, default="merged",
                        help="Output name for merged model")
    
    args = parser.parse_args()
    
    trainer = FineTuner(model_path=args.model)
    
    if args.merge:
        trainer.merge_and_save(args.merge, args.output)
    elif args.latest:
        trainer.train_latest(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            lora_rank=args.rank,
        )
    else:
        trainer.train_from_jsonl(
            args.jsonl,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            lora_rank=args.rank,
        )


if __name__ == "__main__":
    main()
