#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download
from rich import box
from rich.align import Align
from rich.console import Group
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

MODEL_CHOICES = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "smollm2": "HuggingFaceTB/SmolLM2-135M",
    "qwen": "Qwen/Qwen2.5-0.5B-Instruct",
    "nvidia_nemotron": "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
}

console = Console()


def render_header() -> None:
    hero = Group(
        Align.center(Text("Local Model Setup", style="bold white")),
        Align.center(Text("Download a Hugging Face model into ./models", style="bright_black")),
        Text(""),
        Align.center(
            Text.assemble(
                ("Available: ", "white"),
                ("tinyllama", "bold cyan"),
                ("  ", "white"),
                ("smollm2", "bold green"),
                ("  ", "white"),
                ("qwen", "bold yellow"),
                ("  ", "white"),
                ("nvidia_nemotron", "bold magenta"),
                ("  ", "white"),
                ("choose more", "bold red"),
            )
        ),
    )
    console.print(
        Panel(
            hero,
            title="[bold cyan]Model Downloader[/bold cyan]",
            border_style="cyan",
            box=box.DOUBLE_EDGE,
            padding=(1, 2),
        )
    )


def print_note(message: str, title: str = "Status", style: str = "cyan") -> None:
    console.print(
        Panel(
            Text(message, style=f"bold {style}"),
            title=title,
            border_style=style,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a local model into the ./models directory."
    )
    parser.add_argument(
        "--model",
        help="Model key: tinyllama, smollm2, qwen, nvidia_nemotron, or provide any HuggingFace model ID (e.g., org/model)",
    )
    return parser.parse_args()


def pick_model_key(model_arg: str | None) -> str:
    if model_arg:
        return model_arg

    table = Table(box=box.SIMPLE_HEAVY, expand=True, header_style="bold cyan")
    table.add_column("#", style="bold cyan", no_wrap=True)
    table.add_column("Key", style="white", no_wrap=True)
    table.add_column("Model", style="green")
    table.add_row("1", "tinyllama", MODEL_CHOICES["tinyllama"])
    table.add_row("2", "smollm2", MODEL_CHOICES["smollm2"])
    table.add_row("3", "qwen", MODEL_CHOICES["qwen"])
    table.add_row("4", "nvidia_nemotron", MODEL_CHOICES["nvidia_nemotron"])
    table.add_row("5", "choose more", "Type any HuggingFace model ID")
    console.print(
        Panel(
            Group(
                Text("Choose a model to download", style="bold white"),
                Text("Models will be stored in the local ./models directory.", style="bright_black"),
                Text(""),
                table,
            ),
            title="[bold cyan]Model Picker[/bold cyan]",
            border_style="cyan",
            box=box.HEAVY,
            padding=(1, 2),
        )
    )

    selection = Prompt.ask(
        "[bold cyan]Download model[/bold cyan]", choices=["1", "2", "3", "4", "5"], default="1"
    )
    if selection == "1":
        return "tinyllama"
    if selection == "2":
        return "smollm2"
    if selection == "3":
        return "qwen"
    if selection == "4":
        return "nvidia_nemotron"
    
    # "choose more" - prompt for custom HuggingFace model ID
    custom_model_id = Prompt.ask(
        "[bold cyan]Enter HuggingFace model ID[/bold cyan]",
        default="",
    )
    if not custom_model_id:
        print_note("No model ID provided, using default.", title="Info", style="yellow")
        return "tinyllama"
    
    # Validate the format (should contain "/")
    if "/" not in custom_model_id:
        print_note("Invalid model ID format. Expected: org/model (e.g., meta/Llama-3-8B)", title="Error", style="red")
        return "tinyllama"
    
    return custom_model_id


def model_dir_for(model_id: str) -> Path:
    folder_name = model_id.split("/")[-1]
    return Path("models") / folder_name


def main() -> None:
    import sys
    render_header()
    args = parse_args()
    
    # Check for token in environment first
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    # If no token in env, prompt the user
    if not token:
        console.print()
        token_input = Prompt.ask(
            "[bold cyan]HuggingFace Token (optional)[/bold cyan]\n" 
            "[dim]Press Enter to skip (anonymous download may fail for gated models)[/dim]",
            default="",
        )
        token = token_input if token_input else None
    
    model_key = pick_model_key(args.model)
    
    # Check if model_key is a custom model ID (not in predefined choices)
    if model_key not in MODEL_CHOICES:
        # It's a custom model ID
        model_id = model_key
    else:
        model_id = MODEL_CHOICES[model_key]
    
    model_dir = model_dir_for(model_id)

    model_dir.parent.mkdir(parents=True, exist_ok=True)

    auth_state = "token detected" if token else "anonymous download"
    console.print(
        Panel(
            Group(
                Text.assemble(("Repository: ", "bold white"), (model_id, "bold cyan")),
                Text.assemble(("Destination: ", "bold white"), (str(model_dir), "yellow")),
                Text.assemble(("Auth: ", "bold white"), (auth_state, "magenta")),
            ),
            title="[bold cyan]Download Plan[/bold cyan]",
            border_style="cyan",
            box=box.HEAVY,
            padding=(1, 2),
        )
    )

    try:
        with console.status("[bold cyan]Downloading model files...[/bold cyan]", spinner="dots"):
            path = snapshot_download(
                repo_id=model_id,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                token=token,
                resume_download=True,
            )
    except Exception as exc:
        print_note(str(exc), title="Download Failed", style="red")
        raise

    print_note(f"Model ready at {path}", title="Download Complete", style="green")
    
    # Launch the chat CLI after download
    console.print("[bold cyan]Launching chat CLI...[/bold cyan]")
    try:
        subprocess.run([sys.executable, "ai_cli.py"], check=True)
    except Exception as exc:
        print_note(f"Failed to launch chat CLI: {exc}", title="Launch Error", style="yellow")


if __name__ == "__main__":
    main()
