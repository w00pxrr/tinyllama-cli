#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

MODEL_CHOICES = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "smollm2": "HuggingFaceTB/SmolLM2-135M",
    "smollm3": "HuggingFaceTB/SmolLM3-3B",
}

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a local model into the ./models directory."
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CHOICES.keys()),
        help="Model key: tinyllama, smollm2, or smollm3",
    )
    return parser.parse_args()


def pick_model_key(model_arg: str | None) -> str:
    if model_arg:
        return model_arg

    console.print("[bold]Choose a model to download:[/bold]")
    console.print("1) tinyllama  -> TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    console.print("2) smollm2    -> HuggingFaceTB/SmolLM2-135M")
    console.print("3) smollm3    -> HuggingFaceTB/SmolLM3-3B")

    selection = Prompt.ask(
        "Enter choice", choices=["1", "2", "3"], default="1"
    )
    if selection == "1":
        return "tinyllama"
    if selection == "2":
        return "smollm2"
    return "smollm3"


def model_dir_for(model_id: str) -> Path:
    folder_name = model_id.split("/")[-1]
    return Path("models") / folder_name


def main() -> None:
    args = parse_args()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    model_key = pick_model_key(args.model)
    model_id = MODEL_CHOICES[model_key]
    model_dir = model_dir_for(model_id)

    model_dir.parent.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel.fit(
            f"Downloading [bold cyan]{model_id}[/bold cyan]\n"
            f"to [yellow]{model_dir}[/yellow]",
            title="Model Setup",
            border_style="cyan",
        )
    )

    path = snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        token=token,
        resume_download=True,
    )

    console.print(f"[green]Model ready at:[/green] {path}")


if __name__ == "__main__":
    main()
