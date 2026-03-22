#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
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
                ("smollm3", "bold magenta"),
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
        choices=list(MODEL_CHOICES.keys()),
        help="Model key: tinyllama, smollm2, or smollm3",
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
        "[bold cyan]Download model[/bold cyan]", choices=["1", "2", "3"], default="1"
    )
    if selection == "1":
        return "tinyllama"
    if selection == "2":
        return "smollm2"
    return "qwen"


def model_dir_for(model_id: str) -> Path:
    folder_name = model_id.split("/")[-1]
    return Path("models") / folder_name


def main() -> None:
    render_header()
    args = parse_args()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    model_key = pick_model_key(args.model)
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


if __name__ == "__main__":
    main()
