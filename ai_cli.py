#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from transformers import AutoModelForCausalLM, AutoTokenizer

from web_search import WebResult, format_web_context, normalize_query, search_web, should_search_web

DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_MODEL_DIR = Path("models") / "TinyLlama-1.1B-Chat-v1.0"
TRAINING_DATA_FILE = Path("training_data") / "tinyllama_sft.jsonl"
SYSTEM_PROMPT = (
    "You are a helpful, concise AI assistant. Keep answers clear and practical. "
    "When unsure, say what you are uncertain about."
)
WEB_PROMPT_TEMPLATE = (
    "You have access to recent web search results for the user's request.\n"
    "Use them when they are relevant, prefer them over stale model memory for current facts, "
    "and mention the source domains you relied on.\n\n"
    "Web results:\n{web_context}"
)

console = Console()


@dataclass
class GenerationConfig:
    temperature: float = 0.65
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    max_new_tokens: int = 256
    do_sample: bool = True


class TinyLlamaOptimizer:
    """Lightweight runtime tuning for TinyLlama chat generation."""

    FACTUAL_HINTS = ("explain", "what is", "why", "how does", "difference", "compare")
    CREATIVE_HINTS = ("story", "poem", "creative", "brainstorm", "ideas", "fiction")
    CODE_HINTS = ("code", "python", "bug", "error", "function", "class", "api")

    @staticmethod
    def tune(user_input: str, turns: int) -> GenerationConfig:
        text = user_input.lower().strip()
        cfg = GenerationConfig()

        # Long contexts benefit from tighter decoding and shorter outputs.
        if len(text) > 420:
            cfg.max_new_tokens = 192
            cfg.temperature = 0.55
            cfg.top_p = 0.86

        if any(hint in text for hint in TinyLlamaOptimizer.FACTUAL_HINTS):
            cfg.temperature = 0.45
            cfg.top_p = 0.82
            cfg.max_new_tokens = min(cfg.max_new_tokens, 220)

        if any(hint in text for hint in TinyLlamaOptimizer.CODE_HINTS):
            cfg.temperature = 0.4
            cfg.top_p = 0.85
            cfg.repetition_penalty = 1.08
            cfg.max_new_tokens = max(cfg.max_new_tokens, 280)

        if any(hint in text for hint in TinyLlamaOptimizer.CREATIVE_HINTS):
            cfg.temperature = 0.88
            cfg.top_p = 0.95
            cfg.top_k = 60
            cfg.max_new_tokens = 320

        # As chat gets longer, keep responses tighter.
        if turns >= 10:
            cfg.max_new_tokens = min(cfg.max_new_tokens, 180)
            cfg.repetition_penalty = max(cfg.repetition_penalty, 1.12)

        return cfg


class TinyLlamaCLI:
    def __init__(self, model_dir: Path, model_label: str) -> None:
        if not model_dir.exists():
            raise SystemExit(
                "Local model not found. Run `python download_model.py` first to download into models/."
            )
        self.model_dir = model_dir
        self.model_label = model_label

        with console.status("[bold cyan]Loading tokenizer...[/bold cyan]", spinner="dots"):
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), use_fast=True)

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        with console.status("[bold cyan]Loading model...[/bold cyan]", spinner="dots"):
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_dir),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(device)

        self.device = device
        self.history: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _render_banner(self) -> None:
        title = Text(" TinyLlama CLI (Local) ", style="bold black on cyan")
        subtitle = Text("Beautiful terminal chat for TinyLlama-1.1B-Chat", style="bold bright_white")
        body = Text()
        body.append("Model: ", style="bold")
        body.append(f"{self.model_label}\n", style="bright_cyan")
        body.append("Path: ", style="bold")
        body.append(f"{self.model_dir}\n", style="yellow")
        body.append("Device: ", style="bold")
        body.append(f"{self.device}\n", style="green")
        body.append("Type ", style="white")
        body.append("/help", style="bold green")
        body.append(" for commands", style="white")
        console.print(Panel.fit(Text.assemble(title, "\n", subtitle, "\n\n", body), box=box.DOUBLE, border_style="cyan"))

    def _prompt_template(self, extra_system: str | None = None) -> str:
        # TinyLlama chat format used by many instruct checkpoints.
        chunks = []
        for msg in self.history:
            role = msg["role"]
            content = msg["content"].strip()
            if role == "system":
                chunks.append(f"<|system|>\n{content}</s>")
            elif role == "user":
                chunks.append(f"<|user|>\n{content}</s>")
            elif role == "assistant":
                chunks.append(f"<|assistant|>\n{content}</s>")
        if extra_system:
            chunks.append(f"<|system|>\n{extra_system.strip()}</s>")
        chunks.append("<|assistant|>\n")
        return "\n".join(chunks)

    def _clean_output(self, text: str) -> str:
        text = re.sub(r"</s>.*$", "", text, flags=re.DOTALL)
        return text.strip()

    def _show_settings(self, cfg: GenerationConfig) -> None:
        table = Table(title="Active Generation Settings", box=box.SIMPLE_HEAVY)
        table.add_column("Setting", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")
        table.add_row("temperature", str(cfg.temperature))
        table.add_row("top_p", str(cfg.top_p))
        table.add_row("top_k", str(cfg.top_k))
        table.add_row("repetition_penalty", str(cfg.repetition_penalty))
        table.add_row("max_new_tokens", str(cfg.max_new_tokens))
        table.add_row("do_sample", str(cfg.do_sample))
        console.print(table)

    def _help(self) -> None:
        table = Table(title="Commands", box=box.ROUNDED)
        table.add_column("Command", style="bold cyan")
        table.add_column("Description", style="white")
        table.add_row("/help", "Show command help")
        table.add_row("/web <query>", "Search the web first, then answer with fresh results")
        table.add_row("/clear", "Clear conversation history")
        table.add_row("/settings", "Show model settings for last prompt")
        table.add_row("/save", "Save transcript + append training examples")
        table.add_row("/exit", "Quit")
        console.print(table)

    def _save_chat(self, quiet: bool = False) -> Path:
        out_dir = Path("transcripts")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = out_dir / f"chat-{stamp}.json"
        path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        if not quiet:
            console.print(f"[green]Saved transcript:[/green] {path}")
        return path

    def _chat_turn_examples(self) -> list[dict[str, object]]:
        system_msg = SYSTEM_PROMPT
        for msg in self.history:
            if msg.get("role") == "system":
                system_msg = msg.get("content", SYSTEM_PROMPT)
                break

        examples: list[dict[str, object]] = []
        pending_user: str | None = None
        for msg in self.history:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role == "user":
                pending_user = content
            elif role == "assistant" and pending_user:
                examples.append(
                    {
                        "messages": [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": pending_user},
                            {"role": "assistant", "content": content},
                        ]
                    }
                )
                pending_user = None
        return examples

    def _append_training_data(self, transcript_path: Path, quiet: bool = False) -> tuple[Path, int]:
        training_dir = TRAINING_DATA_FILE.parent
        training_dir.mkdir(parents=True, exist_ok=True)
        examples = self._chat_turn_examples()

        with TRAINING_DATA_FILE.open("a", encoding="utf-8") as f:
            for idx, ex in enumerate(examples):
                record = {
                    "id": f"{transcript_path.stem}-{idx}",
                    "source_transcript": str(transcript_path),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "messages": ex["messages"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if not quiet:
            console.print(
                f"[green]Appended {len(examples)} training examples:[/green] {TRAINING_DATA_FILE}"
            )
        return TRAINING_DATA_FILE, len(examples)

    def _save_and_export_training(self, quiet: bool = False) -> tuple[Path, Path, int]:
        transcript_path = self._save_chat(quiet=quiet)
        training_path, count = self._append_training_data(transcript_path, quiet=quiet)
        return transcript_path, training_path, count

    def _fetch_web_context(self, user_input: str) -> tuple[str | None, list[WebResult]]:
        if not should_search_web(user_input):
            return None, []

        query = normalize_query(user_input)
        if not query:
            return None, []

        try:
            with console.status("[bold cyan]Searching the web...[/bold cyan]", spinner="dots"):
                results = search_web(query)
        except Exception as exc:
            console.print(Panel(str(exc), title="Web Search Failed", border_style="yellow"))
            return None, []

        if not results:
            return None, []

        table = Table(title="Web Results", box=box.SIMPLE_HEAVY)
        table.add_column("#", style="cyan", no_wrap=True)
        table.add_column("Title", style="white")
        table.add_column("Domain", style="green")
        for idx, result in enumerate(results, start=1):
            domain = re.sub(r"^www\.", "", result.url.split("/")[2]) if "://" in result.url else result.url
            table.add_row(str(idx), result.title, domain)
        console.print(table)
        return WEB_PROMPT_TEMPLATE.format(web_context=format_web_context(results)), results

    def _reply(self, user_input: str) -> GenerationConfig:
        cfg = TinyLlamaOptimizer.tune(user_input, turns=len(self.history))
        extra_system, _results = self._fetch_web_context(user_input)
        normalized_input = normalize_query(user_input)
        self.history.append({"role": "user", "content": normalized_input})
        prompt = self._prompt_template(extra_system=extra_system)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        with console.status("[bold cyan]TinyLlama is thinking...[/bold cyan]", spinner="dots"):
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                    repetition_penalty=cfg.repetition_penalty,
                    do_sample=cfg.do_sample,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        new_tokens = output_ids[0, input_ids.shape[1] :]
        answer = self._clean_output(self.tokenizer.decode(new_tokens, skip_special_tokens=False))
        self.history.append({"role": "assistant", "content": answer})
        console.print(Panel(Markdown(answer or "*(No response returned.)*"), title="Assistant", border_style="green"))
        return cfg

    def run(self) -> None:
        self._render_banner()
        last_cfg = GenerationConfig()
        try:
            while True:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
                if not user_input:
                    continue

                if user_input == "/exit":
                    console.print("[bold magenta]Bye.[/bold magenta]")
                    return
                if user_input == "/help":
                    self._help()
                    continue
                if user_input == "/clear":
                    self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
                    console.print("[yellow]Conversation cleared.[/yellow]")
                    continue
                if user_input == "/save":
                    self._save_and_export_training()
                    continue
                if user_input == "/settings":
                    self._show_settings(last_cfg)
                    continue

                try:
                    last_cfg = self._reply(user_input)
                except Exception as exc:
                    console.print(Panel(str(exc), title="Error", border_style="red"))
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Session interrupted. Exiting...[/yellow]")
        finally:
            transcript_path, training_path, count = self._save_and_export_training(quiet=True)
            console.print(f"[green]Auto-saved transcript:[/green] {transcript_path}")
            console.print(
                f"[green]Auto-appended {count} training examples:[/green] {training_path}"
            )


def discover_installed_models(models_root: Path = Path("models")) -> list[Path]:
    if not models_root.exists():
        return []
    model_dirs = []
    for child in sorted(models_root.iterdir()):
        if child.is_dir() and (child / "config.json").exists():
            model_dirs.append(child)
    return model_dirs


def select_installed_model(model_arg: str | None) -> tuple[Path, str]:
    installed = discover_installed_models()
    if not installed:
        raise SystemExit(
            "No installed models found in ./models. Run `python download_model.py` first."
        )

    if model_arg:
        candidate = Path(model_arg)
        if candidate.exists() and candidate.is_dir():
            return candidate, candidate.name

        for model_dir in installed:
            if model_dir.name == model_arg:
                return model_dir, model_dir.name
        raise SystemExit(
            f"Model '{model_arg}' not found. Installed: {', '.join(m.name for m in installed)}"
        )

    if len(installed) == 1:
        only = installed[0]
        return only, only.name

    console.print("[bold]Choose an installed model:[/bold]")
    for idx, model_dir in enumerate(installed, start=1):
        console.print(f"{idx}) {model_dir.name}  -> {model_dir}")
    choices = [str(i) for i in range(1, len(installed) + 1)]
    selection = Prompt.ask("Enter choice", choices=choices, default="1")
    chosen = installed[int(selection) - 1]
    return chosen, chosen.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local chat CLI with installed-model selection."
    )
    parser.add_argument(
        "--model",
        help="Installed model folder name (inside ./models) or full local path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir, model_label = select_installed_model(args.model)
    cli = TinyLlamaCLI(model_dir=model_dir, model_label=model_label)
    cli.run()


if __name__ == "__main__":
    main()
