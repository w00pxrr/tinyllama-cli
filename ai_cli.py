#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import copy
import ctypes
import json
import operator
import os
import platform
import re
import subprocess
import time  # Move to top for caching
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    tk = None
    ttk = None
    messagebox = None

try:
    import keyboard
except ImportError:
    keyboard = None

# Note: keyboard module requires sudo on macOS. Users can use /settings command instead.

import torch
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
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


@dataclass(slots=True)
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
    MATH_HINTS = ("calculate", "solve", "math", "equation", "evaluate", "multiply", "divide")

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

        if TinyLlamaOptimizer.looks_like_math(text):
            cfg.temperature = 0.0
            cfg.top_p = 1.0
            cfg.top_k = 0
            cfg.repetition_penalty = 1.0
            cfg.max_new_tokens = min(cfg.max_new_tokens, 96)
            cfg.do_sample = False

        # As chat gets longer, keep responses tighter.
        if turns >= 10:
            cfg.max_new_tokens = min(cfg.max_new_tokens, 180)
            cfg.repetition_penalty = max(cfg.repetition_penalty, 1.12)

        return cfg

    @staticmethod
    def looks_like_math(text: str) -> bool:
        normalized = text.lower().replace("×", "*").replace("÷", "/")
        if any(hint in normalized for hint in TinyLlamaOptimizer.MATH_HINTS):
            return bool(re.search(r"\d", normalized))
        return bool(
            re.search(r"\d", normalized)
            and re.search(r"[\+\-\*/%=()]", normalized)
        )


class TinyLlamaCLI:
    USER_PROFILE_FILE = Path("training_data/user_profile.json")
    
    def __init__(self, model_dir: Path, model_label: str, auto_train: bool = False) -> None:
        if not model_dir.exists():
            raise SystemExit(
                "Local model not found. Run `python download_model.py` first to download into models/."
            )
        self.model_dir = model_dir
        self.model_label = model_label
        self.auto_train = auto_train
        self.user_profile = self._load_user_profile()

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
        # Cache context limit once at load time
        self._cached_context_limit: int | None = None
        self._cached_ram_bytes: int | None = None
        
        # Use personalized system prompt with user info
        personalized_prompt = self._get_personalized_system_prompt()
        self.history: list[dict[str, str]] = [{"role": "system", "content": personalized_prompt}]
        self.last_prompt_tokens = 0
        self.last_response_tokens = 0
        self.turn_count = 0
        self.last_user_message = None  # For /like and /dislike
        self.last_assistant_message = None  # For /like and /dislike
        
        # Settings overrides (set by Ctrl+S settings window)
        self._temp_override: float | None = None
        self._top_p_override: float | None = None
        self._top_k_override: int | None = None
        self._max_tokens_override: int | None = None
        self._rep_pen_override: float | None = None
        self._do_sample_override: bool | None = None

    def _accent(self, value: str, style: str = "bold bright_cyan") -> Text:
        return Text(value, style=style)

    def _status_line(self) -> Columns:
        memory = self._available_ram_bytes()
        if memory is None:
            memory_label = "RAM auto"
        else:
            memory_label = f"{memory / (1024 ** 3):.1f} GiB free"

        chips = [
            Panel.fit(self._accent(self.model_label, "bold white"), title="Model", border_style="cyan", box=box.ROUNDED),
            Panel.fit(self._accent(self.device.upper(), "bold green"), title="Device", border_style="green", box=box.ROUNDED),
            Panel.fit(self._accent(str(self._resolve_context_limit()), "bold yellow"), title="Context", border_style="yellow", box=box.ROUNDED),
            Panel.fit(self._accent(memory_label, "bold magenta"), title="Memory", border_style="magenta", box=box.ROUNDED),
            Panel.fit(self._accent("Ctrl+S", "bold blue"), title="Settings", border_style="blue", box=box.ROUNDED),
        ]
        return Columns(chips, expand=True)

    def _print_note(self, message: str, style: str = "cyan", title: str = "Status") -> None:
        console.print(
            Panel(
                Text(message, style=f"bold {style}"),
                title=title,
                border_style=style,
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    def _chat_stats_panel(self, cfg: GenerationConfig | None = None) -> Panel:
        stats = Table.grid(expand=True)
        stats.add_column(justify="center")
        stats.add_column(justify="center")
        stats.add_column(justify="center")
        stats.add_column(justify="center")

        token_budget = str(cfg.max_new_tokens) if cfg else "n/a"
        creativity = f"{cfg.temperature:.2f}" if cfg else "n/a"
        stats.add_row(
            Text(f"{self.turn_count}", style="bold cyan"),
            Text(f"{self.last_prompt_tokens}", style="bold yellow"),
            Text(f"{self.last_response_tokens}", style="bold green"),
            Text(token_budget, style="bold magenta"),
        )
        stats.add_row(
            Text("Turns", style="bright_black"),
            Text("Prompt Tokens", style="bright_black"),
            Text("Reply Tokens", style="bright_black"),
            Text("Token Budget", style="bright_black"),
        )
        stats.add_row(
            Text(self.device.upper(), style="green"),
            Text(str(self._resolve_context_limit()), style="yellow"),
            Text("sample" if (cfg.do_sample if cfg else True) else "greedy", style="cyan"),
            Text(creativity, style="magenta"),
        )
        stats.add_row(
            Text("Device", style="bright_black"),
            Text("Context", style="bright_black"),
            Text("Decoding", style="bright_black"),
            Text("Temp", style="bright_black"),
        )
        return Panel(
            stats,
            title="[bold cyan]Session Stats[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def _render_assistant_panel(self, answer: str) -> None:
        body = Group(
            Text("Local response", style="bold green"),
            Rule(style="green"),
            Markdown(answer or "*(No response returned.)*"),
        )
        console.print(
            Panel(
                body,
                title="[bold green]Assistant[/bold green]",
                subtitle="TinyLlama",
                border_style="green",
                box=box.HEAVY,
                padding=(1, 2),
            )
        )

    def _render_dashboard(self, cfg: GenerationConfig | None = None) -> None:
        console.print(Columns([self._status_line(), self._chat_stats_panel(cfg)], expand=True))

    def _extract_math_expression(self, user_input: str) -> str | None:
        text = normalize_query(user_input).strip()
        lowered = text.lower()
        if not TinyLlamaOptimizer.looks_like_math(lowered):
            return None

        normalized = (
            text.replace("×", "*")
            .replace("÷", "/")
            .replace("^", "**")
            .replace(",", "")
        )
        normalized = re.sub(
            r"(?i)\b(what is|what's|calculate|compute|evaluate|solve|just answer with.*|answer only.*)\b",
            " ",
            normalized,
        )
        normalized = normalized.strip(" ?.=:")

        match = re.search(r"[-+*/%()\d.\s]+(?:\*\*[-+*/%()\d.\s]+)?", normalized)
        if not match:
            return None

        candidate = re.sub(r"\s+", " ", match.group(0)).strip()
        if not re.search(r"\d", candidate) or not re.search(r"[\+\-\*/%]", candidate):
            return None
        return candidate

    def _safe_eval_math(self, expression: str) -> int | float:
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }

        def visit(node: ast.AST) -> int | float:
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = visit(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp) and type(node.op) in operators:
                left = visit(node.left)
                right = visit(node.right)
                if isinstance(node.op, ast.Pow) and abs(right) > 12:
                    raise ValueError("Exponent too large")
                return operators[type(node.op)](left, right)
            raise ValueError("Unsupported expression")

        parsed = ast.parse(expression, mode="eval")
        return visit(parsed)

    def _format_math_result(self, result: int | float) -> str:
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    
    def _rate_last_response(self, positive: bool) -> None:
        """Rate the last response with /like or /dislike for training."""
        if not self.last_user_message or not self.last_assistant_message:
            self._print_note(
                "No previous response to rate. Chat with the AI first!",
                style="yellow",
                title="Rate Error",
            )
            return
        
        rating_dir = Path("training_data/ratings")
        rating_dir.mkdir(parents=True, exist_ok=True)
        
        rating_file = rating_dir / "ratings.jsonl"
        
        rating = {
            "user_message": self.last_user_message,
            "assistant_message": self.last_assistant_message,
            "rating": 1 if positive else -1,
            "timestamp": datetime.now().isoformat(),
        }
        
        with rating_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rating, ensure_ascii=False) + "\n")
        
        if positive:
            self._print_note(
                "Thanks for the positive feedback! ✓\nThis will be used to improve future responses.",
                style="green",
                title="/like",
            )
        else:
            self._print_note(
                "Sorry the response wasn't helpful. 🙏\nThis feedback will help improve the AI.",
                style="yellow",
                title="/dislike",
            )
    
    def _load_user_profile(self) -> dict:
        """Load user profile from file, or prompt for it on first run."""
        if self.USER_PROFILE_FILE.exists():
            try:
                with open(self.USER_PROFILE_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        
        # First run - prompt for user info
        console.print(Panel(
            "[bold cyan]Welcome! Let's set up your profile.[/bold cyan]\n"
            "This helps the AI remember your preferences.",
            border_style="cyan",
            box=box.ROUNDED,
        ))
        
        name = Prompt.ask("[bold cyan]What's your name?[/bold cyan]", default="")
        
        if name:
            # Get more info
            location = Prompt.ask("[bold cyan]Where are you from?[/bold cyan]", default="")
            interests = Prompt.ask("[bold cyan]What are your interests? (comma-separated)[/bold cyan]", default="")
            occupation = Prompt.ask("[bold cyan]What's your occupation?[/bold cyan]", default="")
            
            profile = {
                "name": name,
                "location": location,
                "interests": [i.strip() for i in interests.split(",") if i.strip()],
                "occupation": occupation,
                "created_at": datetime.now().isoformat(),
            }
            
            # Save profile
            self.USER_PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.USER_PROFILE_FILE, "w") as f:
                json.dump(profile, f, indent=2)
            
            console.print(Panel(
                f"[bold green]Profile saved![/bold green]\n"
                f"Name: {name}\n"
                f"Location: {location or 'Not set'}\n"
                f"Interests: {', '.join(profile['interests']) or 'None'}",
                border_style="green",
                box=box.ROUNDED,
            ))
            
            return profile
        
        return {}
    
    def _get_personalized_system_prompt(self) -> str:
        """Get system prompt with user info injected."""
        if not self.user_profile:
            return SYSTEM_PROMPT
        
        profile_parts = []
        if self.user_profile.get("name"):
            profile_parts.append(f"The user's name is {self.user_profile['name']}.")
        if self.user_profile.get("location"):
            profile_parts.append(f"They are from {self.user_profile['location']}.")
        if self.user_profile.get("occupation"):
            profile_parts.append(f"They work as {self.user_profile['occupation']}.")
        if self.user_profile.get("interests"):
            interests = ", ".join(self.user_profile["interests"])
            profile_parts.append(f"Their interests include: {interests}.")
        
        if profile_parts:
            return SYSTEM_PROMPT + " " + " ".join(profile_parts)
        
        return SYSTEM_PROMPT

    def _available_ram_bytes(self) -> int | None:
        # Cache RAM for 30 seconds to avoid repeated system calls
        current_time = time.monotonic()
        
        if hasattr(self, '_cached_ram_time') and hasattr(self, '_cached_ram_bytes'):
            if current_time - self._cached_ram_time < 30:  # 30 second TTL
                return self._cached_ram_bytes
        
        if self.device == "cuda":
            try:
                free_bytes, _total_bytes = torch.cuda.mem_get_info()
                return int(free_bytes)
            except Exception:
                pass

        try:
            if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_AVPHYS_PAGES" in os.sysconf_names:
                page_size = os.sysconf("SC_PAGE_SIZE")
                available_pages = os.sysconf("SC_AVPHYS_PAGES")
                if isinstance(page_size, int) and isinstance(available_pages, int):
                    return int(page_size * available_pages)
        except Exception:
            pass

        system = platform.system()

        if system == "Darwin":
            try:
                vm_stat = subprocess.run(
                    ["vm_stat"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
                page_size_match = re.search(r"page size of (\d+) bytes", vm_stat)
                page_size = int(page_size_match.group(1)) if page_size_match else 4096
                page_counts: dict[str, int] = {}
                for line in vm_stat.splitlines():
                    match = re.match(r"(.+?):\s+(\d+)\.", line)
                    if match:
                        page_counts[match.group(1)] = int(match.group(2))
                available_pages = (
                    page_counts.get("Pages free", 0)
                    + page_counts.get("Pages inactive", 0)
                    + page_counts.get("Pages speculative", 0)
                )
                if available_pages > 0:
                    return int(available_pages * page_size)
            except Exception:
                pass

        if system == "Linux":
            try:
                for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        return int(parts[1]) * 1024
            except Exception:
                pass

        if system == "Windows":
            try:
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                status = MEMORYSTATUSEX()
                status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                    return int(status.ullAvailPhys)
            except Exception:
                pass
        
        return None
    
    def _resolve_context_limit(self) -> int:
        # Cache context limit to avoid repeated config lookups
        if self._cached_context_limit is not None:
            return self._cached_context_limit
        
        model_limit = getattr(self.model.config, "max_position_embeddings", None)
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)

        if isinstance(model_limit, int) and model_limit > 0:
            self._cached_context_limit = model_limit
            return model_limit
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
            self._cached_context_limit = tokenizer_limit
            return tokenizer_limit
        self._cached_context_limit = 2048
        return 2048

    def _dynamic_max_new_tokens(self, prompt_tokens: int, requested_tokens: int) -> int:
        context_limit = self._resolve_context_limit()
        remaining_context = max(32, context_limit - prompt_tokens - 16)
        available_bytes = self._available_ram_bytes()

        if available_bytes is None:
            return max(32, min(requested_tokens, remaining_context))

        available_gib = available_bytes / (1024 ** 3)
        if available_gib < 8:
            memory_cap = 96
        elif available_gib < 12:
            memory_cap = 160
        elif available_gib < 16:
            memory_cap = 224
        elif available_gib < 24:
            memory_cap = 320
        elif available_gib < 32:
            memory_cap = 448
        else:
            memory_cap = 640

        if available_gib >= 12:
            target_tokens = max(requested_tokens, min(memory_cap, requested_tokens + 64))
        else:
            target_tokens = min(requested_tokens, memory_cap)

        return max(32, min(target_tokens, memory_cap, remaining_context))

    def _render_banner(self) -> None:
        headline = Align.center(Text("TinyLlama Local Chat", style="bold white"))
        subtitle = Align.center(
            Text("A sharper terminal interface for local, prompt-aware conversations", style="bright_black")
        )
        command_strip = Align.center(
            Text.assemble(
                ("Try ", "white"),
                ("/help", "bold green"),
                ("  ", "white"),
                ("/settings", "bold yellow"),
                ("  ", "white"),
                ("/save", "bold magenta"),
                ("  ", "white"),
                ("/exit", "bold red"),
            )
        )
        hero = Group(
            headline,
            subtitle,
            Text(""),
            self._status_line(),
            Text(""),
            command_strip,
        )
        console.print(
            Panel(
                hero,
                title="[bold cyan]TinyLlama CLI[/bold cyan]",
                subtitle=str(self.model_dir),
                border_style="cyan",
                box=box.DOUBLE_EDGE,
                padding=(1, 2),
            )
        )

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
        table = Table(title="Active Generation Settings", box=box.ROUNDED, expand=True, show_edge=False)
        table.add_column("Setting", style="bold cyan", no_wrap=True)
        table.add_column("Value", style="white")
        table.add_row("temperature", str(cfg.temperature))
        table.add_row("top_p", str(cfg.top_p))
        table.add_row("top_k", str(cfg.top_k))
        table.add_row("repetition_penalty", str(cfg.repetition_penalty))
        table.add_row("max_new_tokens", str(cfg.max_new_tokens))
        table.add_row("do_sample", str(cfg.do_sample))
        meta = self._status_line()
        console.print(
            Panel(
                Group(meta, Text(""), self._chat_stats_panel(cfg), Text(""), table),
                title="[bold yellow]/settings[/bold yellow]",
                border_style="yellow",
                box=box.HEAVY,
                padding=(1, 2),
            )
        )

    def _open_settings_window(self) -> None:
        """Open a Tkinter window for settings."""
        settings_window = tk.Toplevel()
        settings_window.title("TinyLlama Settings")
        settings_window.geometry("400x500")
        settings_window.resizable(False, False)
        
        # Style configuration
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Helvetica", 14, "bold"))
        style.configure("Header.TLabel", font=("Helvetica", 11, "bold"))
        
        # Title
        title_label = ttk.Label(settings_window, text="Settings", style="Title.TLabel")
        title_label.pack(pady=10)
        
        # Generation Settings Frame
        gen_frame = ttk.LabelFrame(settings_window, text="Generation Settings")
        gen_frame.pack(fill="x", padx=10, pady=5)
        
        # Temperature
        ttk.Label(gen_frame, text="Temperature:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        temp_var = tk.DoubleVar(value=0.65)
        temp_spin = ttk.Spinbox(gen_frame, from_=0.0, to=2.0, increment=0.05, textvariable=temp_var, width=10)
        temp_spin.grid(row=0, column=1, padx=5, pady=5)
        
        # Top P
        ttk.Label(gen_frame, text="Top P:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        top_p_var = tk.DoubleVar(value=0.9)
        top_p_spin = ttk.Spinbox(gen_frame, from_=0.0, to=1.0, increment=0.05, textvariable=top_p_var, width=10)
        top_p_spin.grid(row=1, column=1, padx=5, pady=5)
        
        # Top K
        ttk.Label(gen_frame, text="Top K:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        top_k_var = tk.IntVar(value=40)
        top_k_spin = ttk.Spinbox(gen_frame, from_=1, to=100, increment=1, textvariable=top_k_var, width=10)
        top_k_spin.grid(row=2, column=1, padx=5, pady=5)
        
        # Max New Tokens
        ttk.Label(gen_frame, text="Max New Tokens:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        max_tokens_var = tk.IntVar(value=256)
        max_tokens_spin = ttk.Spinbox(gen_frame, from_=1, to=2048, increment=16, textvariable=max_tokens_var, width=10)
        max_tokens_spin.grid(row=3, column=1, padx=5, pady=5)
        
        # Repetition Penalty
        ttk.Label(gen_frame, text="Repetition Penalty:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        rep_pen_var = tk.DoubleVar(value=1.1)
        rep_pen_spin = ttk.Spinbox(gen_frame, from_=1.0, to=2.0, increment=0.05, textvariable=rep_pen_var, width=10)
        rep_pen_spin.grid(row=4, column=1, padx=5, pady=5)
        
        # Do Sample
        do_sample_var = tk.BooleanVar(value=True)
        do_sample_check = ttk.Checkbutton(gen_frame, text="Do Sample", variable=do_sample_var)
        do_sample_check.grid(row=5, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Model Settings Frame
        model_frame = ttk.LabelFrame(settings_window, text="Model Settings")
        model_frame.pack(fill="x", padx=10, pady=5)
        
        # Model path display (read-only)
        ttk.Label(model_frame, text="Model Path:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        model_path_label = ttk.Label(model_frame, text=str(self.model_dir), foreground="gray")
        model_path_label.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Device display (read-only)
        ttk.Label(model_frame, text="Device:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        device_label = ttk.Label(model_frame, text=self.device.upper(), foreground="gray")
        device_label.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Buttons Frame
        btn_frame = ttk.Frame(settings_window)
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        def save_settings():
            # Update the config for future generations
            self._temp_override = temp_var.get()
            self._top_p_override = top_p_var.get()
            self._top_k_override = top_k_var.get()
            self._max_tokens_override = max_tokens_var.get()
            self._rep_pen_override = rep_pen_var.get()
            self._do_sample_override = do_sample_var.get()
            messagebox.showinfo("Settings", "Settings saved! They will apply to the next message.")
            settings_window.destroy()
        
        def reset_defaults():
            temp_var.set(0.65)
            top_p_var.set(0.9)
            top_k_var.set(40)
            max_tokens_var.set(256)
            rep_pen_var.set(1.1)
            do_sample_var.set(True)
        
        ttk.Button(btn_frame, text="Save", command=save_settings).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Reset Defaults", command=reset_defaults).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=settings_window.destroy).pack(side="right", padx=5)

    def _help(self) -> None:
        table = Table(box=box.SIMPLE_HEAVY, expand=True, show_header=True, header_style="bold cyan")
        table.add_column("Command", style="bold cyan", no_wrap=True)
        table.add_column("Description", style="white")
        table.add_row("/help", "Show command help")
        table.add_row("/web <query>", "Search the web first, then answer with fresh results")
        table.add_row("/clear", "Clear conversation history")
        table.add_row("/settings", "Show model settings for last prompt")
        table.add_row("/stats", "Show the live session dashboard")
        table.add_row("/save", "Save transcript + append training examples")
        table.add_row("/exit", "Quit")
        tips = Panel(
            Text.assemble(
                ("Tips\n", "bold green"),
                ("Ask factual questions for tighter decoding.\n", "white"),
                ("Use ", "white"),
                ("/web", "bold cyan"),
                (" when freshness matters.\n", "white"),
                ("Use ", "white"),
                ("/settings", "bold yellow"),
                (" to inspect the live token budget.", "white"),
            ),
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 2),
        )
        console.print(
            Panel(
                Group(table, Text(""), tips),
                title="[bold cyan]Command Center[/bold cyan]",
                border_style="cyan",
                box=box.HEAVY,
                padding=(1, 2),
            )
        )

    def _save_chat(self, quiet: bool = False) -> Path:
        out_dir = Path("transcripts")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = out_dir / f"chat-{stamp}.json"
        path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        if not quiet:
            self._print_note(f"Saved transcript to {path}", style="green", title="Saved")
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
            self._print_note(
                f"Appended {len(examples)} training examples to {TRAINING_DATA_FILE}",
                style="green",
                title="Training Export",
            )
        return TRAINING_DATA_FILE, len(examples)

    def _save_and_export_training(self, quiet: bool = False, auto_train: bool = False) -> tuple[Path, Path, int]:
        transcript_path = self._save_chat(quiet=quiet)
        training_path, count = self._append_training_data(transcript_path, quiet=quiet)
        
        # Optionally run auto-training
        if auto_train:
            self._run_auto_training(transcript_path)
        
        return transcript_path, training_path, count
    
    def _run_auto_training(self, transcript_path: Path) -> None:
        """Run automatic fine-tuning in background after CLI exits."""
        import sys
        
        console.print()
        console.print(Panel(
            "[bold cyan]Background training will start after exit...[/bold cyan]\n"
            "Training runs in background - you can continue using the CLI.",
            border_style="cyan",
            box=box.ROUNDED,
        ))
        
        # Spawn training in background subprocess - runs AFTER this process exits
        # This way training doesn't block the CLI
        try:
            # Build the command to run training after exit
            cmd = [
                sys.executable, "train.py",
                "--latest",
                "--model", str(self.model_dir),
            ]
            
            # Use Popen to start background process
            # Training will run independently after CLI closes
            subprocess.Popen(
                cmd,
                cwd=str(Path.cwd()),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,  # Detach from parent
            )
            
            console.print(Panel(
                "[bold green]Training scheduled in background[/bold green]\n"
                "It will start after you close the CLI.",
                border_style="green",
                box=box.ROUNDED,
            ))
        except Exception as e:
            console.print(Panel(
                f"[red]Failed to schedule training:[/red]\n{e}",
                title="Error",
                border_style="red",
            ))

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

        table = Table(title="Web Results", box=box.ROUNDED, expand=True, show_lines=False)
        table.add_column("#", style="bold cyan", no_wrap=True)
        table.add_column("Title", style="white")
        table.add_column("Domain", style="green", no_wrap=True)
        for idx, result in enumerate(results, start=1):
            domain = re.sub(r"^www\.", "", result.url.split("/")[2]) if "://" in result.url else result.url
            table.add_row(str(idx), result.title, domain)
        console.print(
            Panel(
                table,
                title="[bold cyan]Fresh Context[/bold cyan]",
                border_style="cyan",
                box=box.HEAVY,
                padding=(1, 2),
            )
        )
        return WEB_PROMPT_TEMPLATE.format(web_context=format_web_context(results)), results

    def _reset_session(self) -> None:
        """Reset the conversation history when context limit is exceeded."""
        personalized_prompt = self._get_personalized_system_prompt()
        self.history = [{"role": "system", "content": personalized_prompt}]
        self.last_prompt_tokens = 0
        self.last_response_tokens = 0
        self.turn_count = 0
        self._print_note(
            "Context limit reached - starting fresh session",
            style="yellow",
            title="Session Reset",
        )

    def _reply(self, user_input: str) -> GenerationConfig:
        cfg = TinyLlamaOptimizer.tune(user_input, turns=len(self.history))
        
        # Apply settings overrides from Ctrl+S settings window
        if self._temp_override is not None:
            cfg.temperature = self._temp_override
        if self._top_p_override is not None:
            cfg.top_p = self._top_p_override
        if self._top_k_override is not None:
            cfg.top_k = self._top_k_override
        if self._max_tokens_override is not None:
            cfg.max_new_tokens = self._max_tokens_override
        if self._rep_pen_override is not None:
            cfg.repetition_penalty = self._rep_pen_override
        if self._do_sample_override is not None:
            cfg.do_sample = self._do_sample_override
        
        normalized_input = normalize_query(user_input)
        self.history.append({"role": "user", "content": normalized_input})
        self.last_user_message = user_input  # Track for /like and /dislike

        expression = self._extract_math_expression(user_input)
        if expression is not None:
            try:
                result = self._safe_eval_math(expression)
            except Exception:
                pass
            else:
                answer = self._format_math_result(result)
                self.last_prompt_tokens = 0
                self.last_response_tokens = len(self.tokenizer.encode(answer, add_special_tokens=False))
                self.turn_count += 1
                self.last_user_message = user_input  # Track for /like and /dislike
                self.last_assistant_message = answer  # Track for /like and /dislike
                self.history.append({"role": "assistant", "content": answer})
                self._render_assistant_panel(answer)
                return cfg

        extra_system, _results = self._fetch_web_context(user_input)
        prompt = self._prompt_template(extra_system=extra_system)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        self.last_prompt_tokens = int(input_ids.shape[1])
        
        # Check if context limit exceeded - reset session if needed
        context_limit = self._resolve_context_limit()
        buffer_tokens = 64  # Small buffer to ensure we don't hit exact limit
        if self.last_prompt_tokens > context_limit - buffer_tokens:
            # Context limit would be exceeded - reset the session
            self._reset_session()
            # Re-build the prompt with fresh history
            prompt = self._prompt_template(extra_system=extra_system)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)
            self.last_prompt_tokens = int(input_ids.shape[1])
        
        cfg.max_new_tokens = self._dynamic_max_new_tokens(
            prompt_tokens=self.last_prompt_tokens,
            requested_tokens=cfg.max_new_tokens,
        )

        generation_cfg = copy.deepcopy(self.model.generation_config)
        # Only use max_new_tokens, not max_length (avoids conflict)
        generation_cfg.max_new_tokens = cfg.max_new_tokens
        generation_cfg.temperature = cfg.temperature
        generation_cfg.top_p = cfg.top_p
        generation_cfg.top_k = cfg.top_k
        generation_cfg.repetition_penalty = cfg.repetition_penalty
        generation_cfg.do_sample = cfg.do_sample
        generation_cfg.eos_token_id = self.tokenizer.eos_token_id
        generation_cfg.pad_token_id = self.tokenizer.eos_token_id

        with console.status("[bold cyan]TinyLlama is thinking...[/bold cyan]", spinner="dots"):
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_cfg,
                )

        new_tokens = output_ids[0, input_ids.shape[1] :]
        self.last_response_tokens = int(new_tokens.shape[0])
        self.turn_count += 1
        answer = self._clean_output(self.tokenizer.decode(new_tokens, skip_special_tokens=False))
        self.history.append({"role": "assistant", "content": answer})
        self.last_assistant_message = answer  # Track for /like and /dislike
        self._render_assistant_panel(answer)
        return cfg

    def run(self) -> None:
        self._render_banner()
        last_cfg = GenerationConfig()
        
        # Register Ctrl+S hotkey to open settings (if keyboard module available)
        # Note: Keyboard module requires sudo on macOS, so we wrap in try/except
        def open_settings():
            if tk is not None:
                try:
                    self._open_settings_window()
                except Exception:
                    # Tkinter not available, fall back to rich panel
                    self._show_settings(last_cfg)
            else:
                self._show_settings(last_cfg)
        
        if keyboard is not None:
            try:
                keyboard.add_hotkey('ctrl+s', open_settings)
            except Exception:
                # Keyboard module failed (needs sudo or incompatible) - use /settings instead
                console.print("[dim]Ctrl+S hotkey not available. Use /settings to open settings.[/dim]")
        
        try:
            while True:
                user_input = Prompt.ask(
                    "\n[bold cyan]You[/bold cyan]",
                    console=console,
                ).strip()
                if not user_input:
                    continue

                if user_input == "/exit":
                    self._print_note("Session closed. See you next time.", style="magenta", title="Bye")
                    import sys
                    sys.exit(0)
                if user_input == "/help":
                    self._help()
                    continue
                if user_input == "/clear":
                    self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
                    self.last_prompt_tokens = 0
                    self.last_response_tokens = 0
                    self.turn_count = 0
                    self._print_note("Conversation cleared and system prompt reset.", style="yellow", title="Cleared")
                    continue
                if user_input == "/save":
                    self._save_and_export_training(auto_train=self.auto_train)
                    continue
                if user_input == "/like":
                    self._rate_last_response(positive=True)
                    continue
                if user_input == "/dislike":
                    self._rate_last_response(positive=False)
                    continue
                if user_input == "/settings":
                    self._show_settings(last_cfg)
                    continue
                if user_input == "/stats":
                    self._render_dashboard(last_cfg)
                    continue

                try:
                    last_cfg = self._reply(user_input)
                except Exception as exc:
                    console.print(Panel(str(exc), title="Error", border_style="red"))
        except (EOFError, KeyboardInterrupt):
            self._print_note("Session interrupted. Exiting cleanly.", style="yellow", title="Interrupted")
        finally:
            transcript_path, training_path, count = self._save_and_export_training(quiet=True, auto_train=self.auto_train)
            self._print_note(f"Auto-saved transcript to {transcript_path}", style="green", title="Auto Save")
            self._print_note(
                f"Auto-appended {count} training examples to {training_path}",
                style="green",
                title="Auto Export",
            )


# Model strength ranking (higher = more powerful)
MODEL_STRENGTH = {
    "SmolLM2-135M": 1,
    "TinyLlama-1.1B-Chat-v1.0": 2,
    "Qwen2.5-0.5B-Instruct": 3,
    "NVIDIA-Nemotron": 4,
}


def assess_task_difficulty(user_input: str) -> str:
    """Assess task difficulty based on user input.
    
    Returns:
        "simple" for basic queries, "complex" for advanced tasks
    """
    text = user_input.lower().strip()
    
    # Complex task indicators
    complex_indicators = [
        "write a", "create a", "build a", "implement", "develop",
        "explain in detail", "compare and contrast", "analyze",
        "debug", "fix the code", "refactor", "architecture",
        "design a", "plan a", "how would you", "what are the best",
        "traduire", "escribe", "schreibe",  # foreign languages
        "essay", "research", "report", "summary of",
        "math", "calculate", "solve equation",
        "code", "program", "function", "class", "api",
        "story", "poem", "creative writing",
    ]
    
    # Simple task indicators  
    simple_indicators = [
        "what is", "who is", "when did", "where is", "how do i",
        "define", "what's the", "what does", "meaning of",
        "quick", "simple", "basic", "just",
        "hi", "hello", "hey", "thanks", "thank you",
    ]
    
    # Check complexity
    complex_score = sum(1 for indicator in complex_indicators if indicator in text)
    simple_score = sum(1 for indicator in simple_indicators if indicator in text)
    
    # Length also factors in
    if len(text) > 200:
        complex_score += 1
    if len(text) > 500:
        complex_score += 1
    
    if complex_score > simple_score:
        return "complex"
    return "simple"


def get_model_strength(model_name: str) -> int:
    """Get the strength level of a model by name."""
    for key, strength in MODEL_STRENGTH.items():
        if key.lower() in model_name.lower():
            return strength
    return 2  # default to medium


def discover_installed_models(models_root: Path = Path("models")) -> list[Path]:
    if not models_root.exists():
        return []
    model_dirs = []
    for child in sorted(models_root.iterdir()):
        if child.is_dir():
            # Check for either config.json (HuggingFace format) or .gguf files (llama.cpp format)
            has_config = (child / "config.json").exists()
            has_gguf = list(child.glob("*.gguf"))
            if has_config or has_gguf:
                model_dirs.append(child)
    return model_dirs


def select_installed_model(model_arg: str | None) -> tuple[Path, str]:
    installed = discover_installed_models()
    if not installed:
        raise SystemExit(
            "No installed models found in ./models. Run `python download_model.py` first."
        )

    # Handle explicit model argument (including 'auto')
    if model_arg:
        if model_arg.lower() == "auto":
            return select_model_auto(installed)
        
        candidate = Path(model_arg)
        if candidate.exists() and candidate.is_dir():
            return candidate, candidate.name

        for model_dir in installed:
            if model_dir.name == model_arg:
                return model_dir, model_dir.name
            # Also check for partial matches (e.g., "nvidia_nemotron" matches "NVIDIA-Nemotron-3-Nano-4B-GGUF")
            if model_arg.lower().replace("_", "-") in model_dir.name.lower():
                return model_dir, model_dir.name
        raise SystemExit(
            f"Model '{model_arg}' not found. Installed: {', '.join(m.name for m in installed)}"
        )

    table = Table(box=box.SIMPLE_HEAVY, expand=True, header_style="bold cyan")
    table.add_column("#", style="bold cyan", no_wrap=True)
    table.add_column("Installed Model", style="white")
    table.add_column("Path", style="yellow")
    
    # Add auto option at the top
    table.add_row("A", "[bold green]Auto[/bold green]", "Smart selection based on task")
    
    for idx, model_dir in enumerate(installed, start=1):
        strength = get_model_strength(model_dir.name)
        strength_label = "★" * strength
        table.add_row(str(idx), model_dir.name, f"{str(model_dir)} {strength_label}")
    
    console.print(
        Panel(
            Group(
                Text("Choose an installed model to launch", style="bold white"),
                Text("Each folder under ./models with a config.json is listed below.", style="bright_black"),
                Text("Use [bold green]A[/bold green] for Auto mode - it selects the right model based on your query.", style="bold cyan"),
                Text(""),
                table,
            ),
            title="[bold cyan]Model Picker[/bold cyan]",
            border_style="cyan",
            box=box.HEAVY,
            padding=(1, 2),
        )
    )
    
    choices = ["A"] + [str(i) for i in range(1, len(installed) + 1)]
    selection = Prompt.ask("[bold cyan]Launch model[/bold cyan]", choices=choices, default="A")
    
    if selection.upper() == "A":
        return select_model_auto(installed)
    
    chosen = installed[int(selection) - 1]
    return chosen, chosen.name


def select_model_auto(installed: list[Path]) -> tuple[Path, str]:
    """Automatically select a model based on the user's intended task."""
    console.print()
    console.print(Panel(
        Text(
            "Auto mode: Tell me what you want to discuss or what task you need help with.\n"
            "I'll select the appropriate model based on the complexity of your request.\n\n"
            "Examples:\n"
            "  • Simple questions → TinyLlama (fast, efficient)\n"
            "  • Complex tasks → SmolLM3 (more capable)",
            style="bold cyan"
        ),
        title="[bold green]Auto Model Selection[/bold green]",
        border_style="green",
        box=box.ROUNDED,
        padding=(1, 2),
    ))
    
    user_input = Prompt.ask("[bold green]What would you like to do?[/bold green]").strip()
    
    if not user_input:
        console.print("[yellow]No input provided, defaulting to first model.[/yellow]")
        chosen = installed[0]
        return chosen, chosen.name
    
    difficulty = assess_task_difficulty(user_input)
    
    # Sort models by strength
    sorted_models = sorted(installed, key=lambda m: get_model_strength(m.name))
    
    if difficulty == "complex":
        # Use the most powerful available model
        chosen = sorted_models[-1]
        console.print(f"[bold cyan]Detected complex task → Using {chosen.name}[/bold cyan]")
    else:
        # Use a weaker but faster model for simple tasks
        chosen = sorted_models[0]
        console.print(f"[bold cyan]Detected simple task → Using {chosen.name}[/bold cyan]")
    
    return chosen, chosen.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local chat CLI with installed-model selection."
    )
    parser.add_argument(
        "--model",
        help="Installed model folder name (inside ./models), full local path, or 'auto' for smart selection.",
    )
    parser.add_argument(
        "--auto-train",
        action="store_true",
        help="Automatically fine-tune model after each conversation using LoRA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir, model_label = select_installed_model(args.model)
    cli = TinyLlamaCLI(model_dir=model_dir, model_label=model_label, auto_train=args.auto_train)
    cli.run()


if __name__ == "__main__":
    main()
