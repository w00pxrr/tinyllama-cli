#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = [
    "ai_cli.py",
    "chat.py",
    "download_model.py",
    "tinyllama.sh",
    "requirements.txt",
    "README.md",
    ".env.example",
]


def copy_item(src: Path, dst: Path) -> None:
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble distributable runtime bundle.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True, help="Output directory for packaged files")
    args = parser.parse_args()

    out = Path(args.output).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    app_dir = out / "app"
    app_dir.mkdir(parents=True, exist_ok=True)

    for rel in RUNTIME_FILES:
        src = ROOT / rel
        if src.exists():
            copy_item(src, app_dir / rel)

    models_dir = app_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.target.startswith("windows"):
        (app_dir / "run_cli.bat").write_text(
            "@echo off\r\n"
            "call .venv\\Scripts\\activate.bat\r\n"
            "python chat.py\r\n",
            encoding="utf-8",
        )
    else:
        (app_dir / "run_cli.sh").write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "./tinyllama.sh\n",
            encoding="utf-8",
        )
        (app_dir / "run_cli.sh").chmod(0o755)


if __name__ == "__main__":
    main()
