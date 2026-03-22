#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
MODEL_DIR="$PROJECT_ROOT/models"
REQ_FILE="$PROJECT_ROOT/requirements.txt"

log() {
  printf "[tinyllama] %s\n" "$*"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

run_as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif have_cmd sudo; then
    sudo "$@"
  else
    log "Need elevated privileges to run: $*"
    exit 1
  fi
}

install_python_macos() {
  log "Python not found. Installing with Homebrew..."

  if ! have_cmd brew; then
    log "Homebrew is not installed. Install Homebrew first: https://brew.sh"
    exit 1
  fi

  brew install python
}

install_python_windows() {
  log "Python not found. Installing with winget..."

  if ! have_cmd winget; then
    log "winget not found. Install App Installer from Microsoft Store and retry."
    exit 1
  fi

  winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
}

install_python_linux() {
  log "Python not found. Detecting Linux package manager..."

  if have_cmd apt-get; then
    run_as_root apt-get update
    run_as_root apt-get install -y python3 python3-venv python3-pip
    return
  fi

  if have_cmd pacman; then
    run_as_root pacman -Sy --noconfirm python python-pip
    return
  fi

  if have_cmd dnf; then
    run_as_root dnf install -y python3 python3-pip
    return
  fi

  if have_cmd yum; then
    run_as_root yum install -y python3 python3-pip
    return
  fi

  if have_cmd zypper; then
    run_as_root zypper --non-interactive install python3 python3-pip
    return
  fi

  if have_cmd rpm; then
    log "Detected RPM-based system, but no supported installer (dnf/yum/zypper) was found."
    log "Please install Python 3 manually, then rerun this script."
    exit 1
  fi

  log "No supported Linux package manager found (apt, pacman, dnf/yum/zypper)."
  exit 1
}

ensure_python() {
  if have_cmd python3; then
    PYTHON_BIN="python3"
    return
  fi

  if have_cmd python; then
    PYTHON_BIN="python"
    return
  fi

  os_name="$(uname -s 2>/dev/null || echo unknown)"

  case "$os_name" in
    Darwin)
      install_python_macos
      ;;
    Linux)
      install_python_linux
      ;;
    MINGW*|MSYS*|CYGWIN*)
      install_python_windows
      ;;
    *)
      log "Unsupported OS: $os_name"
      exit 1
      ;;
  esac

  if have_cmd python3; then
    PYTHON_BIN="python3"
  elif have_cmd python; then
    PYTHON_BIN="python"
  else
    log "Python install appears to have failed (python not found in PATH)."
    exit 1
  fi
}

create_venv() {
  if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment in $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  else
    log "Using existing virtual environment: $VENV_DIR"
  fi
}

activate_venv() {
  if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
  elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/Scripts/activate"
  else
    log "Could not find virtualenv activate script."
    exit 1
  fi
}

install_dependencies() {
  if [ ! -f "$REQ_FILE" ]; then
    log "Missing requirements.txt at $REQ_FILE"
    exit 1
  fi

  log "Installing Python dependencies..."
  python -m pip install --upgrade pip
  python -m pip install -r "$REQ_FILE"
}

has_installed_model() {
  if [ ! -d "$MODEL_DIR" ]; then
    return 1
  fi
  if find "$MODEL_DIR" -mindepth 2 -maxdepth 2 -name config.json | grep -q .; then
    return 0
  fi
  return 1
}

bootstrap_only_if_requested() {
  if [ "${BOOTSTRAP_ONLY:-0}" != "1" ]; then
    return
  fi

  if has_installed_model; then
    log "Bootstrap complete. Model already installed."
    exit 0
  fi

  if [ -n "${AUTO_DOWNLOAD_MODEL:-}" ]; then
    log "No installed model found. Auto-downloading: ${AUTO_DOWNLOAD_MODEL}"
    python download_model.py --model "${AUTO_DOWNLOAD_MODEL}"
    if has_installed_model; then
      log "Bootstrap complete. Model download finished."
      exit 0
    fi
    log "Bootstrap finished, but no installed model was detected."
    exit 1
  fi

  log "Bootstrap complete, but no model is installed."
  log "Run: python download_model.py --model tinyllama|smollm2|smollm3"
  exit 0
}

run_cli_if_models_present() {
  if ! has_installed_model; then
    log "No installed model found in $MODEL_DIR."
    log "Run: python download_model.py"
    exit 0
  fi

  cd "$PROJECT_ROOT"
  log "Launching CLI..."
  python chat.py
}

main() {
  cd "$PROJECT_ROOT"
  ensure_python
  log "Using Python: $PYTHON_BIN"
  create_venv
  activate_venv
  install_dependencies
  bootstrap_only_if_requested
  run_cli_if_models_present
}

main "$@"
