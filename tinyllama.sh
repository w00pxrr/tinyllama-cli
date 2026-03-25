#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
MODEL_DIR="$PROJECT_ROOT/models"
REQ_FILE="$PROJECT_ROOT/requirements.txt"
GUI_DIR="$PROJECT_ROOT/tinyllama_gui"

if [ -t 1 ]; then
  C_RESET="$(printf '\033[0m')"
  C_BOLD="$(printf '\033[1m')"
  C_DIM="$(printf '\033[2m')"
  C_CYAN="$(printf '\033[36m')"
  C_GREEN="$(printf '\033[32m')"
  C_YELLOW="$(printf '\033[33m')"
  C_RED="$(printf '\033[31m')"
  C_MAGENTA="$(printf '\033[35m')"
else
  C_RESET=""
  C_BOLD=""
  C_DIM=""
  C_CYAN=""
  C_GREEN=""
  C_YELLOW=""
  C_RED=""
  C_MAGENTA=""
fi

print_header() {
  printf "\n%s%sTinyLlama Bootstrap%s\n" "$C_BOLD" "$C_CYAN" "$C_RESET"
  printf "%sPrepare Python, install dependencies, build GUI, and launch%s\n\n" "$C_DIM" "$C_RESET"
}

print_rule() {
  printf "%s%s== %s ==%s\n" "$C_BOLD" "$C_CYAN" "$1" "$C_RESET"
}

log() {
  printf "%s[%sinfo%s]%s %s\n" "$C_BOLD" "$C_CYAN" "$C_RESET" "$C_RESET" "$*"
}

success() {
  printf "%s[%s ok %s]%s %s\n" "$C_BOLD" "$C_GREEN" "$C_RESET" "$C_RESET" "$*"
}

warn() {
  printf "%s[%swarn%s]%s %s\n" "$C_BOLD" "$C_YELLOW" "$C_RESET" "$C_RESET" "$*"
}

fail() {
  printf "%s[%serr %s]%s %s\n" "$C_BOLD" "$C_RED" "$C_RESET" "$C_RESET" "$*" >&2
}

hint() {
  printf "%s%s%s\n" "$C_DIM" "$*" "$C_RESET"
}

show_environment() {
  printf "%sProject%s  %s\n" "$C_BOLD" "$C_RESET" "$PROJECT_ROOT"
  printf "%sVenv%s     %s\n" "$C_BOLD" "$C_RESET" "$VENV_DIR"
  printf "%sModels%s   %s\n" "$C_BOLD" "$C_RESET" "$MODEL_DIR"
  printf "%sGUI%s      %s\n" "$C_BOLD" "$C_RESET" "$GUI_DIR"
  printf "\n"
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
    fail "Need elevated privileges to run: $*"
    exit 1
  fi
}

ensure_node() {
  if have_cmd node && have_cmd npm; then
    success "Using Node.js: $(node --version)"
    return
  fi
  
  warn "Node.js not found. Installing via nvm..."
  
  if ! have_cmd curl; then
    fail "curl is required to install Node.js"
    exit 1
  fi
  
  # Install nvm
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
  
  # Source nvm
  export NVM_DIR="$HOME/.nvm"
  # shellcheck disable=SC1091
  [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
  
  nvm install --lts
  nvm use --lts
  
  if have_cmd node && have_cmd npm; then
    success "Using Node.js: $(node --version)"
  else
    fail "Node.js installation appears to have failed."
    exit 1
  fi
}

build_gui() {
  if [ -d "$GUI_DIR/release/mac-arm64/TinyLlama GUI.app" ]; then
    success "Using existing GUI app"
    return
  fi
  
  if [ -d "$GUI_DIR/dist" ] && [ -f "$GUI_DIR/package.json" ]; then
    success "Using existing GUI build"
    return
  fi
  
  log "Building GUI (this may take a few minutes)..."
  
  cd "$GUI_DIR"
  
  # Install dependencies if needed
  if [ ! -d "node_modules" ]; then
    if have_cmd pnpm; then
      pnpm install
    elif have_cmd npm; then
      npm install
    else
      fail "No package manager found (npm or pnpm required)"
      exit 1
    fi
  fi
  
  # Build the renderer
  if have_cmd pnpm; then
    pnpm run build
  else
    npm run build
  fi
  
  cd "$PROJECT_ROOT"
  
  if [ -d "$GUI_DIR/release/mac-arm64/TinyLlama GUI.app" ]; then
    success "GUI built successfully"
  elif [ -d "$GUI_DIR/dist" ]; then
    success "GUI renderer built successfully"
  else
    fail "GUI build failed"
    exit 1
  fi
}

install_python_macos() {
  warn "Python not found. Installing with Homebrew..."

  if ! have_cmd brew; then
    fail "Homebrew is not installed. Install Homebrew first: https://brew.sh"
    exit 1
  fi

  brew install python
}

install_python_windows() {
  warn "Python not found. Installing with winget..."

  if ! have_cmd winget; then
    fail "winget not found. Install App Installer from Microsoft Store and retry."
    exit 1
  fi

  winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
}

install_python_linux() {
  warn "Python not found. Detecting Linux package manager..."

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
    fail "Detected RPM-based system, but no supported installer (dnf/yum/zypper) was found."
    hint "Please install Python 3 manually, then rerun this script."
    exit 1
  fi

  fail "No supported Linux package manager found (apt, pacman, dnf/yum/zypper)."
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
      fail "Unsupported OS: $os_name"
      exit 1
      ;;
  esac

  if have_cmd python3; then
    PYTHON_BIN="python3"
  elif have_cmd python; then
    PYTHON_BIN="python"
  else
    fail "Python install appears to have failed (python not found in PATH)."
    exit 1
  fi
}

create_venv() {
  if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment in $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    success "Virtual environment created"
  else
    success "Using existing virtual environment: $VENV_DIR"
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
    fail "Could not find virtualenv activate script."
    exit 1
  fi
  success "Virtual environment activated"
}

install_dependencies() {
  if [ ! -f "$REQ_FILE" ]; then
    fail "Missing requirements.txt at $REQ_FILE"
    exit 1
  fi

  log "Installing Python dependencies..."
  python -m pip install --upgrade pip
  python -m pip install -r "$REQ_FILE"
  success "Dependencies installed"
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
    success "Bootstrap complete. Model already installed."
    exit 0
  fi

  if [ -n "${AUTO_DOWNLOAD_MODEL:-}" ]; then
    log "No installed model found. Auto-downloading: ${AUTO_DOWNLOAD_MODEL}"
    python download_model.py --model "${AUTO_DOWNLOAD_MODEL}"
    if has_installed_model; then
      success "Bootstrap complete. Model download finished."
      exit 0
    fi
    fail "Bootstrap finished, but no installed model was detected."
    exit 1
  fi

  warn "Bootstrap complete, but no model is installed."
  hint "Run: python download_model.py --model tinyllama|smollm2|smollm3"
  exit 0
}

run_gui_if_models_present() {
  if ! has_installed_model; then
    warn "No installed model found in $MODEL_DIR."
    log "Launching download script..."
    python download_model.py
    exit 0
  fi

  cd "$PROJECT_ROOT"
  print_rule "Launch"
  success "Environment ready"
  log "Launching GUI..."
  
  # Try to launch the app if it exists
  if [ -d "$GUI_DIR/release/mac-arm64/TinyLlama GUI.app" ]; then
    open "$GUI_DIR/release/mac-arm64/TinyLlama GUI.app"
  elif [ -d "$GUI_DIR/dist" ]; then
    cd "$GUI_DIR"
    if have_cmd pnpm; then
      pnpm run start
    else
      npm run start
    fi
  else
    fail "GUI not built. Run bootstrap first."
    exit 1
  fi
}

main() {
  cd "$PROJECT_ROOT"
  print_header
  show_environment
  print_rule "Python"
  ensure_python
  success "Using Python: $PYTHON_BIN"
  print_rule "Environment"
  create_venv
  activate_venv
  print_rule "Dependencies"
  install_dependencies
  print_rule "Node.js"
  ensure_node
  print_rule "Build"
  build_gui
  print_rule "Models"
  bootstrap_only_if_requested
  run_gui_if_models_present
}

main "$@"
