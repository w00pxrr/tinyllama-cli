#!/usr/bin/env python3
"""Image processing utilities for the AI CLI."""
from __future__ import annotations

import io
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
)


def is_valid_image_url(url: str) -> bool:
    """Check if the URL points to a valid image."""
    if not url.startswith(("http://", "https://")):
        return False
    
    parsed = urlparse(url)
    path = parsed.path.lower()
    valid_extensions = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff")
    return path.endswith(valid_extensions) or any(ext in path for ext in valid_extensions)


def download_image(url: str, timeout: int = 30) -> Image.Image | None:
    """Download an image from a URL and return a PIL Image."""
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data))
    except Exception:
        return None


def get_image_info(image: Image.Image) -> dict:
    """Get basic information about an image."""
    return {
        "format": image.format,
        "mode": image.mode,
        "size": image.size,
        "width": image.width,
        "height": image.height,
    }


def resize_image_for_display(image: Image.Image, max_width: int = 80, max_height: int = 24) -> Image.Image:
    """Resize image to fit terminal display constraints."""
    # Get terminal cell aspect ratio (characters are taller than wide)
    char_aspect = 2.0  # typical terminal character aspect ratio
    
    # Calculate max pixel dimensions based on terminal size and aspect ratio
    max_pixels_width = max_width
    max_pixels_height = int(max_height * char_aspect)
    
    # Current dimensions
    width, height = image.size
    
    # Calculate scaling factors
    scale_w = max_pixels_width / width if width > max_pixels_width else 1
    scale_h = max_pixels_height / height if height > max_pixels_height else 1
    scale = min(scale_w, scale_h, 1.0)
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image


def display_image_macos(image: Image.Image, title: str = "Image") -> bool:
    """Display image using macOS Quick Look."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp, format="PNG")
            tmp_path = tmp.name
        
        subprocess.run(["open", "-a", "Preview.app", tmp_path], check=True)
        
        # Clean up after a delay (Quick Look takes ownership)
        def cleanup():
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        
        # Schedule cleanup
        import threading
        timer = threading.Timer(5.0, cleanup)
        timer.daemon = True
        timer.start()
        
        return True
    except Exception:
        return False


def display_image_linux(image: Image.Image, title: str = "Image") -> bool:
    """Display image using Linux tools (xdg-open, eog, etc.)."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp, format="PNG")
            tmp_path = tmp.name
        
        subprocess.run(["xdg-open", tmp_path], check=True, env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", "")})
        
        def cleanup():
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        
        import threading
        timer = threading.Timer(5.0, cleanup)
        timer.daemon = True
        timer.start()
        
        return True
    except Exception:
        return False


def display_image_windows(image: Image.Image, title: str = "Image") -> bool:
    """Display image using Windows default viewer."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp, format="PNG")
            tmp_path = tmp.name
        
        subprocess.run(["start", "", tmp_path], shell=True, check=True)
        
        def cleanup():
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        
        import threading
        timer = threading.Timer(5.0, cleanup)
        timer.daemon = True
        timer.start()
        
        return True
    except Exception:
        return False


def display_image(image: Image.Image, title: str = "Image") -> bool:
    """Display image using the appropriate method for the current platform."""
    system = platform.system()
    
    if system == "Darwin":
        return display_image_macos(image, title)
    elif system == "Linux":
        return display_image_linux(image, title)
    elif system == "Windows":
        return display_image_windows(image, title)
    
    return False


def convert_to_ansi_image(image: Image.Image) -> str:
    """Convert PIL Image to ANSI escape sequence for terminal display."""
    # Resize for terminal
    resized = resize_image_for_display(image)
    
    # Convert to RGB if needed
    if resized.mode != "RGB":
        resized = resized.convert("RGB")
    
    # Get pixels
    pixels = list(resized.getdata())
    width, height = resized.size
    
    # Build ANSI string
    lines = []
    
    for y in range(height):
        row = []
        for x in range(width):
            idx = y * width + x
            r, g, b = pixels[idx]
            # Use 216-color ANSI codes
            r_idx = r // 51
            g_idx = g // 51
            b_idx = b // 51
            color_code = 16 + r_idx * 36 + g_idx * 6 + b_idx
            row.append(f"\033[38;5;{color_code}m▀")
        lines.append("".join(row))
    
    # Add reset at the end
    ansi = "\n".join(lines) + "\033[0m"
    return ansi
