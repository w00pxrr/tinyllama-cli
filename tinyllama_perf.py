"""
TinyLlama performance module.

This module provides optimized implementations of critical functions.
It automatically uses C++ extensions if available, otherwise falls back
to pure Python implementations.

Usage:
    from tinyllama_perf import MathExtractor, RamDetector, StringUtils, TokenCounter
    
    # Check for C++ extension availability
    if tinyllama_perf.has_cpp_extensions():
        # Use optimized C++ implementations
        ...
    else:
        # Use pure Python fallbacks
        ...
"""

import os
import sys
import re

# Try to import C++ extension, fall back to Python implementation
try:
    from tinyllama_cpp import (
        MathExtractor,
        RamDetector, 
        StringUtils,
        TokenCounter,
    )
    _HAS_CPP = True
except (ImportError, ModuleNotFoundError):
    _HAS_CPP = False
    # Use Python fallbacks
    from typing import Optional

# Python fallback implementations
class _PythonMathExtractor:
    """Pure Python math extraction (fallback)."""
    
    MATH_HINTS = ("calculate", "solve", "math", "equation", "evaluate", "multiply", "divide")
    
    @staticmethod
    def extract(input_text: str) -> Optional[str]:
        normalized = input_text.lower().replace("×", "*").replace("÷", "/")
        normalized = re.sub(r"(?i)\b(what is|what's|calculate|compute|evaluate|solve)\b", " ", normalized)
        
        match = re.search(r"[-+*/%()\d.\s]+(?:\*\*[-+*/%()\d.\s]+)?", normalized)
        if match:
            return match.group().replace(" ", "")
        return None
    
    @staticmethod
    def is_math(text: str) -> bool:
        normalized = text.lower()
        has_keyword = any(word in normalized for word in _PythonMathExtractor.MATH_HINTS)
        has_number = bool(re.search(r"\d", text))
        has_operator = bool(re.search(r"[\+\-\*/%=()]", text))
        return has_keyword or (has_number and has_operator)
    
    @staticmethod
    def normalize(expr: str) -> str:
        return expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    
    @staticmethod
    def evaluate(expr: str) -> float:
        # Note: Use with caution - eval can be dangerous
        # This is a simplified version
        import ast
        import operator
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
        }
        
        def visit(node):
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
                return operators[type(node.op)](left, right)
            raise ValueError("Unsupported expression")
        
        try:
            parsed = ast.parse(expr, mode="eval")
            return visit(parsed)
        except:
            return 0.0


class _PythonRamDetector:
    """Pure Python RAM detection (fallback)."""
    
    @staticmethod
    def get_available_bytes() -> int:
        import platform
        import subprocess
        import ctypes
        
        system = platform.system()
        
        if system == "Darwin":
            try:
                result = subprocess.run(["vm_stat"], capture_output=True, text=True)
                # Parse vm_stat output...
                pass
            except:
                pass
        elif system == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            return int(line.split()[1]) * 1024
            except:
                pass
        elif system == "Windows":
            try:
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                    ]
                mem = MEMORYSTATUSEX()
                mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
                    return int(mem.ullAvailPhys)
            except:
                pass
        
        return -1
    
    @staticmethod
    def get_total_bytes() -> int:
        return -1
    
    @staticmethod
    def has_gpu() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False


class _PythonStringUtils:
    """Pure Python string utilities (fallback)."""
    
    @staticmethod
    def clean_output(text: str) -> str:
        return re.sub(r"</s>.*$", "", text, flags=re.DOTALL).strip()
    
    @staticmethod
    def to_lower(text: str) -> str:
        return text.lower()
    
    @staticmethod
    def trim(text: str) -> str:
        return text.strip()
    
    @staticmethod
    def contains(haystack: str, needle: str) -> bool:
        return needle in haystack
    
    @staticmethod
    def starts_with(text: str, prefix: str) -> bool:
        return text.startswith(prefix)


class _PythonTokenCounter:
    """Pure Python token counting (fallback)."""
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        return len(text) // 4 + 1
    
    @staticmethod
    def calculate_max_tokens(prompt_tokens: int, context_limit: int, available_ram: int) -> int:
        if available_ram < 0:
            return max(32, context_limit - prompt_tokens - 16)
        
        available_gib = available_ram / (1024 ** 3)
        
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
        
        remaining = max(32, context_limit - prompt_tokens - 16)
        
        if available_gib >= 12:
            return max(32, min(memory_cap + 64, min(memory_cap, remaining)))
        
        return max(32, min(memory_cap, remaining))


# Export the appropriate implementations
if _HAS_CPP:
    # Use C++ implementations
    MathExtractor = MathExtractor
    RamDetector = RamDetector
    StringUtils = StringUtils
    TokenCounter = TokenCounter
else:
    # Use Python fallbacks
    MathExtractor = _PythonMathExtractor
    RamDetector = _PythonRamDetector
    StringUtils = _PythonStringUtils
    TokenCounter = _PythonTokenCounter


def has_cpp_extensions() -> bool:
    """Check if C++ extensions are available."""
    return _HAS_CPP


def get_version() -> str:
    """Get the version of the performance module."""
    if _HAS_CPP:
        try:
            import tinyllama_cpp
            return tinyllama_cpp.VERSION
        except:
            pass
    return "1.0.0-python"


# Convenience functions with automatic fallback
def extract_math(text: str) -> str:
    """Extract mathematical expression from text."""
    return MathExtractor.extract(text)


def is_math_input(text: str) -> bool:
    """Check if text contains mathematical content."""
    return MathExtractor.is_math(text)


def clean_output(text: str) -> str:
    """Clean model output (remove special tokens)."""
    return StringUtils.clean_output(text)


def get_available_ram() -> int:
    """Get available RAM in bytes."""
    return RamDetector.get_available_bytes()


def has_gpu() -> bool:
    """Check if GPU is available."""
    return RamDetector.has_gpu()


def estimate_tokens(text: str) -> int:
    """Estimate token count."""
    return TokenCounter.estimate_tokens(text)


def calculate_max_tokens(prompt_tokens: int, context_limit: int = 2048, available_ram: int = -1) -> int:
    """Calculate dynamic max tokens."""
    return TokenCounter.calculate_max_tokens(prompt_tokens, context_limit, available_ram)
