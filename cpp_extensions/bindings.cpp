#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include "optimizer.h"

namespace py = pybind11;

PYBIND11_MODULE(tinyllama_cpp, m) {
    m.doc() = "TinyLlama C++ extensions for performance optimization";
    
    // MathExtractor bindings
    py::class_<tinyllama::MathExtractor>(m, "MathExtractor")
        .def_static("extract", &tinyllama::MathExtractor::extract, 
            "Extract math expression from user input")
        .def_static("is_math", &tinyllama::MathExtractor::is_math,
            "Check if input looks like math")
        .def_static("normalize", &tinyllama::MathExtractor::normalize,
            "Normalize math expression")
        .def_static("evaluate", &tinyllama::MathExtractor::evaluate,
            "Safely evaluate math expression");
    
    // RamDetector bindings
    py::class_<tinyllama::RamDetector>(m, "RamDetector")
        .def_static("get_available_bytes", &tinyllama::RamDetector::get_available_bytes,
            "Get available RAM in bytes")
        .def_static("get_total_bytes", &tinyllama::RamDetector::get_total_bytes,
            "Get total RAM in bytes")
        .def_static("has_gpu", &tinyllama::RamDetector::has_gpu,
            "Check if GPU is available");
    
    // StringUtils bindings
    py::class_<tinyllama::StringUtils>(m, "StringUtils")
        .def_static("clean_output", &tinyllama::StringUtils::clean_output,
            "Clean model output (remove </s> tags)")
        .def_static("to_lower", &tinyllama::StringUtils::to_lower,
            "Convert string to lowercase")
        .def_static("trim", &tinyllama::StringUtils::trim,
            "Trim whitespace from string")
        .def_static("contains", &tinyllama::StringUtils::contains,
            "Check if string contains substring")
        .def_static("starts_with", &tinyllama::StringUtils::starts_with,
            "Check if string starts with prefix");
    
    // TokenCounter bindings
    py::class_<tinyllama::TokenCounter>(m, "TokenCounter")
        .def_static("estimate_tokens", &tinyllama::TokenCounter::estimate_tokens,
            "Estimate token count (faster than tokenization)")
        .def_static("calculate_max_tokens", &tinyllama::TokenCounter::calculate_max_tokens,
            "Calculate dynamic max tokens based on context and RAM");
    
    // Expose constants
    m.attr("VERSION") = "1.0.0";
}
