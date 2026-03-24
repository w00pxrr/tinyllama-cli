#include "optimizer.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <thread>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_init.h>
#include <mach/task_info.h>
#endif

namespace tinyllama {

// Math expression extraction - optimized C++ implementation
std::string MathExtractor::extract(const std::string& input) {
    // Normalize common math keywords
    std::string normalized = input;
    std::regex keywords(R"(what is|what's|calculate|compute|evaluate|solve|just answer with|answer only)", std::regex::icase);
    normalized = std::regex_replace(normalized, keywords, " ");
    
    // Extract math expression using regex
    std::regex expr_r(R"([-+*/%()\d.\s]+(?:\*\*[-+*/%()\d.\s]+)?)");
    std::smatch match;
    if (std::regex_search(normalized, match, expr_r)) {
        std::string result = match.str();
        // Remove spaces
        result.erase(std::remove(result.begin(), result.end(), ' '), result.end());
        return result;
    }
    
    return "";
}

bool MathExtractor::is_math(const std::string& input) {
    std::string lower = input;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    // Check for math keywords
    std::vector<std::string> math_words = {"calculate", "solve", "math", "equation", "evaluate", "multiply", "divide"};
    bool has_keyword = false;
    for (const auto& word : math_words) {
        if (lower.find(word) != std::string::npos) {
            has_keyword = true;
            break;
        }
    }
    
    // Check for numbers and operators
    bool has_number = input.find_first_of("0123456789") != std::string::npos;
    bool has_operator = input.find_first_of("+-*/%=") != std::string::npos;
    
    return has_keyword || (has_number && has_operator);
}

std::string MathExtractor::normalize(const std::string& expr) {
    std::string result = expr;
    // Handle Unicode symbols using escape sequences
    for (char& c : result) {
        if (c == static_cast<char>(0xD7)) c = '*';  // ×
        else if (c == static_cast<char>(0xF7)) c = '/';  // ÷
    }
    return result;
}

double MathExtractor::evaluate(const std::string& expr) {
    size_t pos = 0;
    try {
        return eval_expression(expr, pos);
    } catch (...) {
        return 0.0;
    }
}

double MathExtractor::eval_expression(const std::string& expr, size_t& pos) {
    double result = eval_term(expr, pos);
    while (pos < expr.length()) {
        char op = expr[pos];
        if (op != '+' && op != '-') break;
        pos++;
        double term = eval_term(expr, pos);
        if (op == '+') result += term;
        else result -= term;
    }
    return result;
}

double MathExtractor::eval_term(const std::string& expr, size_t& pos) {
    double result = eval_factor(expr, pos);
    while (pos < expr.length()) {
        char op = expr[pos];
        if (op != '*' && op != '/' && op != '%') break;
        pos++;
        double factor = eval_factor(expr, pos);
        if (op == '*') result *= factor;
        else if (op == '/') result /= factor;
        else result = fmod(result, factor);
    }
    return result;
}

double MathExtractor::eval_factor(const std::string& expr, size_t& pos) {
    if (pos >= expr.length()) return 0;
    
    // Handle unary operators
    if (expr[pos] == '-') {
        pos++;
        return -eval_factor(expr, pos);
    }
    if (expr[pos] == '+') {
        pos++;
        return eval_factor(expr, pos);
    }
    
    // Handle parentheses
    if (expr[pos] == '(') {
        pos++;
        double result = eval_expression(expr, pos);
        if (pos < expr.length() && expr[pos] == ')') pos++;
        return result;
    }
    
    // Handle power operator
    if (expr[pos] == '^' || (pos + 1 < expr.length() && expr[pos] == '*' && expr[pos + 1] == '*')) {
        if (expr[pos] == '^') {
            pos++;
        } else {
            pos += 2;
        }
        double base = eval_factor(expr, pos);
        double exp = eval_factor(expr, pos);
        return pow(base, exp);
    }
    
    // Parse number
    size_t start = pos;
    while (pos < expr.length() && (isdigit(expr[pos]) || expr[pos] == '.')) {
        pos++;
    }
    if (start == pos) return 0;
    
    return std::stod(expr.substr(start, pos - start));
}

// RAM detection - cross-platform implementation
int64_t RamDetector::get_available_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX mem_info;
    mem_info.dwLength = sizeof(mem_info);
    if (GlobalMemoryStatusEx(&mem_info)) {
        return mem_info.ullAvailPhys;
    }
#elif defined(__APPLE__)
    vm_size_t page_size;
    vm_statistics64_data_t vm_stats;
    mach_port_t mach_port = mach_host_self();
    host_page_size(mach_port, &page_size);
    natural_t num_info = sizeof(vm_stats) / sizeof(integer_t);
    if (host_statistics64(mach_port, HOST_VM_INFO64, (host_info64_t)&vm_stats, &num_info) == KERN_SUCCESS) {
        int64_t free_bytes = (int64_t)vm_stats.free_count * page_size;
        int64_t inactive_bytes = (int64_t)vm_stats.inactive_count * page_size;
        return free_bytes + inactive_bytes;
    }
#else
    // Linux: read from /proc/meminfo
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.find("MemAvailable:") == 0) {
            std::istringstream iss(line);
            std::string key;
            int64_t value;
            iss >> key >> value;
            return value * 1024; // Convert from KB to bytes
        }
    }
#endif
    return -1;
}

int64_t RamDetector::get_total_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX mem_info;
    mem_info.dwLength = sizeof(mem_info);
    if (GlobalMemoryStatusEx(&mem_info)) {
        return mem_info.ullTotalPhys;
    }
#elif defined(__APPLE__)
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    int64_t mem_size = 0;
    size_t length = sizeof(mem_size);
    sysctl(mib, 2, &mem_size, &length, NULL, 0);
    return mem_size;
#else
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal:") == 0) {
            std::istringstream iss(line);
            std::string key;
            int64_t value;
            iss >> key >> value;
            return value * 1024;
        }
    }
#endif
    return -1;
}

bool RamDetector::has_gpu() {
    // Check via PyTorch at runtime - this is a placeholder
    // GPU detection should be done in Python
    return false;
}

// String utilities - optimized implementations
std::string StringUtils::clean_output(const std::string& input) {
    size_t pos = input.find("</s>");
    if (pos != std::string::npos) {
        return input.substr(0, pos);
    }
    return input;
}

std::string StringUtils::to_lower(const std::string& input) {
    std::string result = input;
    std::transform(result.begin(), result.end(), result.begin(), 
        [](unsigned char c) { return std::tolower(c); });
    return result;
}

std::string StringUtils::trim(const std::string& input) {
    size_t start = input.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = input.find_last_not_of(" \t\n\r");
    return input.substr(start, end - start + 1);
}

bool StringUtils::contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

bool StringUtils::starts_with(const std::string& str, const std::string& prefix) {
    return str.compare(0, prefix.length(), prefix) == 0;
}

// Token estimation - faster than actual tokenization
size_t TokenCounter::estimate_tokens(const std::string& text) {
    // Rough estimate: ~4 characters per token on average
    // This is much faster than actual tokenization
    size_t char_count = text.length();
    return (char_count / 4) + 1;
}

int TokenCounter::calculate_max_tokens(size_t prompt_tokens, size_t context_limit, int64_t available_ram) {
    if (available_ram < 0) {
        return std::max(32, static_cast<int>(context_limit - prompt_tokens - 16));
    }
    
    double available_gib = available_ram / (1024.0 * 1024.0 * 1024.0);
    int memory_cap;
    
    if (available_gib < 8) memory_cap = 96;
    else if (available_gib < 12) memory_cap = 160;
    else if (available_gib < 16) memory_cap = 224;
    else if (available_gib < 24) memory_cap = 320;
    else if (available_gib < 32) memory_cap = 448;
    else memory_cap = 640;
    
    int remaining_context = std::max(32, static_cast<int>(context_limit - prompt_tokens - 16));
    
    if (available_gib >= 12) {
        return std::max(32, std::min(memory_cap + 64, std::min(memory_cap, remaining_context)));
    }
    
    return std::max(32, std::min(memory_cap, remaining_context));
}

} // namespace tinyllama
