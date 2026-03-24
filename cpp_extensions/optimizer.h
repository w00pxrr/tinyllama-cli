#ifndef TINYLLAMA_OPTIMIZER_H
#define TINYLLAMA_OPTIMIZER_H

#include <string>
#include <vector>
#include <regex>
#include <cstdint>

namespace tinyllama {

// Optimized math expression extraction using C++ regex
class MathExtractor {
public:
    // Extract mathematical expression from user input
    static std::string extract(const std::string& input);
    
    // Check if input looks like math
    static bool is_math(const std::string& input);
    
    // Normalize math expression (handle Unicode symbols)
    static std::string normalize(const std::string& expr);
    
    // Safe math evaluation (supports basic operations)
    static double evaluate(const std::string& expr);
    
private:
    static double eval_expression(const std::string& expr, size_t& pos);
    static double eval_term(const std::string& expr, size_t& pos);
    static double eval_factor(const std::string& expr, size_t& pos);
};

// Fast RAM detection for all platforms
class RamDetector {
public:
    // Get available RAM in bytes (-1 if unknown)
    static int64_t get_available_bytes();
    
    // Get total RAM in bytes (-1 if unknown)  
    static int64_t get_total_bytes();
    
    // Check if running on GPU
    static bool has_gpu();
};

// Optimized string utilities
class StringUtils {
public:
    // Fast string cleaning (replaces </s> tags)
    static std::string clean_output(const std::string& input);
    
    // Fast lowercase conversion
    static std::string to_lower(const std::string& input);
    
    // Fast whitespace trimming
    static std::string trim(const std::string& input);
    
    // Check if string contains substring (faster than std::string::find)
    static bool contains(const std::string& haystack, const std::string& needle);
    
    // Check if string starts with prefix
    static bool starts_with(const std::string& str, const std::string& prefix);
};

// Performance-optimized token counting
class TokenCounter {
public:
    // Estimate token count (faster than actual tokenization)
    static size_t estimate_tokens(const std::string& text);
    
    // Calculate dynamic max tokens based on context limit and available RAM
    static int calculate_max_tokens(
        size_t prompt_tokens, 
        size_t context_limit, 
        int64_t available_ram
    );
};

} // namespace tinyllama

#endif // TINYLLAMA_OPTIMIZER_H
