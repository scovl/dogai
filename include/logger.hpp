#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <string>

// Enum para níveis de log
enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3
};

// Sistema de logging com níveis
class Logger {
private:
    std::ofstream log_file;
    LogLevel current_level = LogLevel::ERROR; // Default: only errors
    
    std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }
    
    std::string level_to_string(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARNING";
            case LogLevel::ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }
    
    void write_log(LogLevel level, const std::string& message) {
        if (level >= current_level) {
            std::string timestamp = get_timestamp();
            std::string level_str = level_to_string(level);
            std::string log_message = "[" + timestamp + "] [" + level_str + "] " + message;
            
            // Log to file
            if (log_file.is_open()) {
                log_file << log_message << std::endl;
                log_file.flush();
            }
            
            // Log to console
            if (level == LogLevel::ERROR) {
                std::cerr << log_message << std::endl;
            } else {
                std::cout << log_message << std::endl;
            }
        }
    }

public:
    Logger() {
        log_file.open("dogai.log", std::ios::out | std::ios::app);
        if (log_file.is_open()) {
            info("=== DOGAI LOG STARTED ===");
        }
    }
    
    ~Logger() {
        if (log_file.is_open()) {
            info("=== DOGAI LOG ENDED ===");
            log_file.close();
        }
    }
    
    // log level to set
    void set_log_level(LogLevel level) {
        current_level = level;
    }
    
    // log methods by level
    void debug(const std::string& message) {
        write_log(LogLevel::DEBUG, message);
    }
    
    void info(const std::string& message) {
        write_log(LogLevel::INFO, message);
    }
    
    void warning(const std::string& message) {
        write_log(LogLevel::WARNING, message);
    }
    
    void error(const std::string& message) {
        write_log(LogLevel::ERROR, message);
    }
};

// Instância global do logger
extern Logger logger; 