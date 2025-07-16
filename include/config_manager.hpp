#pragma once

#include "logger.hpp"
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

class ConfigManager {
private:
    std::map<std::string, std::map<std::string, std::string>> config;
    std::string config_file;
    
public:
    ConfigManager(const std::string& filename = "models/blood.cfg") : config_file(filename) {
        load_config();
    }
    
    bool load_config() {
        std::ifstream file(config_file);
        if (!file.is_open()) {
            logger.error("[CONFIG][ERROR] Could not open configuration file: " + config_file);
            return false;
        }
        
        std::string current_section = "";
        std::string line;
        
        while (std::getline(file, line)) {
            // Remove comentários
            size_t comment_pos = line.find('#');
            if (comment_pos != std::string::npos) {
                line = line.substr(0, comment_pos);
            }
            
            // Remove blank spaces
            line.erase(0, line.find_first_not_of(" \t"));
            line.erase(line.find_last_not_of(" \t") + 1);
            
            if (line.empty()) continue;
            
            // Check if it's a section
            if (line[0] == '[' && line[line.length()-1] == ']') {
                current_section = line.substr(1, line.length()-2);
                config[current_section] = std::map<std::string, std::string>();
            }
            // Check if it's a key=value
            else if (!current_section.empty() && line.find('=') != std::string::npos) {
                size_t equal_pos = line.find('=');
                std::string key = line.substr(0, equal_pos);
                std::string value = line.substr(equal_pos + 1);
                
                // Remove blank spaces
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);
                
                config[current_section][key] = value;
            }
        }
        
        return true;
    }
    
    std::string get_string(const std::string& section, const std::string& key, const std::string& default_value = "") {
        if (config.find(section) != config.end() && config[section].find(key) != config[section].end()) {
            return config[section][key];
        }
        return default_value;
    }
    
    int get_int(const std::string& section, const std::string& key, int default_value = 0) {
        std::string value = get_string(section, key, "");
        if (!value.empty()) {
            try {
                return std::stoi(value);
            } catch (...) {
                logger.error("[CONFIG][ERROR] Invalid value for " + section + "." + key + ": " + value);
            }
        }
        return default_value;
    }
    
    float get_float(const std::string& section, const std::string& key, float default_value = 0.0f) {
        std::string value = get_string(section, key, "");
        if (!value.empty()) {
            try {
                return std::stof(value);
            } catch (...) {
                logger.error("[CONFIG][ERROR] Invalid value for " + section + "." + key + ": " + value);
            }
        }
        return default_value;
    }
    
    std::vector<int> get_int_array(const std::string& section, const std::string& key, const std::vector<int>& default_value = {}) {
        std::string value = get_string(section, key, "");
        if (!value.empty()) {
            std::vector<int> result;
            std::stringstream ss(value);
            std::string item;
            while (std::getline(ss, item, ',')) {
                try {
                    result.push_back(std::stoi(item));
                } catch (...) {
                    logger.error("[CONFIG][ERROR] Invalid value in array " + section + "." + key + ": " + item);
                }
            }
            return result;
        }
        return default_value;
    }
    
    void log_config() {
        // Removed logging - only errors are logged now
    }
}; 