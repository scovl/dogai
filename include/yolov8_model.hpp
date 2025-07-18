#pragma once

#include "config_manager.hpp"
#include "logger.hpp"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

class YOLOv8Model {
private:
    Ort::Session session{nullptr};
    Ort::Env env;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    int input_height = 640;
    int input_width = 640;
    float conf_threshold = 0.2f;
    float iou_threshold = 0.2f;
    ConfigManager config;
    Logger logger;

public:
    YOLOv8Model(const std::string& model_path, float conf_thres = 0.2f, float iou_thres = 0.2f);
    ~YOLOv8Model() = default;
    
    std::vector<Ort::Value> run_inference(const std::vector<float>& input_tensor);
    int get_input_width() const { return input_width; }
    int get_input_height() const { return input_height; }
    float get_conf_threshold() const { return conf_threshold; }
    float get_iou_threshold() const { return iou_threshold; }

private:
    void load_config_from_file();
    void initialize_model(const std::string& model_path);
}; 