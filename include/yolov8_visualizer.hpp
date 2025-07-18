#pragma once

#include "config_manager.hpp"
#include "yolov8_postprocessor.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <random>

class YOLOv8Visualizer {
private:
    ConfigManager config;
    std::vector<cv::Scalar> colors;
    std::vector<std::string> class_names = {
        "player",      // Jogadores aliados
        "enemy",       // Inimigos
        "head",        // Cabeça (headshot)
        "body",        // Corpo
        "weapon",      // Armas
        "blood"        // Sangue/efeitos visuais
    };

public:
    YOLOv8Visualizer(const std::string& config_file = "blood.cfg");
    ~YOLOv8Visualizer() = default;
    
    cv::Mat draw_detections(const cv::Mat& image, const std::vector<Detection>& detections);
    cv::Mat draw_fov_detections(const cv::Mat& fov_image, const std::vector<Detection>& detections, int fov_width, int fov_height);

private:
    void initialize_colors();
}; 