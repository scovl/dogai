#pragma once

#include "logger.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

class YOLOv8Preprocessor {
private:
    int input_width = 640;
    int input_height = 640;
    Logger logger;

public:
    YOLOv8Preprocessor(int width = 640, int height = 640);
    ~YOLOv8Preprocessor() = default;
    
    std::vector<float> prepare_input(const cv::Mat& image);
    void set_input_size(int width, int height);
    int get_input_width() const { return input_width; }
    int get_input_height() const { return input_height; }
}; 