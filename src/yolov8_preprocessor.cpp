#include "yolov8_preprocessor.hpp"

YOLOv8Preprocessor::YOLOv8Preprocessor(int width, int height) 
    : input_width(width), input_height(height) {
}

void YOLOv8Preprocessor::set_input_size(int width, int height) {
    input_width = width;
    input_height = height;
}

std::vector<float> YOLOv8Preprocessor::prepare_input(const cv::Mat& image) {
    if (image.empty()) {
        logger.error("[YOLOv8Preprocessor][ERROR] Input image is empty!");
        return std::vector<float>();
    }
    
    // 1. Convert BGR to RGB
    cv::Mat rgb_img;
    cv::cvtColor(image, rgb_img, cv::COLOR_BGR2RGB);
    
    // 2. Resize
    cv::Mat resized;
    cv::resize(rgb_img, resized, cv::Size(input_width, input_height));
    
    // 3. Normalize to [0,1]
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32F, 1.0/255.0);
    
    // 4. Transpose to NCHW
    std::vector<cv::Mat> chw(3);
    cv::split(float_img, chw);
    std::vector<float> input_tensor(input_width * input_height * 3);
    size_t channel_size = input_width * input_height;
    
    for (int c = 0; c < 3; ++c) {
        memcpy(input_tensor.data() + c * channel_size, chw[c].data, channel_size * sizeof(float));
    }
    
    return input_tensor;
} 