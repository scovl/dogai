#pragma once

#include "yolov8_postprocessor.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

class FOVProcessor {
private:
    int fov_width = 400;
    int fov_height = 400;
    cv::Size fov_size;

public:
    FOVProcessor(int width = 400, int height = 400);
    ~FOVProcessor() = default;
    
    void set_fov_size(int width, int height);
    cv::Size get_fov_size() const;
    std::vector<Detection> process_fov_detections(const std::vector<Detection>& detections);
    cv::Point2f calculate_fov_center(const cv::Rect& box) const;
    float calculate_fov_distance(const cv::Point2f& center) const;
    float calculate_fov_angle(const cv::Point2f& center) const;
    void calculate_fov_metrics(Detection& detection);
}; 