#include "fov_processor.hpp"
#include <cmath>

FOVProcessor::FOVProcessor(int width, int height) 
    : fov_width(width), fov_height(height) {
    fov_size = cv::Size(width, height);
}

void FOVProcessor::set_fov_size(int width, int height) {
    fov_width = width;
    fov_height = height;
    fov_size = cv::Size(width, height);
}

cv::Size FOVProcessor::get_fov_size() const {
    return fov_size;
}

std::vector<Detection> FOVProcessor::process_fov_detections(const std::vector<Detection>& detections) {
    std::vector<Detection> processed_detections = detections;
    
    // Calculate FOV metrics for each detection
    for (auto& detection : processed_detections) {
        calculate_fov_metrics(detection);
    }
    
    return processed_detections;
}

cv::Point2f FOVProcessor::calculate_fov_center(const cv::Rect& box) const {
    cv::Point2f center;
    center.x = (box.x + box.width / 2.0f) / fov_width;
    center.y = (box.y + box.height / 2.0f) / fov_height;
    return center;
}

float FOVProcessor::calculate_fov_distance(const cv::Point2f& center) const {
    // Calculate distance from FOV center (0,0) to detection center
    float dx = center.x - 0.5f;  // Center of FOV is (0.5, 0.5)
    float dy = center.y - 0.5f;
    return std::sqrt(dx * dx + dy * dy);
}

float FOVProcessor::calculate_fov_angle(const cv::Point2f& center) const {
    // Calculate angle from FOV center to detection center
    float dx = center.x - 0.5f;
    float dy = center.y - 0.5f;
    return std::atan2(dy, dx);
}

void FOVProcessor::calculate_fov_metrics(Detection& detection) {
    detection.fov_center = calculate_fov_center(detection.box);
    detection.fov_distance = calculate_fov_distance(detection.fov_center);
    detection.fov_angle = calculate_fov_angle(detection.fov_center);
} 