#pragma once

#include "logger.hpp"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>

struct Detection {
    cv::Rect box;
    float score;
    int class_id;
    
    // FOV relative coordinates
    cv::Point2f fov_center;  // Center point relative to FOV (0-1)
    float fov_distance;      // Distance from FOV center (0-1)
    float fov_angle;         // Angle from FOV center in radians
};

class YOLOv8Postprocessor {
private:
    float conf_threshold = 0.2f;
    float iou_threshold = 0.2f;
    int input_width = 640;
    int input_height = 640;
    Logger logger;

public:
    YOLOv8Postprocessor(float conf_thres = 0.2f, float iou_thres = 0.2f, int width = 640, int height = 640);
    ~YOLOv8Postprocessor() = default;
    
    std::vector<Detection> process_output(const std::vector<Ort::Value>& outputs, const cv::Size& original_size);
    std::vector<Detection> non_max_suppression(const std::vector<Detection>& detections);
    void set_thresholds(float conf_thres, float iou_thres);
    void set_input_size(int width, int height);

private:
    float calculate_iou(const cv::Rect& box1, const cv::Rect& box2);
}; 