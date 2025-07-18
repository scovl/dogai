#pragma once

#include "yolov8_model.hpp"
#include "yolov8_preprocessor.hpp"
#include "yolov8_postprocessor.hpp"
#include "yolov8_visualizer.hpp"
#include "fov_processor.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

class YOLOv8 {
private:
    std::unique_ptr<YOLOv8Model> model;
    std::unique_ptr<YOLOv8Preprocessor> preprocessor;
    std::unique_ptr<YOLOv8Postprocessor> postprocessor;
    std::unique_ptr<YOLOv8Visualizer> visualizer;
    std::unique_ptr<FOVProcessor> fov_processor;

public:
    YOLOv8(const std::string& model_path, float conf_thres = 0.2f, float iou_thres = 0.2f);
    ~YOLOv8() = default;
    
    std::vector<Detection> detect_objects(const cv::Mat& image);
    cv::Mat draw_detections(const cv::Mat& image, const std::vector<Detection>& detections);
    
    // FOV specific methods
    void set_fov_size(int width, int height);
    cv::Size get_fov_size() const;
    std::vector<Detection> detect_objects_fov(const cv::Mat& fov_image);
    cv::Mat draw_fov_detections(const cv::Mat& fov_image, const std::vector<Detection>& detections);
}; 