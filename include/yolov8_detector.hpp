#pragma once

#include "config_manager.hpp"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <numeric>

class YOLOv8 {
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
    
    // FOV configuration
    int fov_width = 400;
    int fov_height = 400;
    cv::Size fov_size;
    
    std::vector<std::string> class_names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    
    std::vector<cv::Scalar> colors;
    
    struct Detection {
        cv::Rect box;
        float score;
        int class_id;
        
        // FOV relative coordinates
        cv::Point2f fov_center;  // Center point relative to FOV (0-1)
        float fov_distance;      // Distance from FOV center (0-1)
        float fov_angle;         // Angle from FOV center in radians
    };

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
    cv::Point2f calculate_fov_center(const cv::Rect& box) const;
    float calculate_fov_distance(const cv::Point2f& center) const;
    float calculate_fov_angle(const cv::Point2f& center) const;

private:
    void load_config_from_file();
    void initialize_model(const std::string& model_path);
    std::vector<float> prepare_input(const cv::Mat& image);
    std::vector<Ort::Value> inference(const std::vector<float>& input_tensor);
    std::vector<Detection> process_output(const std::vector<Ort::Value>& outputs, const cv::Size& original_size);
    std::vector<Detection> non_max_suppression(const std::vector<Detection>& detections);
    float calculate_iou(const cv::Rect& box1, const cv::Rect& box2);
    void calculate_fov_metrics(Detection& detection);
}; 