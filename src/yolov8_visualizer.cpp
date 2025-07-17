#include "yolov8_visualizer.hpp"
#include <random>

YOLOv8Visualizer::YOLOv8Visualizer(const std::string& config_file) 
    : config(config_file) {
    initialize_colors();
}

void YOLOv8Visualizer::initialize_colors() {
    // Initialize colors
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 255.0f);
    
    for (size_t i = 0; i < class_names.size(); ++i) {
        colors.push_back(cv::Scalar(dis(gen), dis(gen), dis(gen)));
    }
}

cv::Mat YOLOv8Visualizer::draw_detections(const cv::Mat& image, const std::vector<Detection>& detections) {
    cv::Mat result = image.clone();
    
    // Load display configuration
    std::vector<int> box_color = config.get_int_array("Display", "box_color", {0, 0, 255});
    std::vector<int> text_color = config.get_int_array("Display", "text_color", {255, 255, 255});
    int box_thickness = config.get_int("Display", "box_thickness", 2);
    float text_scale = config.get_float("Display", "text_scale", 0.5f);
    bool show_confidence = config.get_string("Display", "show_confidence", "true") == "true";
    bool show_class_name = config.get_string("Display", "show_class_name", "true") == "true";
    
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        
        // Use configured color or default color
        cv::Scalar color;
        if (box_color.size() >= 3) {
            color = cv::Scalar(box_color[0], box_color[1], box_color[2]);
        } else {
            color = colors[det.class_id % colors.size()];
        }
        
        // Draw bounding box
        cv::rectangle(result, det.box, color, box_thickness);
        
        // Prepare label
        std::string label;
        if (show_class_name) {
            if (det.class_id < class_names.size()) {
                label = class_names[det.class_id];
            } else {
                label = "Class " + std::to_string(det.class_id);
            }
        }
        if (show_confidence) {
            if (!label.empty()) label += " ";
            label += std::to_string(static_cast<int>(det.score * 100)) + "%";
        }
        
        if (!label.empty()) {
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, text_scale, 1, &baseline);
            
            // Background of text
            cv::rectangle(result, 
                         cv::Point(det.box.x, det.box.y - text_size.height - 10),
                         cv::Point(det.box.x + text_size.width, det.box.y),
                         color, -1);
            
            // Texto
            cv::Scalar text_color_scalar;
            if (text_color.size() >= 3) {
                text_color_scalar = cv::Scalar(text_color[0], text_color[1], text_color[2]);
            } else {
                text_color_scalar = cv::Scalar(255, 255, 255);
            }
            
            cv::putText(result, label, 
                       cv::Point(det.box.x, det.box.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, text_scale, text_color_scalar, 1);
        }
    }
    
    return result;
}

cv::Mat YOLOv8Visualizer::draw_fov_detections(const cv::Mat& fov_image, const std::vector<Detection>& detections, int fov_width, int fov_height) {
    cv::Mat result = fov_image.clone();
    
    // Draw FOV center crosshair
    cv::Point fov_center(fov_width / 2, fov_height / 2);
    cv::line(result, cv::Point(fov_center.x - 10, fov_center.y), cv::Point(fov_center.x + 10, fov_center.y), cv::Scalar(0, 255, 0), 2);
    cv::line(result, cv::Point(fov_center.x, fov_center.y - 10), cv::Point(fov_center.x, fov_center.y + 10), cv::Scalar(0, 255, 0), 2);
    
    // Draw FOV border
    cv::rectangle(result, cv::Rect(0, 0, fov_width, fov_height), cv::Scalar(0, 255, 0), 2);
    
    // Draw detections with FOV information
    for (const auto& det : detections) {
        // Draw bounding box
        cv::rectangle(result, det.box, cv::Scalar(0, 0, 255), 2);
        
        // Draw line from FOV center to detection center
        cv::Point det_center(det.box.x + det.box.width / 2, det.box.y + det.box.height / 2);
        cv::line(result, fov_center, det_center, cv::Scalar(255, 0, 0), 1);
        
        // Draw FOV metrics
        std::string info = "D:" + std::to_string(static_cast<int>(det.fov_distance * 100)) + 
                          " A:" + std::to_string(static_cast<int>(det.fov_angle * 180 / 3.14159f));
        cv::putText(result, info, cv::Point(det.box.x, det.box.y - 5), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
    
    return result;
} 