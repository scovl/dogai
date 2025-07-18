#include "yolov8_detector.hpp"

YOLOv8::YOLOv8(const std::string& model_path, float conf_thres, float iou_thres) {
    // Initialize all components
    model = std::make_unique<YOLOv8Model>(model_path, conf_thres, iou_thres);
    preprocessor = std::make_unique<YOLOv8Preprocessor>(model->get_input_width(), model->get_input_height());
    postprocessor = std::make_unique<YOLOv8Postprocessor>(model->get_conf_threshold(), model->get_iou_threshold(), 
                                                         model->get_input_width(), model->get_input_height());
    visualizer = std::make_unique<YOLOv8Visualizer>("blood.cfg");
    fov_processor = std::make_unique<FOVProcessor>(400, 400);
}

std::vector<Detection> YOLOv8::detect_objects(const cv::Mat& image) {
    // 1. Preprocess image
    auto input_tensor = preprocessor->prepare_input(image);
    
    // 2. Run inference
    auto outputs = model->run_inference(input_tensor);
    
    // 3. Postprocess results
    auto detections = postprocessor->process_output(outputs, image.size());
    
    return detections;
}

cv::Mat YOLOv8::draw_detections(const cv::Mat& image, const std::vector<Detection>& detections) {
    return visualizer->draw_detections(image, detections);
}

void YOLOv8::set_fov_size(int width, int height) {
    fov_processor->set_fov_size(width, height);
}

cv::Size YOLOv8::get_fov_size() const {
    return fov_processor->get_fov_size();
}

std::vector<Detection> YOLOv8::detect_objects_fov(const cv::Mat& fov_image) {
    if (fov_image.empty()) {
        return std::vector<Detection>();
    }
    
    // Detect objects in FOV
    auto detections = detect_objects(fov_image);
    
    // Process FOV-specific calculations
    return fov_processor->process_fov_detections(detections);
}

cv::Mat YOLOv8::draw_fov_detections(const cv::Mat& fov_image, const std::vector<Detection>& detections) {
    auto fov_size = fov_processor->get_fov_size();
    return visualizer->draw_fov_detections(fov_image, detections, fov_size.width, fov_size.height);
} 