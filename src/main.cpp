#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include "logger.hpp"
#include "yolov8_detector.hpp"
#include "windows_graphics_capture.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

// Global logger instance
Logger logger;

int main() {
    // Initialize Windows Graphics Capture
    WindowsGraphicsCapture capture;
    if (!capture.is_initialized()) {
        logger.error("[MAIN][ERROR] Failed to initialize screen capture!");
        return -1;
    }
    
    // Get screen information
    cv::Size screen_size = capture.get_screen_size();
    cv::Point screen_center = capture.get_screen_center();
    logger.info("[MAIN][INFO] Screen size: " + std::to_string(screen_size.width) + "x" + std::to_string(screen_size.height));
    logger.info("[MAIN][INFO] Screen center: (" + std::to_string(screen_center.x) + ", " + std::to_string(screen_center.y) + ")");
    
    // Initialize YOLOv8 model for Bloodstrike
    std::string model_path = "models/blood.onnx";
    
    try {
        YOLOv8 yolov8_detector(model_path, 0.2f, 0.2f);
        
        // Configure FOV for Bloodstrike detection
        const int FOV_WIDTH = 400;
        const int FOV_HEIGHT = 400;
        yolov8_detector.set_fov_size(FOV_WIDTH, FOV_HEIGHT);
        
        logger.info("[MAIN][INFO] FOV configured: " + std::to_string(FOV_WIDTH) + "x" + std::to_string(FOV_HEIGHT));
        logger.info("[MAIN][INFO] FOV center relative to screen: (" + 
                   std::to_string(screen_center.x - FOV_WIDTH/2) + ", " + 
                   std::to_string(screen_center.y - FOV_HEIGHT/2) + ")");
        
        // Create windows for display
        cv::namedWindow("Bloodstrike FOV Detection", cv::WINDOW_NORMAL);
        cv::resizeWindow("Bloodstrike FOV Detection", FOV_WIDTH, FOV_HEIGHT);
        
        int frame_count = 0;
        while (true) {
            frame_count++;
            
            // Capture FOV region (400x400 centered on screen)
            cv::Mat fov_frame = capture.capture_fov(FOV_WIDTH, FOV_HEIGHT);
            
            if (fov_frame.empty()) {
                logger.error("[MAIN][ERROR] Failed to capture FOV!");
                continue;
            }
            
            // Detect objects in FOV
            auto fov_detections = yolov8_detector.detect_objects_fov(fov_frame);
            
            // Draw FOV detections with crosshair and metrics
            cv::Mat fov_result = yolov8_detector.draw_fov_detections(fov_frame, fov_detections);
            
            // Show FOV detection
            cv::imshow("Bloodstrike FOV Detection", fov_result);
            
            // Display detection info
            if (!fov_detections.empty()) {
                logger.info("[MAIN][INFO] Frame " + std::to_string(frame_count) + 
                          " - Detected " + std::to_string(fov_detections.size()) + " objects in FOV");
                
                for (size_t i = 0; i < fov_detections.size(); ++i) {
                    const auto& det = fov_detections[i];
                    logger.info("[MAIN][INFO] Detection " + std::to_string(i) + 
                              " - Class: " + std::to_string(det.class_id) + 
                              " - Score: " + std::to_string(det.score) +
                              " - Distance: " + std::to_string(det.fov_distance) +
                              " - Angle: " + std::to_string(det.fov_angle * 180 / 3.14159f) + "°");
                }
            }
            
            // Press 'q' to stop
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }
        
        cv::destroyAllWindows();
        
    } catch (const std::exception& e) {
        logger.error("[MAIN][ERROR] Exception captured: " + std::string(e.what()));
        return -1;
    } catch (...) {
        logger.error("[MAIN][ERROR] Unknown exception captured!");
        return -1;
    }
    
    return 0;
} 