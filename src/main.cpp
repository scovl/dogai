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
    
    // Initialize YOLOv8 model
    std::string model_path = "models/blood.onnx";
    
    try {
        YOLOv8 yolov8_detector(model_path, 0.2f, 0.2f);
        
        cv::namedWindow("Screen Object Detection", cv::WINDOW_NORMAL);
        
        int frame_count = 0;
        while (true) {
            frame_count++;
            
            // Capture screen
            cv::Mat frame = capture.capture_screen();
            
            if (frame.empty()) {
                logger.error("[MAIN][ERROR] Failed to capture screen!");
                continue;
            }
            
            // Detect objects
            auto detections = yolov8_detector.detect_objects(frame);
            
            // Draw detections
            cv::Mat result = yolov8_detector.draw_detections(frame, detections);
            
            cv::imshow("Screen Object Detection", result);
            
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