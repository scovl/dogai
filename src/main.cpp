#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include "logger.hpp"
#include "yolov8_detector.hpp"
#include "windows_graphics_capture.hpp"
#include "config_manager.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <fstream>

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
    
    // Load unified configuration
    ConfigManager config("blood.cfg");
    
    // Check performance mode
    std::string perf_mode = config.get_string("Performance", "performance_mode", "normal");
    if (perf_mode == "maximum") {
        logger.info("[MAIN][INFO] Maximum performance mode enabled - using ultra high FPS settings");
    }
    
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
        
        // FPS Control Configuration
        const int TARGET_FPS = config.get_int("Performance", "target_fps", 120);
        const std::chrono::microseconds FRAME_TIME(1000000 / TARGET_FPS);
        
        int frame_count = 0;
        auto last_frame_time = std::chrono::high_resolution_clock::now();
        auto fps_start_time = last_frame_time;
        
        // FPS measurement variables
        std::vector<double> fps_history;
        double current_fps = 0.0;
        double average_fps = 0.0;
        int fps_measurement_interval = config.get_int("Performance", "fps_measurement_interval", 60);
        bool enable_fps_logging = config.get_string("Performance", "enable_fps_logging", "true") == "true";
        
        logger.info("[MAIN][INFO] Target FPS: " + std::to_string(TARGET_FPS));
        logger.info("[MAIN][INFO] FPS measurement enabled - logging every " + std::to_string(fps_measurement_interval) + " frames");
        logger.info("[MAIN][INFO] FPS will be displayed on screen and in logs");
        
        while (true) {
            auto frame_start_time = std::chrono::high_resolution_clock::now();
            frame_count++;
            
            // Log first frame to show FPS measurement is active
            if (frame_count == 1) {
                logger.info("[MAIN][INFO] Starting FPS measurement...");
            }
            
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
            
            // Add FPS text to the image
            cv::Mat display_image = fov_result.clone();
            std::string fps_text = "FPS: " + std::to_string(static_cast<int>(current_fps)) + 
                                  " | Avg: " + std::to_string(static_cast<int>(average_fps)) + 
                                  " | Target: " + std::to_string(TARGET_FPS);
            
            // Draw FPS text on image
            cv::putText(display_image, fps_text, cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            // Show FOV detection with FPS
            cv::imshow("Bloodstrike FOV Detection", display_image);
            
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
            
            // Calculate and display FPS every measurement interval
            if (frame_count % fps_measurement_interval == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - fps_start_time);
                
                if (elapsed.count() > 0) {
                    current_fps = (fps_measurement_interval * 1000.0) / elapsed.count();
                    fps_history.push_back(current_fps);
                    
                    // Calculate average FPS (keep last 10 measurements)
                    if (fps_history.size() > 10) {
                        fps_history.erase(fps_history.begin());
                    }
                    
                    average_fps = 0.0;
                    for (double fps : fps_history) {
                        average_fps += fps;
                    }
                    average_fps /= fps_history.size();
                    
                    // Log detailed FPS information if enabled
                    if (enable_fps_logging) {
                        logger.info("[MAIN][FPS] Frame: " + std::to_string(frame_count) + 
                                  " | Current: " + std::to_string(static_cast<int>(current_fps)) + 
                                  " | Average: " + std::to_string(static_cast<int>(average_fps)) + 
                                  " | Target: " + std::to_string(TARGET_FPS) +
                                  " | Elapsed: " + std::to_string(elapsed.count()) + "ms");
                    } else {
                        // Always log basic FPS info
                        logger.info("[MAIN][FPS] Current: " + std::to_string(static_cast<int>(current_fps)) + 
                                  " | Average: " + std::to_string(static_cast<int>(average_fps)));
                    }
                    
                    fps_start_time = current_time;
                }
            }
            
            // FPS Control - Sleep if we're running too fast
            auto frame_end_time = std::chrono::high_resolution_clock::now();
            auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>(frame_end_time - frame_start_time);
            
            if (frame_duration < FRAME_TIME) {
                auto sleep_time = FRAME_TIME - frame_duration;
                std::this_thread::sleep_for(sleep_time);
            }
            
            // Press 'q' to stop
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }
        
        // Final FPS statistics
        if (!fps_history.empty()) {
            logger.info("[MAIN][FINAL] ===== FPS STATISTICS =====");
            logger.info("[MAIN][FINAL] Total frames processed: " + std::to_string(frame_count));
            logger.info("[MAIN][FINAL] Final average FPS: " + std::to_string(static_cast<int>(average_fps)));
            logger.info("[MAIN][FINAL] Target FPS: " + std::to_string(TARGET_FPS));
            
            // Calculate min/max FPS
            double min_fps = *std::min_element(fps_history.begin(), fps_history.end());
            double max_fps = *std::max_element(fps_history.begin(), fps_history.end());
            logger.info("[MAIN][FINAL] Min FPS: " + std::to_string(static_cast<int>(min_fps)));
            logger.info("[MAIN][FINAL] Max FPS: " + std::to_string(static_cast<int>(max_fps)));
            logger.info("[MAIN][FINAL] Performance: " + std::string(average_fps >= TARGET_FPS * 0.9 ? "EXCELLENT" : 
                                                                   average_fps >= TARGET_FPS * 0.7 ? "GOOD" : "NEEDS OPTIMIZATION"));
            logger.info("[MAIN][FINAL] =========================");
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