#include "yolov8_postprocessor.hpp"
#include <algorithm>
#include <numeric>

YOLOv8Postprocessor::YOLOv8Postprocessor(float conf_thres, float iou_thres, int width, int height)
    : conf_threshold(conf_thres), iou_threshold(iou_thres), input_width(width), input_height(height) {
}

void YOLOv8Postprocessor::set_thresholds(float conf_thres, float iou_thres) {
    conf_threshold = conf_thres;
    iou_threshold = iou_thres;
}

void YOLOv8Postprocessor::set_input_size(int width, int height) {
    input_width = width;
    input_height = height;
}

std::vector<Detection> YOLOv8Postprocessor::process_output(const std::vector<Ort::Value>& outputs, const cv::Size& original_size) {
    std::vector<Detection> detections;
    if (outputs.empty()) {
        logger.error("[YOLOv8Postprocessor][ERROR] Model output is empty!");
        return detections;
    }
    
    // Get the first output tensor
    const Ort::Value& output = outputs[0];
    std::vector<int64_t> output_shape = output.GetTensorTypeAndShapeInfo().GetShape();

    // Check expected shape
    if (output_shape.size() < 2) {
        logger.error("[YOLOv8Postprocessor][ERROR] Unexpected output shape! Expected at least 2 dimensions.");
        return detections;
    }
    
    float* output_data = nullptr;
    try {
        output_data = const_cast<float*>(output.GetTensorData<float>());
    } catch (const std::exception& e) {
        logger.error(std::string("[YOLOv8Postprocessor][ERROR] Failed to access output tensor data: ") + e.what());
        return detections;
    }
    if (!output_data) {
        logger.error("[YOLOv8Postprocessor][ERROR] Output tensor data pointer is null!");
        return detections;
    }
    
    // Suporte ao formato [1, 5, 8400] do blood.onnx
    if (output_shape.size() == 3 && output_shape[1] == 5) {
        int num_boxes = static_cast<int>(output_shape[2]);
        int img_width = original_size.width;
        int img_height = original_size.height;
        int input_w = input_width;
        int input_h = input_height;

        for (int i = 0; i < num_boxes; ++i) {
            float x = output_data[0 * num_boxes + i];
            float y = output_data[1 * num_boxes + i];
            float w = output_data[2 * num_boxes + i];
            float h = output_data[3 * num_boxes + i];
            float score = output_data[4 * num_boxes + i];

            if (score > conf_threshold) {
                float x_scaled = x / input_w * img_width;
                float y_scaled = y / input_h * img_height;
                float w_scaled = w / input_w * img_width;
                float h_scaled = h / input_h * img_height;
                float x1 = x_scaled - w_scaled / 2.0f;
                float y1 = y_scaled - h_scaled / 2.0f;
                float x2 = x_scaled + w_scaled / 2.0f;
                float y2 = y_scaled + h_scaled / 2.0f;
                x1 = std::max(0.0f, std::min(x1, static_cast<float>(img_width-1)));
                y1 = std::max(0.0f, std::min(y1, static_cast<float>(img_height-1)));
                x2 = std::max(0.0f, std::min(x2, static_cast<float>(img_width-1)));
                y2 = std::max(0.0f, std::min(y2, static_cast<float>(img_height-1)));
                Detection det;
                det.box = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
                det.score = score;
                det.class_id = 0;
                detections.push_back(det);
            }
        }
        // Aplica NMS e retorna
        return non_max_suppression(detections);
    }
    // Option 2: [1, N, 4+num_classes] - default YOLOv8 format
    else if (output_shape.size() == 3) {
        int num_boxes = static_cast<int>(output_shape[2]); // 8400
        int num_classes = static_cast<int>(output_shape[1]) - 4; // 1 (se blood.onnx for custom)
        int img_width = original_size.width;
        int img_height = original_size.height;
        int input_w = input_width;
        int input_h = input_height;

        // Transpose: predictions = np.squeeze(output[0]).T
        std::vector<float> transposed_output(num_boxes * output_shape[1]);
        for (int i = 0; i < output_shape[1]; ++i) {
            for (int j = 0; j < num_boxes; ++j) {
                transposed_output[j * output_shape[1] + i] = output_data[i * num_boxes + j];
            }
        }

        // For each box
        for (int i = 0; i < num_boxes; ++i) {
            float* pred = transposed_output.data() + i * output_shape[1];
            float x = pred[0];
            float y = pred[1];
            float w = pred[2];
            float h = pred[3];
            float score = pred[4];
            int class_id = 0;
            float max_score = score;
            // If there is more than 1 class, get the one with the highest score
            if (num_classes > 1) {
                max_score = pred[4];
                class_id = 0;
                for (int c = 0; c < num_classes; ++c) {
                    float s = pred[4 + c];
                    if (s > max_score) {
                        max_score = s;
                        class_id = c;
                    }
                }
            }
            if (max_score > conf_threshold) {
                // Rescale like in Python
                float x_scaled = x / input_w * img_width;
                float y_scaled = y / input_h * img_height;
                float w_scaled = w / input_w * img_width;
                float h_scaled = h / input_h * img_height;
                // xywh2xyxy
                float x1 = x_scaled - w_scaled / 2.0f;
                float y1 = y_scaled - h_scaled / 2.0f;
                float x2 = x_scaled + w_scaled / 2.0f;
                float y2 = y_scaled + h_scaled / 2.0f;
                // Clamp
                x1 = std::max(0.0f, std::min(x1, static_cast<float>(img_width-1)));
                y1 = std::max(0.0f, std::min(y1, static_cast<float>(img_height-1)));
                x2 = std::max(0.0f, std::min(x2, static_cast<float>(img_width-1)));
                y2 = std::max(0.0f, std::min(y2, static_cast<float>(img_height-1)));
                Detection det;
                det.box = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
                det.score = max_score;
                det.class_id = class_id;
                detections.push_back(det);
            }
        }
    }
    // Option 2: [1, num_classes, N] - transposed format
    else if (output_shape.size() == 3 && output_shape[1] <= 100) {
        int num_classes = static_cast<int>(output_shape[1]);
        int num_boxes = static_cast<int>(output_shape[2]);
        
        // For this format, we would need to restructure the data
    }
    // Option 3: [1, N] - linear format
    else if (output_shape.size() == 2) {
        int total_elements = static_cast<int>(output_shape[1]);
        
        // Tentar interpretar como formato linear
        int elements_per_box = total_elements / 100; // Assumir ~100 boxes
        if (elements_per_box >= 5) { // x, y, w, h, class_id
            int num_boxes = total_elements / elements_per_box;
            
            for (int i = 0; i < num_boxes; ++i) {
                float* box_data = output_data + i * elements_per_box;
                float x = box_data[0];
                float y = box_data[1];
                float w = box_data[2];
                float h = box_data[3];
                float score = box_data[4];
                int class_id = static_cast<int>(box_data[5]);
                
                if (score > conf_threshold && class_id >= 0 && class_id < 100) {
                    // Convert to xyxy format
                    float x1 = x - w / 2;
                    float y1 = y - h / 2;
                    float x2 = x + w / 2;
                    float y2 = y + h / 2;
                    
                    // Scale to original image size
                    x1 *= original_size.width;
                    y1 *= original_size.height;
                    x2 *= original_size.width;
                    y2 *= original_size.height;
                    
                    // Protect against absurd values
                    x1 = std::max(0.0f, std::min(x1, static_cast<float>(original_size.width-1)));
                    y1 = std::max(0.0f, std::min(y1, static_cast<float>(original_size.height-1)));
                    x2 = std::max(0.0f, std::min(x2, static_cast<float>(original_size.width-1)));
                    y2 = std::max(0.0f, std::min(y2, static_cast<float>(original_size.height-1)));
                    
                    Detection det;
                    det.box = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
                    det.score = score;
                    det.class_id = class_id;
                    detections.push_back(det);
                }
            }
        }
    }
    else {
        logger.error("[YOLOv8Postprocessor][ERROR] Output format not recognized!");
    }
    
    // Apply NMS
    return non_max_suppression(detections);
}

std::vector<Detection> YOLOv8Postprocessor::non_max_suppression(const std::vector<Detection>& detections) {
    if (detections.empty()) return detections;
    
    std::vector<Detection> result;
    std::vector<bool> used(detections.size(), false);
    
    // Sort by confidence
    std::vector<size_t> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) { return detections[a].score > detections[b].score; });
    
    for (size_t i = 0; i < indices.size(); ++i) {
        if (used[indices[i]]) continue;
        result.push_back(detections[indices[i]]);
        used[indices[i]] = true;
        for (size_t j = i + 1; j < indices.size(); ++j) {
            if (used[indices[j]]) continue;
            if (calculate_iou(detections[indices[i]].box, detections[indices[j]].box) > iou_threshold) {
                used[indices[j]] = true;
            }
        }
    }
    return result;
}

float YOLOv8Postprocessor::calculate_iou(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    
    int intersection = (x2 - x1) * (y2 - y1);
    int area1 = box1.width * box1.height;
    int area2 = box2.width * box2.height;
    int union_area = area1 + area2 - intersection;
    
    return static_cast<float>(intersection) / union_area;
} 