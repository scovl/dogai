#include "yolov8_detector.hpp"

YOLOv8::YOLOv8(const std::string& model_path, float conf_thres, float iou_thres) 
    : conf_threshold(conf_thres), iou_threshold(iou_thres), config() {
    
    // Load configuration from file
    load_config_from_file();
    
    // Initialize colors
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 255.0f);
    
    for (size_t i = 0; i < class_names.size(); ++i) {
        colors.push_back(cv::Scalar(dis(gen), dis(gen), dis(gen)));
    }
    
    initialize_model(model_path);
}

std::vector<YOLOv8::Detection> YOLOv8::detect_objects(const cv::Mat& image) {
    std::vector<float> input_tensor = prepare_input(image);
    std::vector<Ort::Value> outputs = inference(input_tensor);
    auto dets = process_output(outputs, image.size());
    return dets;
}

cv::Mat YOLOv8::draw_detections(const cv::Mat& image, const std::vector<Detection>& detections) {
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
            label = "Class " + std::to_string(det.class_id);
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

void YOLOv8::load_config_from_file() {
    // Carregar parâmetros do modelo
    input_width = config.get_int("Model", "input_width", 640);
    input_height = config.get_int("Model", "input_height", 640);
    conf_threshold = config.get_float("Model", "conf_threshold", 0.3f);
    iou_threshold = config.get_float("Model", "iou_threshold", 0.5f);
    
    // Log de todas as configurações
    config.log_config();
}

void YOLOv8::initialize_model(const std::string& model_path) {
    try {
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
        
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Fix: use wstring for model path
        std::wstring wmodel_path(model_path.begin(), model_path.end());
        session = Ort::Session(env, wmodel_path.c_str(), session_options);
        
        // Get input and output names (fixed for new API)
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Get input names
        size_t num_inputs = session.GetInputCount();
        
        for (size_t i = 0; i < num_inputs; ++i) {
            auto input_name_alloc = session.GetInputNameAllocated(i, allocator);
            std::string input_name = input_name_alloc.get();
            input_names.push_back(input_name);
        }
        
        // Get output names
        size_t num_outputs = session.GetOutputCount();
        
        for (size_t i = 0; i < num_outputs; ++i) {
            auto output_name_alloc = session.GetOutputNameAllocated(i, allocator);
            std::string output_name = output_name_alloc.get();
            output_names.push_back(output_name);
        }
        
        // Log detailed input of the model (equal to Python)
        for (size_t i = 0; i < num_inputs; ++i) {
            auto input_name_alloc = session.GetInputNameAllocated(i, allocator);
            std::string input_name = input_name_alloc.get();
            
            Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType type = tensor_info.GetElementType();
            std::vector<int64_t> input_dims = tensor_info.GetShape();
            
            std::string type_str;
            switch (type) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: type_str = "float32"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: type_str = "uint8"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: type_str = "int8"; break;
                default: type_str = "unknown"; break;
            }
            
            std::string dims_str;
            for (size_t j = 0; j < input_dims.size(); ++j) {
                dims_str += std::to_string(input_dims[j]);
                if (j + 1 < input_dims.size()) dims_str += ", ";
            }
        }
        
    } catch (const std::exception& e) {
        logger.error("[YOLOv8][ERROR] Failed to initialize model: " + std::string(e.what()));
        throw;
    }
}

std::vector<float> YOLOv8::prepare_input(const cv::Mat& image) {
    if (image.empty()) {
        logger.error("[YOLOv8][ERROR] Input image is empty!");
    }
    // 1. Convert BGR to RGB
    cv::Mat rgb_img;
    cv::cvtColor(image, rgb_img, cv::COLOR_BGR2RGB);
    // 2. Resize
    cv::Mat resized;
    cv::resize(rgb_img, resized, cv::Size(input_width, input_height));
    // 3. Normalize to [0,1]
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32F, 1.0/255.0);
    // 4. Transpose to NCHW
    std::vector<cv::Mat> chw(3);
    cv::split(float_img, chw);
    std::vector<float> input_tensor(input_width * input_height * 3);
    size_t channel_size = input_width * input_height;
    for (int c = 0; c < 3; ++c) {
        memcpy(input_tensor.data() + c * channel_size, chw[c].data, channel_size * sizeof(float));
    }
    return input_tensor;
}

std::vector<Ort::Value> YOLOv8::inference(const std::vector<float>& input_tensor) {
    try {
        if (input_names.empty() || output_names.empty()) {
            logger.error("[YOLOv8][ERROR] Input or output names are empty!");
            throw std::runtime_error("Input or output names are empty");
        }
        std::vector<const char*> input_names_char(input_names.size());
        std::vector<const char*> output_names_char(output_names.size());
        for (size_t i = 0; i < input_names.size(); ++i) {
            input_names_char[i] = input_names[i].c_str();
        }
        for (size_t i = 0; i < output_names.size(); ++i) {
            output_names_char[i] = output_names[i].c_str();
        }
        std::vector<Ort::Value> input_tensors;
        try {
            std::vector<int64_t> input_shape = {1, 3, input_height, input_width};
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
                const_cast<float*>(input_tensor.data()),
                input_tensor.size(),
                input_shape.data(),
                input_shape.size()));
        } catch (const std::exception& e) {
            logger.error("[YOLOv8][ERROR] Failed to create input tensor: " + std::string(e.what()));
            throw;
        }
        auto result = session.Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(), input_tensors.size(), output_names_char.data(), output_names_char.size());
        return result;
    } catch (const std::exception& e) {
        logger.error("[YOLOv8][ERROR] Failed to execute inference: " + std::string(e.what()));
        throw;
    }
}

std::vector<YOLOv8::Detection> YOLOv8::process_output(const std::vector<Ort::Value>& outputs, const cv::Size& original_size) {
    std::vector<Detection> detections;
    if (outputs.empty()) {
        logger.error("[YOLOv8][ERROR] Model output is empty!");
        return detections;
    }
    
    // Get the first output tensor
    const Ort::Value& output = outputs[0];
    std::vector<int64_t> output_shape = output.GetTensorTypeAndShapeInfo().GetShape();

    // Check expected shape
    if (output_shape.size() < 2) {
        logger.error("[YOLOv8][ERROR] Unexpected output shape! Expected at least 2 dimensions.");
        return detections;
    }
    
    float* output_data = nullptr;
    try {
        output_data = const_cast<float*>(output.GetTensorData<float>());
    } catch (const std::exception& e) {
        logger.error(std::string("[YOLOv8][ERROR] Failed to access output tensor data: ") + e.what());
        return detections;
    }
    if (!output_data) {
        logger.error("[YOLOv8][ERROR] Output tensor data pointer is null!");
        return detections;
    }
    
    // Option 1: [1, N, 4+num_classes] - default YOLOv8 format
    if (output_shape.size() == 3) {
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
        logger.error("[YOLOv8][ERROR] Output format not recognized!");
    }
    
    // Apply NMS
    return non_max_suppression(detections);
}

std::vector<YOLOv8::Detection> YOLOv8::non_max_suppression(const std::vector<Detection>& detections) {
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

float YOLOv8::calculate_iou(const cv::Rect& box1, const cv::Rect& box2) {
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