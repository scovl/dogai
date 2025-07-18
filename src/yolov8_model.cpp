#include "yolov8_model.hpp"

YOLOv8Model::YOLOv8Model(const std::string& model_path, float conf_thres, float iou_thres) 
    : conf_threshold(conf_thres), iou_threshold(iou_thres), config("blood.cfg") {
    
    // Load configuration from file
    load_config_from_file();
    
    initialize_model(model_path);
}

void YOLOv8Model::load_config_from_file() {
    // Carregar parâmetros do modelo
    input_width = config.get_int("Model", "input_width", 640);
    input_height = config.get_int("Model", "input_height", 640);
    conf_threshold = config.get_float("Model", "conf_threshold", 0.3f);
    iou_threshold = config.get_float("Model", "iou_threshold", 0.5f);
    
    // Log de todas as configurações
    config.log_config();
}

void YOLOv8Model::initialize_model(const std::string& model_path) {
    try {
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
        
        auto session_options = Ort::SessionOptions();
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // CPU Optimization for AMD RX 7600 XT
        // Using CPU with maximum optimizations for high FPS
        session_options.SetIntraOpNumThreads(8); // Use more CPU threads
        session_options.SetInterOpNumThreads(4); // Parallel execution
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        logger.info("[YOLOv8Model][INFO] CPU optimization enabled for high FPS");
        logger.info("[YOLOv8Model][INFO] Using 8 threads for maximum performance");
        
        // Fix: use wstring for model path
        auto wmodel_path = std::wstring(model_path.begin(), model_path.end());
        session = Ort::Session(env, wmodel_path.c_str(), session_options);
        
        // Get input and output names (fixed for new API)
        auto allocator = Ort::AllocatorWithDefaultOptions();
        
        // Get input names
        auto num_inputs = session.GetInputCount();
        
        for (size_t i = 0; i < num_inputs; ++i) {
            auto input_name_alloc = session.GetInputNameAllocated(i, allocator);
            auto input_name = input_name_alloc.get();
            input_names.push_back(input_name);
        }
        
        // Get output names
        auto num_outputs = session.GetOutputCount();
        
        for (size_t i = 0; i < num_outputs; ++i) {
            auto output_name_alloc = session.GetOutputNameAllocated(i, allocator);
            auto output_name = output_name_alloc.get();
            output_names.push_back(output_name);
        }
        
        // Log detailed input of the model (equal to Python)
        for (size_t i = 0; i < num_inputs; ++i) {
            auto input_name_alloc = session.GetInputNameAllocated(i, allocator);
            auto input_name = input_name_alloc.get();
            
            auto type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto type = tensor_info.GetElementType();
            auto input_dims = tensor_info.GetShape();
            
            auto type_str = std::string();
            switch (type) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: type_str = "float32"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: type_str = "uint8"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: type_str = "int8"; break;
                default: type_str = "unknown"; break;
            }
            
            auto dims_str = std::string();
            for (size_t j = 0; j < input_dims.size(); ++j) {
                dims_str += std::to_string(input_dims[j]);
                if (j + 1 < input_dims.size()) dims_str += ", ";
            }
        }
        
    } catch (const std::exception& e) {
        logger.error("[YOLOv8Model][ERROR] Failed to initialize model: " + std::string(e.what()));
        throw;
    }
}

std::vector<Ort::Value> YOLOv8Model::run_inference(const std::vector<float>& input_tensor) {
    try {
        if (input_names.empty() || output_names.empty()) {
            logger.error("[YOLOv8Model][ERROR] Input or output names are empty!");
            throw std::runtime_error("Input or output names are empty");
        }
        auto input_names_char = std::vector<const char*>(input_names.size());
        auto output_names_char = std::vector<const char*>(output_names.size());
        for (size_t i = 0; i < input_names.size(); ++i) {
            input_names_char[i] = input_names[i].c_str();
        }
        for (size_t i = 0; i < output_names.size(); ++i) {
            output_names_char[i] = output_names[i].c_str();
        }
        auto input_tensors = std::vector<Ort::Value>();
        try {
            auto input_shape = std::vector<int64_t>{1, 3, input_height, input_width};
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
                const_cast<float*>(input_tensor.data()),
                input_tensor.size(),
                input_shape.data(),
                input_shape.size()));
        } catch (const std::exception& e) {
            logger.error("[YOLOv8Model][ERROR] Failed to create input tensor: " + std::string(e.what()));
            throw;
        }
        auto result = session.Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(), input_tensors.size(), output_names_char.data(), output_names_char.size());
        return result;
    } catch (const std::exception& e) {
        logger.error("[YOLOv8Model][ERROR] Failed to execute inference: " + std::string(e.what()));
        throw;
    }
} 