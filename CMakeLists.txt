cmake_minimum_required(VERSION 3.16)
project(YOLOv8VideoDetection)

# Enable compile commands for IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GPU Optimization Options
option(USE_GPU "Enable GPU acceleration" ON)
option(GPU_PROVIDER "GPU provider (DirectML/CUDA)" "DirectML")
option(OPTIMIZE_FOR_AMD "Optimize for AMD GPUs" ON)

# Performance optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /GL /arch:AVX2")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /fp:fast /ffast-math")
    message(STATUS "Release build optimizations enabled")
endif()

# Try to find OpenCV with multiple methods
find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
    # Try to find OpenCV in common installation paths
    set(OpenCV_POSSIBLE_PATHS
        "D:/softwares/opencv/build"
        "D:/softwares/opencv/build/x64/vc16"
        "D:/softwares/opencv/build/x64/vc15"
        "D:/softwares/opencv/build/x64/vc14"
        "C:/opencv/build"
        "C:/opencv/build/x64/vc16"
        "C:/opencv/build/x64/vc15"
        "C:/opencv/build/x64/vc14"
        "C:/Program Files/opencv/build"
        "C:/Program Files (x86)/opencv/build"
        "$ENV{OpenCV_DIR}"
        "$ENV{OPENCV_DIR}"
    )
    
    foreach(OpenCV_PATH ${OpenCV_POSSIBLE_PATHS})
        if(EXISTS "${OpenCV_PATH}/OpenCVConfig.cmake")
            set(OpenCV_DIR "${OpenCV_PATH}")
            break()
        endif()
    endforeach()
    
    find_package(OpenCV REQUIRED)
endif()

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV and set OpenCV_DIR environment variable.")
endif()

message(STATUS "OpenCV found: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# Set ONNX Runtime paths - use ONLY 1.22.1
set(ONNXRUNTIME_POSSIBLE_PATHS
    "D:/softwares/onnxruntime/onnxruntime-win-x64-1.22.1"
)

set(ONNXRUNTIME_ROOT_DIR "")

foreach(ONNX_PATH ${ONNXRUNTIME_POSSIBLE_PATHS})
    if(EXISTS "${ONNX_PATH}/include/onnxruntime_cxx_api.h")
        set(ONNXRUNTIME_ROOT_DIR "${ONNX_PATH}")
        break()
    endif()
endforeach()

if(ONNXRUNTIME_ROOT_DIR STREQUAL "")
    message(FATAL_ERROR "ONNX Runtime not found. Please install ONNX Runtime and update the paths in CMakeLists.txt")
endif()

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT_DIR}/lib")

message(STATUS "ONNX Runtime found at: ${ONNXRUNTIME_ROOT_DIR}")

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Create executable with all source files
add_executable(video_object_detection 
    src/main.cpp
    src/yolov8_detector.cpp
    src/yolov8_model.cpp
    src/yolov8_preprocessor.cpp
    src/yolov8_postprocessor.cpp
    src/yolov8_visualizer.cpp
    src/fov_processor.cpp
    src/windows_graphics_capture.cpp
)

# Add GPU optimization definitions
if(USE_GPU)
    target_compile_definitions(video_object_detection PRIVATE
        USE_GPU=1
        GPU_PROVIDER="${GPU_PROVIDER}"
    )
    
    if(OPTIMIZE_FOR_AMD)
        target_compile_definitions(video_object_detection PRIVATE
            OPTIMIZE_FOR_AMD=1
        )
        message(STATUS "AMD GPU optimizations enabled")
    endif()
    
    message(STATUS "GPU acceleration enabled with provider: ${GPU_PROVIDER}")
else()
    message(STATUS "GPU acceleration disabled - using CPU only")
endif()

# Link libraries
target_link_libraries(video_object_detection ${OpenCV_LIBS})

# Link ONNX Runtime
if(EXISTS "${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib")
    target_link_libraries(video_object_detection "${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib")
    message(STATUS "ONNX Runtime linked successfully")
else()
    message(FATAL_ERROR "ONNX Runtime library not found at ${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib")
endif()

# Windows Graphics Capture libraries
if(WIN32)
    target_link_libraries(video_object_detection 
        d3d11.lib 
        dxgi.lib 
        windowsapp.lib
        d3dcompiler.lib
        dxguid.lib
    )
    
    # Add Windows SDK include directories - try multiple possible paths
    target_include_directories(video_object_detection PRIVATE
        "$ENV{WINDOWSSDK_DIR}/Include/$ENV{WINDOWSSDK_VERSION}/um"
        "$ENV{WINDOWSSDK_DIR}/Include/$ENV{WINDOWSSDK_VERSION}/shared"
        "$ENV{WINDOWSSDK_DIR}/Include/$ENV{WINDOWSSDK_VERSION}/winrt"
        "C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/um"
        "C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/shared"
        "C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/winrt"
        "C:/Program Files (x86)/Windows Kits/10/Include/10.0.22000.0/um"
        "C:/Program Files (x86)/Windows Kits/10/Include/10.0.22000.0/shared"
        "C:/Program Files (x86)/Windows Kits/10/Include/10.0.22000.0/winrt"
        "C:/Program Files (x86)/Windows Kits/10/Include/10.0.19041.0/um"
        "C:/Program Files (x86)/Windows Kits/10/Include/10.0.19041.0/shared"
        "C:/Program Files (x86)/Windows Kits/10/Include/10.0.19041.0/winrt"
    )
    
    # Add Windows SDK library directories - try multiple possible paths
    target_link_directories(video_object_detection PRIVATE
        "$ENV{WINDOWSSDK_DIR}/Lib/$ENV{WINDOWSSDK_VERSION}/um/x64"
        "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22621.0/um/x64"
        "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22000.0/um/x64"
        "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.19041.0/um/x64"
    )
    
    message(STATUS "Windows Graphics Capture libraries linked")
endif()

# Copy ONNX Runtime DLL to output directory (Windows)
if(WIN32 AND EXISTS "${ONNXRUNTIME_ROOT_DIR}/lib/onnxruntime.dll")
    add_custom_command(TARGET video_object_detection POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT_DIR}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:video_object_detection>
    )
    message(STATUS "ONNX Runtime DLL will be copied to output directory")
endif()

# Set output directory
set_target_properties(video_object_detection PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

message(STATUS "Configuration complete. Build the project with: cmake --build . --config Release") 