cmake_minimum_required(VERSION 3.8)
project(wust_vision)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(${PROJECT_NAME}_openvino
  src/driver/hik.cpp
  src/wust_vision_openvino.cpp

  src/common/tools.cpp
  src/detect/mono_measure_tool.cpp
  src/detect/openvino.cpp
  src/detect/light_corner_corrector.cpp
  src/tracker/tracker.cpp
  src/tracker/extended_kalman_filter.cpp
  src/driver/serial.cpp
  src/common/gobal.cpp
  src/driver/crc8_crc16.cpp


)
add_executable(${PROJECT_NAME}_trt
  src/driver/hik.cpp
  src/wust_vision_trt.cpp
  src/detect/trt.cpp
  src/common/tools.cpp
  src/detect/mono_measure_tool.cpp

  src/detect/light_corner_corrector.cpp
  src/tracker/tracker.cpp
  src/tracker/extended_kalman_filter.cpp
  src/driver/serial.cpp
  src/common/gobal.cpp
  src/driver/crc8_crc16.cpp



)
include_directories(
  /home/hy/TensorRT-8.5.2.2/include
  /home/hy/wust_vision/thirdparty/matplotlibcpp
  /usr/local/cuda-11.8/include
)

link_directories(
  /home/hy/TensorRT-8.5.2.2/lib
  /usr/local/cuda-11.8/lib64
)



# 查找依赖
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)



# 包含路径
target_include_directories(${PROJECT_NAME}_openvino PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${OpenVINO_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/hikSDK/include

  ${Python3_INCLUDE_DIRS}


)
target_include_directories(${PROJECT_NAME}_trt PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${OpenVINO_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/hikSDK/include

  ${Python3_INCLUDE_DIRS}


)
set(TENSORRT_LIBS
  /home/hy/TensorRT-8.5.2.2/lib/libnvinfer.so
  /home/hy/TensorRT-8.5.2.2/lib/libnvinfer_plugin.so
  /home/hy/TensorRT-8.5.2.2/lib/libnvonnxparser.so
  /home/hy/TensorRT-8.5.2.2/lib/libnvparsers.so
)

# 链接路径
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(${PROJECT_NAME}_openvino PUBLIC hikSDK/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(${PROJECT_NAME}_openvino PUBLIC hikSDK/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported host system architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(${PROJECT_NAME}_trt PUBLIC hikSDK/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(${PROJECT_NAME}_trt PUBLIC hikSDK/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported host system architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()
# 链接库
target_link_libraries(${PROJECT_NAME}_openvino
  FormatConversion
  MediaProcess
  MvCameraControl
  MVRender
  MvUsb3vTL
  ${OpenCV_LIBS}
  openvino::frontend::onnx
  openvino::runtime
  ${OpenVINO_LIBRARIES}
  ${OpenVINO_RUNTIME_LIBRARIES}
  ${OpenVINO_ONNX_LIBRARIES}
  ${TENSORRT_LIBS}
  /usr/local/cuda-11.8/lib64/libcudart.so
  fmt::fmt
  yaml-cpp
  ${Python3_LIBRARIES}

  


 
)
target_link_libraries(${PROJECT_NAME}_trt
  FormatConversion
  MediaProcess
  MvCameraControl
  MVRender
  MvUsb3vTL
  ${OpenCV_LIBS}
  openvino::frontend::onnx
  openvino::runtime
  ${OpenVINO_LIBRARIES}
  ${OpenVINO_RUNTIME_LIBRARIES}
  ${OpenVINO_ONNX_LIBRARIES}
  ${TENSORRT_LIBS}
  /usr/local/cuda-11.8/lib64/libcudart.so
  fmt::fmt
  yaml-cpp
  ${Python3_LIBRARIES}


 
)
