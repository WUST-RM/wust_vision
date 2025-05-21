cmake_minimum_required(VERSION 3.8)
project(wust_vision)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(${PROJECT_NAME}_trt
  src/driver/hik.cpp
  src/wust_vision_trt.cpp
  src/detect/trt.cpp
  src/common/tools.cpp
  src/detect/mono_measure_tool.cpp
  src/detect/light_corner_corrector.cpp
  src/tracker/tracker.cpp
  src/tracker/extended_kalman_filter.cpp
  src/driver/crc8_crc16.cpp
  src/driver/serial.cpp
  src/control/armor_solver.cpp
  src/control/manual_compensator.cpp
  src/control/trajectory_compensator.cpp
  src/common/gobal.cpp
  src/common/calculation.cpp
)

# 基础 include 路径
include_directories(
  /usr/local/cuda/include
  /usr/include/aarch64-linux-gnu  # Jetson 上 TensorRT 的默认 include 路径
)

# TensorRT library 路径（Jetson 默认）
link_directories(
  /usr/local/cuda/lib64
  /usr/lib/aarch64-linux-gnu
)

# 查找依赖
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)

# 本项目 include 路径
target_include_directories(${PROJECT_NAME}_trt PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/hikSDK/include
)

# 相机 SDK 库路径
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(${PROJECT_NAME}_trt PUBLIC hikSDK/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(${PROJECT_NAME}_trt PUBLIC hikSDK/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported host system architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

# 链接库
target_link_libraries(${PROJECT_NAME}_trt
  FormatConversion
  MediaProcess
  MvCameraControl
  MVRender
  MvUsb3vTL

  ${OpenCV_LIBS}
  fmt::fmt
  yaml-cpp

  # CUDA runtime
  /usr/local/cuda/lib64/libcudart.so

  # TensorRT 相关库
  nvinfer
  nvonnxparser
  nvparsers
  nvinfer_plugin
)