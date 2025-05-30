cmake_minimum_required(VERSION 3.8)
project(wust_vision)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")
add_executable(${PROJECT_NAME}_openvino

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
  src/control/trajectory_compensator.cpp
  src/control/armor_solver.cpp
  src/control/manual_compensator.cpp
  src/common/calculation.cpp
  src/detect/pnp_solver.cpp
  src/detect/graph_optimizer.cpp
  src/detect/armor_pose_estimator.cpp
  src/detect/ba_solver.cpp
  src/driver/image_capturer.cpp
  
)



set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXTERNAL_INCLUDE_DIRS ${G2O_INCLUDE_DIR})
set(EXTERNAL_LIBS ${G2O_LIBRARIES})

# 查找依赖
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(G2O REQUIRED)



# 包含路径
target_include_directories(${PROJECT_NAME}_openvino PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${OpenVINO_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/hikSDK/include
  ${G2O_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}


)


# 链接路径
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(${PROJECT_NAME}_openvino PUBLIC hikSDK/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(${PROJECT_NAME}_openvino PUBLIC hikSDK/lib/arm64)
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


  fmt::fmt
  yaml-cpp
  ${Python3_LIBRARIES}
  g2o_core
  g2o_stuff
  g2o_solver_csparse
  g2o_types_sba
  g2o_types_slam3d
  g2o_solver_dense

  


 
)