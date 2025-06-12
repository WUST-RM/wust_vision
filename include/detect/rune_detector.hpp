// Copyright 2023 Yunlong Feng
//
// Additional modifications and features by Chengfu Zou, 2024.
//
// Copyright (C) FYT Vision Group. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef RUNE_DETECTOR_RUNE_DETECTOR_HPP_
#define RUNE_DETECTOR_RUNE_DETECTOR_HPP_

// std
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
// third party
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
// project
#include "type/type.hpp"
#include "common/ThreadPool.h"
class RuneDetector {
public:
  using CallbackType = std::function<void(std::vector<RuneObject> &,
                                          std::chrono::steady_clock::time_point,
                                          const cv::Mat &)>;

public:
  // Construct a new OpenVINO Detector object
  explicit RuneDetector(const std::filesystem::path &model_path,
                        const std::string &device_name,
                        float conf_threshold = 0.25, int top_k = 128,
                        float nms_threshold = 0.3, bool auto_init = false);

  void init();

  // Push an inference request to the detector
  void pushInput(const cv::Mat &rgb_img,
                 std::chrono::steady_clock::time_point timestamp);

  void setCallback(CallbackType callback);

  // Detect R tag using traditional method
  // Return the center of the R tag and binary roi image (for debug)
  std::tuple<cv::Point2f, cv::Mat>
  detectRTag(const cv::Mat &img, int binary_thresh, const cv::Point2f &prior);

private:
  // Do inference and call the infer_callback_ after inference
  bool processCallback(const cv::Mat resized_img,
                       Eigen::Matrix3f transform_matrix,
                       std::chrono::steady_clock::time_point timestamp,
                       const cv::Mat &src_img);

private:
  std::string model_path_;
  std::string device_name_;
  float conf_threshold_;
  int top_k_;
  float nms_threshold_;
  std::mutex mtx_;
  std::vector<int> strides_;
  std::vector<GridAndStride> grid_strides_;

  CallbackType infer_callback_;
  std::unique_ptr<ThreadPool> thread_pool_;

  std::unique_ptr<ov::Core> ov_core_;
  std::unique_ptr<ov::CompiledModel> compiled_model_;
};

#endif // RUNE_DETECTOR_RUNE_DETECTOR_HPP_
