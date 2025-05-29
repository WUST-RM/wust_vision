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

#ifndef ARMOR_DETECTOR_ARMOR_POSE_ESTIMATOR_HPP_
#define ARMOR_DETECTOR_ARMOR_POSE_ESTIMATOR_HPP_

// std
#include <array>
#include <memory>
#include <vector>
// OpenCV
#include <opencv2/opencv.hpp>
// Eigen
#include <Eigen/Dense>
// ros2
// #include <geometry_msgs/msg/pose.hpp>
// #include <rclcpp/rclcpp.hpp>
// #include <sensor_msgs/msg/camera_info.hpp>
// #include <tf2_ros/buffer.h>
// project
#include "detect/ba_solver.hpp"

#include "detect/pnp_solver.hpp"
#include "type/type.hpp"
#include "yaml-cpp/yaml.h"

class ArmorPoseEstimator {
public:
  explicit ArmorPoseEstimator(const std::string &camera_info_path);

  std::vector<Armor> extractArmorPoses(const std::vector<ArmorObject> &armors,
                                       Eigen::Matrix3d R_imu_camera);

  void enableBA(bool enable) { use_ba_ = enable; }

private:
  // Select the best PnP solution according to the armor's direction in image,
  // only available for SOLVEPNP_IPPE
  void sortPnPResult(const ArmorObject &armor, std::vector<cv::Mat> &rvecs,
                     std::vector<cv::Mat> &tvecs,
                     std::string coord_frame_name) const;

  // Convert a rotation matrix to RPY
  static Eigen::Vector3d rotationMatrixToRPY(const Eigen::Matrix3d &R);

  bool use_ba_;

  Eigen::Matrix3d R_gimbal_camera_;

  std::unique_ptr<BaSolver> ba_solver_;
  std::unique_ptr<PnPSolver> pnp_solver_;
};

#endif // ARMOR_POSE_ESTIMATOR_HPP_