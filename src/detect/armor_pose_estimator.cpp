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

#include "detect/armor_pose_estimator.hpp"

#include "common/gobal.hpp"
#include "common/logger.hpp"
#include "common/utils.hpp"
#include "type/type.hpp"
#include "yaml-cpp/yaml.h"
#include <iostream>

ArmorPoseEstimator::ArmorPoseEstimator(const std::string &camera_info_path,
                                       Eigen::Matrix3d R_gimbal_camera) {
  YAML::Node config = YAML::LoadFile(camera_info_path);

  std::array<double, 9> camera_k =
      config["camera_matrix"]["data"].as<std::array<double, 9>>();
  std::vector<double> camera_d =
      config["distortion_coefficients"]["data"].as<std::vector<double>>();
  pnp_solver_ = std::make_unique<PnPSolver>(camera_k, camera_d);
  pnp_solver_->setObjectPoints(
      "small", ArmorObject::buildObjectPoints<cv::Point3f>(SMALL_ARMOR_WIDTH,
                                                           SMALL_ARMOR_HEIGHT));
  pnp_solver_->setObjectPoints(
      "large", ArmorObject::buildObjectPoints<cv::Point3f>(LARGE_ARMOR_WIDTH,
                                                           LARGE_ARMOR_HEIGHT));
  ba_solver_ = std::make_unique<BaSolver>(camera_k, camera_d);

  R_gimbal_camera_ = Eigen::Matrix3d::Identity();
  // R_gimbal_camera_ << 0, 1, 0, 0, 0, -1, -1, 0, 0;
  // R_gimbal_camera_ << 0, 0, -1, 1, 0, 0, 0, -1, 0;
  // R_gimbal_camera_ << R_gimbal_camera;
  R_gimbal_camera_ << 0, 0, 1, -1, 0, 0, 0, -1, 0;
}

std::vector<Armor>
ArmorPoseEstimator::extractArmorPoses(const std::vector<ArmorObject> &armors,
                                      Eigen::Matrix3d R_imu_camera) {
  std::vector<Armor> armors_msg;

  for (const auto &armor : armors) {
    if (!armor.is_ok) {
      continue;
    }
    if (detect_color_ == 0 && armor.color != ArmorColor::RED) {
      continue;
    } else if (detect_color_ == 1 && armor.color != ArmorColor::BLUE) {
      continue;
    }
    std::vector<cv::Mat> rvecs, tvecs;
    Armor armor_;
    ArmorObject temp_armor = armor;
    std::string temp_type;
    int temp_number;
    if (temp_armor.number == ArmorNumber::NO1 ||
        temp_armor.number == ArmorNumber::BASE) {
      temp_type = "large";
      temp_number = 0;
    } else {
      temp_type = "small";
      temp_number = 1;
    }

    // Use PnP to get the initial pose information
    if (pnp_solver_->solvePnPGeneric(armor.landmarks(), rvecs, tvecs,
                                     (temp_type))) {

      sortPnPResult(armor, rvecs, tvecs, temp_type);

      cv::Mat rmat;
      cv::Rodrigues(rvecs[0], rmat);

      Eigen::Matrix3d R = utils::cvToEigen(rmat);
      Eigen::Vector3d t = utils::cvToEigen(tvecs[0]);

      double armor_roll =
          rotationMatrixToRPY(R_gimbal_camera_ * R)[0] * 180 / M_PI;
      Eigen::Quaterniond q1(R);

      if (armor_roll < 105) {

        // Use BA alogorithm to optimize the pose from PnP
        // solveBa() will modify the rotation_matrix
        R = ba_solver_->solveBa(armor, t, R, R_imu_camera, temp_number);
      }
      Eigen::Quaterniond q(R);
      // 定义绕 X 轴的额外旋转（90 度）
      double roll_angle = M_PI / 2; // 弧度
      // double yaw_angle = M_PI / 2;

      Eigen::Quaterniond additional_roll(
          Eigen::AngleAxisd(roll_angle, Eigen::Vector3d::UnitX()));
      // Eigen::Quaterniond additional_yaw(Eigen::AngleAxisd(yaw_angle,
      // Eigen::Vector3d::UnitY()));

      // 应用额外旋转
      Eigen::Quaterniond new_q = q * additional_roll;

      // 转换回旋转矩阵
      Eigen::Matrix3d new_R = new_q.toRotationMatrix();

      // Fill the armor message

      // Fill basic info
      armor_.type = temp_type;
      armor_.number = armor.number;

      // Fill pose
      armor_.pos.x = t(0);
      armor_.pos.y = t(1);
      armor_.pos.z = t(2);
      armor_.ori.x = new_q.x();
      armor_.ori.y = new_q.y();
      armor_.ori.z = new_q.z();
      armor_.ori.w = new_q.w();
      armor_.distance_to_image_center =
          pnp_solver_->calculateDistanceToCenter(armor.center);

      // std::cout << "Roll: " << rpy.x() * 180 / M_PI << " degrees" <<
      // std::endl; std::cout << "Pitch: " << rpy.y() * 180 / M_PI << " degrees"
      // << std::endl; std::cout << "Yaw: " << rpy.z() * 180 / M_PI << "
      // degrees" << std::endl;

      armors_msg.push_back(std::move(armor_));
    } else {
      WUST_WARN("PNP") << "PNP failed";
    }
  }

  return armors_msg;
}

Eigen::Vector3d
ArmorPoseEstimator::rotationMatrixToRPY(const Eigen::Matrix3d &R) {
  // Transform to camera frame
  Eigen::Quaterniond q(R);
  // Get armor yaw
  tf2::Quaternion tf_q(q.x(), q.y(), q.z(), q.w());
  Eigen::Vector3d rpy;
  tf2::Matrix3x3(tf_q).getRPY(rpy[0], rpy[1], rpy[2]);
  return rpy;
}

void ArmorPoseEstimator::sortPnPResult(const ArmorObject &armor,
                                       std::vector<cv::Mat> &rvecs,
                                       std::vector<cv::Mat> &tvecs,
                                       std::string coord_frame_name) const {
  constexpr float PROJECT_ERR_THRES = 3.0;

  // 获取这两个解
  cv::Mat &rvec1 = rvecs.at(0);
  cv::Mat &tvec1 = tvecs.at(0);
  cv::Mat &rvec2 = rvecs.at(1);
  cv::Mat &tvec2 = tvecs.at(1);

  // 将旋转向量转换为旋转矩阵
  cv::Mat R1_cv, R2_cv;
  cv::Rodrigues(rvec1, R1_cv);
  cv::Rodrigues(rvec2, R2_cv);

  // 转换为Eigen矩阵
  Eigen::Matrix3d R1 = utils::cvToEigen(R1_cv);
  Eigen::Matrix3d R2 = utils::cvToEigen(R2_cv);

  // 计算云台系下装甲板的RPY角
  auto rpy1 = rotationMatrixToRPY(R_gimbal_camera_ * R1);
  auto rpy2 = rotationMatrixToRPY(R_gimbal_camera_ * R2);

  double error1 = pnp_solver_->calculateReprojectionError(
      armor.landmarks(), rvec1, tvec1, coord_frame_name);
  double error2 = pnp_solver_->calculateReprojectionError(
      armor.landmarks(), rvec2, tvec2, coord_frame_name);

  // 两个解的重投影误差差距较大或者roll角度较大时，不做选择
  if ((error2 / error1 > PROJECT_ERR_THRES) || (rpy1[0] > 10 * 180 / M_PI) ||
      (rpy2[0] > 10 * 180 / M_PI)) {
    return;
  }

  // 计算灯条在图像中的倾斜角度
  double l_angle =
      std::atan2(armor.lights[0].axis.y, armor.lights[0].axis.x) * 180 / M_PI;
  double r_angle =
      std::atan2(armor.lights[1].axis.y, armor.lights[1].axis.x) * 180 / M_PI;
  double angle = (l_angle + r_angle) / 2;

  angle += 90.0;

  if (armor.number == ArmorNumber::OUTPOST)
    angle = -angle;
  // double aa=rpy1[2]/M_PI*180;
  // double bb=rpy2[2]/M_PI*180;

  //  std::cout<<aa<<"  "<<bb<<"   "<<angle<<std::endl;

  // 根据倾斜角度选择解
  // 如果装甲板左倾（angle > 0），选择Yaw为负的解
  // 如果装甲板右倾（angle < 0），选择Yaw为正的解
  if ((angle > 0 && rpy1[2] > 0 && rpy2[2] < 0) ||
      (angle < 0 && rpy1[2] < 0 && rpy2[2] > 0)) {
    std::swap(rvec1, rvec2);
    std::swap(tvec1, tvec2);

    // std::cout<<"armor_detector"<<"PnP Solution 2 Selected"<<std::endl;
  }
}
