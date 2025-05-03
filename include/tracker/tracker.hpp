// Copyright Chen Jun 2023. Licensed under the MIT License.
//
// Additional modifications and features by Chengfu Zou, Labor. Licensed under Apache License 2.0.
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

#ifndef ARMOR_SOLVER_TRACKER_HPP_
#define ARMOR_SOLVER_TRACKER_HPP_

// std
#include <memory>
#include <string>

// third party
#include <Eigen/Eigen>

// project
#include "tracker/extended_kalman_filter.hpp"
#include "type/type.hpp"
#include "tracker/motion_model.hpp"
enum class ArmorsNum { NORMAL_4 = 4, OUTPOST_3 = 3 };

class Tracker {
public:
  Tracker(double max_match_distance, double max_match_yaw);

  void init(const Armors &armors_msg) noexcept;
  void update(const Armors &armors_msg) noexcept;

  enum State {
    LOST,
    DETECTING,
    TRACKING,
    TEMP_LOST,
  } tracker_state;

  std::unique_ptr<RobotStateEKF> ekf;

  int tracking_thres ;  
  int lost_thres ;

  Armor tracked_armor;
  ArmorNumber tracked_id;
  ArmorsNum tracked_armors_num;
  std::string type;
  Eigen::VectorXd measurement;
  Eigen::VectorXd target_state;

  double d_za, another_r;
  double d_zc;

private:
  void initEKF(const Armor &a) noexcept;
  void handleArmorJump(const Armor &a) noexcept;

  double orientationToYaw(const tf2::Quaternion &q) noexcept;
  static Eigen::Vector3d getArmorPositionFromState(const Eigen::VectorXd &x) noexcept;

  double max_match_distance_;
  double max_match_yaw_diff_;

  int detect_count_;
  int lost_count_;

  double last_yaw_;

  std::string tracker_logger = "tracker";
};

#endif  // ARMOR_SOLVER_TRACKER_HPP_
