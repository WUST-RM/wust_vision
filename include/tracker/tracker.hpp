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
#include <vector>

// project
#include "tracker/extended_kalman_filter.hpp"
#include "type/type.hpp"
#include "tracker/motion_model.hpp"
enum class ArmorsNum { NORMAL_4 = 4, OUTPOST_3 = 3 };
inline double normalizeAngle(double angle) {
  while (angle > M_PI) angle -= 2 * M_PI;
  while (angle < -M_PI) angle += 2 * M_PI;
  return angle;
}

class Tracker {
public:
  Tracker(double max_match_distance, double max_match_yaw );

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
  int retype;
  Eigen::VectorXd measurement;
  Eigen::VectorXd target_state;

  double d_za, another_r;
  double d_zc;
  float yaw_diff_;
  float position_diff_;
  int buffer_size_ = 5;
  float obs_yaw_stationary_thresh = 0.8;   
  float pred_yaw_stationary_thresh = 0.5; 
  float min_valid_velocity = 0.01;
  int max_inconsistent_count_ = 3;
  int rotation_inconsistent_cooldown_limit_ = 5;  

private:
  void initEKF(const Armor &a) noexcept;
  void handleArmorJump(const Armor &a) noexcept;

  double orientationToYaw(const tf2::Quaternion &q) noexcept;
  static Eigen::Vector3d getArmorPositionFromState(const Eigen::VectorXd &x) noexcept;
  void updateBestYawdiff(const Armor &armor1,const Armor &armor2);

  double max_match_distance_;
  double max_match_yaw_diff_;

  int detect_count_;
  int lost_count_;

  double last_yaw_;


  std::chrono::steady_clock::time_point last_track_time_;
  std::deque<float> yaw_velocity_buffer_;
  
  
  int track_update_count_ = 0;
  bool if_have_last_track_ = false;
  double last_track_yaw_;
  
  
  int rotation_inconsistent_count_ = 0;
  
  
  int rotation_inconsistent_cooldown_ = 0;
  
  
  



  std::string tracker_logger = "tracker";
  std::deque<std::chrono::steady_clock::time_point> armor_jump_timestamps_;

};

#endif  // ARMOR_SOLVER_TRACKER_HPP_
