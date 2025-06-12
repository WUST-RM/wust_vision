#include "tracker/tracker.hpp"
#include "common/angles.h"
#include "common/gobal.hpp"
#include "common/logger.hpp"
#include "type/type.hpp"

// std
#include <algorithm>
#include <cfloat>
#include <fmt/format.h>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>

Tracker::Tracker(double max_match_distance, double max_match_yaw_diff,
                 double max_match_z_diff)
    : tracker_state(LOST), tracked_id(ArmorNumber::UNKNOWN),
      measurement(Eigen::VectorXd::Zero(4)),
      target_state(Eigen::VectorXd::Zero(9)),
      max_match_distance_(max_match_distance),
      max_match_yaw_diff_(max_match_yaw_diff),
      max_match_z_diff_(max_match_z_diff), detect_count_(0), lost_count_(0),
      last_yaw_(0) {}

void Tracker::init(const Armors &armors_msg) noexcept {
  if (armors_msg.armors.empty())
    return;

  double min_distance = DBL_MAX;
  tracked_armor = armors_msg.armors[0];
  for (const auto &armor : armors_msg.armors) {
    if (armor.distance_to_image_center < min_distance) {
      min_distance = armor.distance_to_image_center;
      tracked_armor = armor;
      retype = retypetotracker(armor.number);
      type = armor.type;
    }
  }
  WUST_INFO(tracker_logger) << "INIT EKF";
  initEKF(tracked_armor);
  tracked_id = tracked_armor.number;
  tracker_state = DETECTING;

  if (tracked_id == ArmorNumber::OUTPOST) {
    tracked_armors_num = ArmorsNum::OUTPOST_3;
  } else {
    tracked_armors_num = ArmorsNum::NORMAL_4;
  }
}

void Tracker::update(const Armors &armors_msg) noexcept {
  Eigen::VectorXd ekf_prediction = ekf->predict();
  bool matched = false;
  target_state = ekf_prediction;
  std::vector<Armor> another_armors;
  if (if_manual_reset) {
    tracker_state = LOST;
    return;
  }

  if (!armors_msg.armors.empty()) {
    Armor same_id_armor;
    int same_id_armors_count = 0;
    auto predicted_position = getArmorPositionFromState(ekf_prediction);
    double min_position_diff = DBL_MAX;
    double min_z_diff = DBL_MAX;
    double yaw_diff = DBL_MAX;

    for (auto &armor : armors_msg.armors) {

      if (retypetotracker(armor.number) == retype) {
        same_id_armor = armor;
        same_id_armors_count++;
        // WUST_INFO(tracker_logger)<<"Same ID armor
        // found!"<<fmt::format("count: {}\n", same_id_armors_count);
        auto p = armor.target_pos;
        Eigen::Vector3d position_vec(p.x, p.y, p.z);
        double position_diff = (predicted_position - position_vec).norm();
        double z_diff = std::abs(armor.target_pos.z - predicted_position.z());
        // WUST_INFO(tracker_logger)<<"Armor
        // found!"<<fmt::format("position_diff: {}\n", position_diff);

        if (position_diff < min_position_diff) {
          min_position_diff = position_diff;
          min_z_diff = z_diff;
          yaw_diff =
              std::abs(orientationToYaw(armor.target_ori) - ekf_prediction(6));
          tracked_armor = armor;
          tracked_armor.timestamp = armors_msg.timestamp;
          yaw_diff_ = yaw_diff;

          if (tracked_id == ArmorNumber::OUTPOST) {
            tracked_armors_num = ArmorsNum::OUTPOST_3;
          } else {
            tracked_armors_num = ArmorsNum::NORMAL_4;
          }
        } else {
          another_armors.push_back(armor);
          position_diff_ = position_diff;
        }
      }
    }

    Armor *closest_armor = nullptr;
    double min_diff = M_PI; // 初始化最小差值为最大可能值（π）

    for (auto &armor : another_armors) {
      double yaw_diff =
          std::fabs(armor.yaw - tracked_armor.yaw); // 计算 yaw 差值
      yaw_diff = std::fmod(yaw_diff, M_PI * 2); // 确保差值在 [0, 2π] 范围内

      // 将差值限制到 [-π, π] 范围以获取最小的角度差
      if (yaw_diff > M_PI) {
        yaw_diff -= M_PI * 2;
      }

      double diff_to_90_deg = std::fabs(yaw_diff - M_PI / 2);

      if (diff_to_90_deg < min_diff) {
        min_diff = diff_to_90_deg;
        closest_armor = &armor;
      }
    }

    if (closest_armor != nullptr && min_diff > -0.09 && min_diff < 0.09) {
      // std::cout << "Tracker armor: "
      //           << tracked_armor.target_pos.x << " "
      //           << tracked_armor.target_pos.y << " "
      //           << tracked_armor.target_pos.z << " yaw "
      //           << tracked_armor.yaw << std::endl;

      // std::cout << "Best armor: "
      //           << closest_armor->target_pos.x << " "
      //           << closest_armor->target_pos.y << " "
      //           << closest_armor->target_pos.z << " yaw (before adjust) "
      //           << closest_armor->yaw << std::endl;

      // 基于 Tracker yaw 构造 yaw2 = yaw1 - 90度（π/2）
      double yaw1 = tracked_armor.yaw;
      double yaw2 = yaw1 - M_PI / 2.0;

      Eigen::Vector3d p1(tracked_armor.target_pos.x, tracked_armor.target_pos.y,
                         tracked_armor.target_pos.z);
      Eigen::Vector3d p2(closest_armor->target_pos.x,
                         closest_armor->target_pos.y,
                         closest_armor->target_pos.z);

      // 反向方向向量
      Eigen::Vector3d dir1(-std::cos(yaw1), -std::sin(yaw1), 0);
      Eigen::Vector3d dir2(-std::cos(yaw2), -std::sin(yaw2), 0);

      // 解最近点
      Eigen::Vector3d delta = p1 - p2;
      double a = dir1.dot(dir1);
      double b = dir1.dot(dir2);
      double c = dir2.dot(dir2);
      double d = dir1.dot(delta);
      double e = dir2.dot(delta);
      double denom = a * c - b * b;

      double s = (b * e - c * d) / denom;
      double t = (a * e - b * d) / denom;

      Eigen::Vector3d point1 = p1 + s * dir1;
      Eigen::Vector3d point2 = p2 + t * dir2;
      Eigen::Vector3d center = 0.5 * (point1 + point2);

      double dist_to_p1 = (center - p1).norm();
      double dist_to_p2 = (center - p2).norm();

      // std::cout << "Estimated center (yaw diff 90°, Tracker is true): "
      //           << center.x() << " " << center.y() << " " << center.z() <<
      //           std::endl;

      // std::cout << "Distance to Tracker armor: " << dist_to_p1 << " m" <<
      // std::endl; std::cout << "Distance to Best armor: " << dist_to_p2 << "
      // m" << std::endl;
    }

    if (min_position_diff < max_match_distance_ &&
        yaw_diff < max_match_yaw_diff_ && min_z_diff < max_match_z_diff_) {
      matched = true;
      auto p = tracked_armor.target_pos;
      double measured_yaw = orientationToYaw(tracked_armor.target_ori);
      measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);
      target_state = ekf->update(measurement);
      if (if_have_last_track_) {
        updateYawStateConsistency(measured_yaw);

      } else {

        if_have_last_track_ = true;
        last_track_yaw_ = measured_yaw;
        last_track_time_ = tracked_armor.timestamp;
        yaw_velocity_buffer_.clear();
        track_update_count_ = 0;
      }

    } else if (same_id_armors_count == 1 && yaw_diff > max_match_yaw_diff_ &&
               min_z_diff < max_match_z_diff_) {

      handleArmorJump(same_id_armor);
      if_have_last_track_ = false;

    } else {
      // WUST_DEBUG(tracker_logger)<<"No matched armor found!";
    }
  }

  // 限制状态变量范围
  if (target_state(8) < 0.12) {
    target_state(8) = 0.12;
    ekf->setState(target_state);
  } else if (target_state(8) > 0.4) {
    target_state(8) = 0.4;
    ekf->setState(target_state);
  }

  // 状态机管理
  if (tracker_state == DETECTING) {
    if (matched) {
      detect_count_++;
      if (detect_count_ > tracking_thres) {
        detect_count_ = 0;
        tracker_state = TRACKING;
      }
    } else {
      detect_count_ = 0;
      tracker_state = LOST;
    }
  } else if (tracker_state == TRACKING) {
    if (!matched) {
      tracker_state = TEMP_LOST;
      lost_count_++;
    }
  } else if (tracker_state == TEMP_LOST) {
    if (!matched) {
      lost_count_++;
      if (lost_count_ > lost_thres) {
        lost_count_ = 0;
        tracker_state = LOST;
      }
    } else {
      tracker_state = TRACKING;
      lost_count_ = 0;
    }
  }
}

void Tracker::initEKF(const Armor &a) noexcept {
  double xa = a.target_pos.x;
  double ya = a.target_pos.y;
  double za = a.target_pos.z;
  last_yaw_ = 0;
  double yaw = orientationToYaw(a.target_ori);

  target_state = Eigen::VectorXd::Zero(armor_motion_model::X_N);
  double r = 0.24;
  double xc = xa + r * cos(yaw);
  double yc = ya + r * sin(yaw);
  double zc = za;
  d_za = 0, d_zc = 0, another_r = r;
  target_state << xc, 0, yc, 0, zc, 0, yaw, 0, r, d_zc;
  ekf->setState(target_state);
}

void Tracker::handleArmorJump(const Armor &current_armor) noexcept {

  double last_yaw = target_state(6);
  double yaw = orientationToYaw(current_armor.target_ori);
  double delta_yaw = normalizeAngle(yaw - last_yaw);

  if (std::abs(delta_yaw) > jump_thresh) {
    target_state(6) = yaw;

    if (tracked_armors_num == ArmorsNum::NORMAL_4) {
      d_za = target_state(4) + target_state(9) - current_armor.target_pos.z;
      std::swap(target_state(8), another_r);
      // std::cout<<d_za<<"c"<<d_zc<<"t4"<<target_state(4)<<"az"<<current_armor.target_pos.z<<std::endl;
      d_zc = d_zc == 0 ? -d_za : 0;

      target_state(9) = d_zc;
    }
    WUST_DEBUG(tracker_logger) << "Armor Jump!";
  }

  Eigen::Vector3d current_p(current_armor.target_pos.x,
                            current_armor.target_pos.y,
                            current_armor.target_pos.z);
  Eigen::Vector3d infer_p = getArmorPositionFromState(target_state);

  if ((current_p - infer_p).norm() > max_match_distance_) {
    d_zc = 0;
    double r = target_state(8);
    target_state(0) = current_armor.target_pos.x + r * cos(yaw);
    target_state(1) = 0;
    target_state(2) = current_armor.target_pos.y + r * sin(yaw);
    target_state(3) = 0;
    target_state(4) = current_armor.target_pos.z;
    target_state(5) = 0;
    target_state(9) = d_zc;
  }

  ekf->setState(target_state);
}

double Tracker::orientationToYaw(const tf2::Quaternion &q) noexcept {
  double roll, pitch, yaw;
  tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
  yaw = last_yaw_ + angles::shortest_angular_distance(last_yaw_, yaw);
  last_yaw_ = yaw;
  return yaw;
}

Eigen::Vector3d
Tracker::getArmorPositionFromState(const Eigen::VectorXd &x) noexcept {
  double xc = x(0), yc = x(2), za = x(4) + x(9);
  double yaw = x(6), r = x(8);
  double xa = xc - r * cos(yaw);
  double ya = yc - r * sin(yaw);
  return Eigen::Vector3d(xa, ya, za);
}
void Tracker::updateYawStateConsistency(double measured_yaw) {
  track_update_count_++;

  if (track_update_count_ >= 10) {
    double dt = std::chrono::duration_cast<std::chrono::duration<double>>(
                    tracked_armor.timestamp - last_track_time_)
                    .count();

    if (dt > 1e-5) {
      double yaw_diff_a = normalizeAngle(measured_yaw - last_track_yaw_);
      float yaw_velocity = yaw_diff_a / dt;

      yaw_velocity_buffer_.push_back(yaw_velocity);
      if (yaw_velocity_buffer_.size() > buffer_size_) {
        yaw_velocity_buffer_.pop_front();
      }

      float yaw_velocity_avg =
          std::accumulate(yaw_velocity_buffer_.begin(),
                          yaw_velocity_buffer_.end(), 0.0f) /
          yaw_velocity_buffer_.size();

      float v_yaw_target = target_state(7);

      auto getRotationState = [](float v, float stationary_thresh,
                                 float min_valid) {
        if (std::abs(v) < stationary_thresh)
          return 0; // 静止
        else if (v > min_valid)
          return 1; // 正转
        else if (v < -min_valid)
          return -1; // 反转
        else
          return 0;
      };

      int obs_state = getRotationState(
          yaw_velocity_avg, obs_yaw_stationary_thresh, min_valid_velocity);
      int pred_state = getRotationState(
          v_yaw_target, pred_yaw_stationary_thresh, min_valid_velocity);

      if (rotation_inconsistent_cooldown_ == 0) {
        if (obs_state != pred_state) {
          rotation_inconsistent_count_++;
          if (rotation_inconsistent_count_ >= max_inconsistent_count_) {
            WUST_WARN(tracker_logger)
                << "yaw rotation mismatch: OBS-PRED change ";
            // tracker_state = LOST;
            target_state(7) = yaw_velocity_avg;
            ekf->setState(target_state);
            rotation_inconsistent_count_ = 0;
            rotation_inconsistent_cooldown_ =
                rotation_inconsistent_cooldown_limit_;
          }
        } else {
          rotation_inconsistent_count_ = 0;
        }
      } else {
        rotation_inconsistent_cooldown_--;
      }
    }

    last_track_yaw_ = measured_yaw;
    last_track_time_ = tracked_armor.timestamp;
    track_update_count_ = 0;
  }
}
