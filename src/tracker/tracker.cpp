#include "tracker/tracker.hpp"
#include "common/angles.h"

// std
#include <cfloat>
#include <memory>
#include <string>
#include <algorithm>  // 如果需要转换大小写
#include <fmt/format.h>



Tracker::Tracker(double max_match_distance, double max_match_yaw_diff)
: tracker_state(LOST)
, tracked_id()  
, measurement(Eigen::VectorXd::Zero(4))
, target_state(Eigen::VectorXd::Zero(9))
, max_match_distance_(max_match_distance)
, max_match_yaw_diff_(max_match_yaw_diff)
, detect_count_(0)
, lost_count_(0)
, last_yaw_(0) {}

void Tracker::init(const Armors &armors_msg) noexcept {
  if (armors_msg.armors.empty()) return;

  double min_distance = DBL_MAX;
  tracked_armor = armors_msg.armors[0];
  for (const auto &armor : armors_msg.armors) {
    if (armor.distance_to_image_center < min_distance) {
      min_distance = armor.distance_to_image_center;
      tracked_armor = armor;
    }
  }

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

  if (!armors_msg.armors.empty()) {
    Armor same_id_armor;
    int same_id_armors_count = 0;
    auto predicted_position = getArmorPositionFromState(ekf_prediction);
    double min_position_diff = DBL_MAX;
    double yaw_diff = DBL_MAX;

    for (const auto &armor : armors_msg.armors) {
      if (armor.number == tracked_id) {
        same_id_armor = armor;
        same_id_armors_count++;
        auto p = armor.pos;
        Eigen::Vector3d position_vec(p.x, p.y, p.z);
        double position_diff = (predicted_position - position_vec).norm();

        if (position_diff < min_position_diff) {
          min_position_diff = position_diff;
          yaw_diff = std::abs(orientationToYaw(armor.ori) - ekf_prediction(6));
          tracked_armor = armor;

          
        if (tracked_id == ArmorNumber::OUTPOST) {
            tracked_armors_num = ArmorsNum::OUTPOST_3;
          } else {
            tracked_armors_num = ArmorsNum::NORMAL_4;
          }
        }
      }
    }

    if (min_position_diff < max_match_distance_ && yaw_diff < max_match_yaw_diff_) {
      matched = true;
      auto p = tracked_armor.pos;
      double measured_yaw = orientationToYaw(tracked_armor.ori);
      measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);
      target_state = ekf->update(measurement);
    } else if (same_id_armors_count == 1 && yaw_diff > max_match_yaw_diff_) {
      handleArmorJump(same_id_armor);
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
  double xa = a.pos.x;
  double ya = a.pos.y;
  double za = a.pos.z;
  last_yaw_ = 0;
  double yaw = orientationToYaw(a.ori);

  target_state = Eigen::VectorXd::Zero(X_N);
  double r = 0.26;
  double xc = xa + r * cos(yaw);
  double yc = ya + r * sin(yaw);
  double zc = za;
  d_za = 0, d_zc = 0, another_r = r;
  target_state << xc, 0, yc, 0, zc, 0, yaw, 0, r, d_zc;
  ekf->setState(target_state);
}

void Tracker::handleArmorJump(const Armor &current_armor) noexcept {
  double last_yaw = target_state(6);
  double yaw = orientationToYaw(current_armor.ori);

  if (std::abs(yaw - last_yaw) > 0.4) {
    target_state(6) = yaw;

    if (tracked_armors_num == ArmorsNum::NORMAL_4) {
      d_za = target_state(4) + target_state(9) - current_armor.pos.z;
      std::swap(target_state(8), another_r);
      d_zc = d_zc == 0 ? -d_za : 0;
      target_state(9) = d_zc;
    }
  }

  Eigen::Vector3d current_p(current_armor.pos.x, current_armor.pos.y, current_armor.pos.z);
  Eigen::Vector3d infer_p = getArmorPositionFromState(target_state);

  if ((current_p - infer_p).norm() > max_match_distance_) {
    d_zc = 0;
    double r = target_state(8);
    target_state(0) = current_armor.pos.x + r * cos(yaw);
    target_state(1) = 0;
    target_state(2) = current_armor.pos.y + r * sin(yaw);
    target_state(3) = 0;
    target_state(4) = current_armor.pos.z;
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

Eigen::Vector3d Tracker::getArmorPositionFromState(const Eigen::VectorXd &x) noexcept {
  double xc = x(0), yc = x(2), za = x(4) + x(9);
  double yaw = x(6), r = x(8);
  double xa = xc - r * cos(yaw);
  double ya = yc - r * sin(yaw);
  return Eigen::Vector3d(xa, ya, za);
}
