#include "tracker/tracker.hpp"
#include "common/angles.h"
#include "common/logger.hpp"
#include "type/type.hpp"

// std
#include <cfloat>
#include <iostream>
#include <memory>
#include <string>
#include <algorithm>  // 如果需要转换大小写
#include <fmt/format.h>



Tracker::Tracker(double max_match_distance, double max_match_yaw_diff)
: tracker_state(LOST)
, tracked_id(ArmorNumber::UNKNOWN)  
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
      retype=retypetotracker(armor.number);
      type=armor.type;
    }
  }
  WUST_INFO(tracker_logger)<<"INIT EKF";
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

  if (!armors_msg.armors.empty()) {
    Armor same_id_armor;
    int same_id_armors_count = 0;
    auto predicted_position = getArmorPositionFromState(ekf_prediction);
    double min_position_diff = DBL_MAX;
    double yaw_diff = DBL_MAX;


    for ( auto &armor : armors_msg.armors) 
  {
   
    
      if (retypetotracker(armor.number) == retype) {
        same_id_armor = armor;
        same_id_armors_count++;
        //WUST_INFO(tracker_logger)<<"Same ID armor found!"<<fmt::format("count: {}\n", same_id_armors_count);
        auto p = armor.target_pos;
        Eigen::Vector3d position_vec(p.x, p.y, p.z);
        double position_diff = (predicted_position - position_vec).norm();
       // WUST_INFO(tracker_logger)<<"Armor found!"<<fmt::format("position_diff: {}\n", position_diff);

        if (position_diff < min_position_diff) {
          min_position_diff = position_diff;
          yaw_diff = std::abs(orientationToYaw(armor.target_ori) - ekf_prediction(6));
          tracked_armor = armor;

          
        if (tracked_id == ArmorNumber::OUTPOST) {
            tracked_armors_num = ArmorsNum::OUTPOST_3;
          } else {
            tracked_armors_num = ArmorsNum::NORMAL_4;
          }
        }else {
          another_armors.push_back(armor);
          position_diff_ = position_diff;
        }
      }
    }
    //几何法热加载----fail！！！！！！！
    // if (!another_armors.empty()) {
    //   double min_pose_diff = DBL_MAX;
    //   double yaw_diff_best = 0;
    //   Armor closest_armor;
      
    //   Eigen::Vector3d tracked_pos(tracked_armor.target_pos.x,
    //                               tracked_armor.target_pos.y,
    //                               tracked_armor.target_pos.z);
    //   double tracked_yaw = tracked_armor.yaw;
    
    //   // 可调参数：位置差和朝向差的权重
    //   constexpr double position_weight = 1.0;
    //   constexpr double yaw_weight = 0.5;
    
    //   for (const auto &armor : another_armors) {
    //     Eigen::Vector3d other_pos(armor.target_pos.x,
    //                               armor.target_pos.y,
    //                               armor.target_pos.z);
    //     double position_diff = (tracked_pos - other_pos).norm();
    
    //     double other_yaw = armor.yaw;
    //     double diff = other_yaw - tracked_yaw;
    //     while (diff > M_PI) diff -= 2 * M_PI;
    //     while (diff < -M_PI) diff += 2 * M_PI;
          
    //     double yaw_diff_a = std::abs(diff); 
        
    
    //     // 综合 pose 差
    //     double pose_diff = position_weight * position_diff + yaw_weight * yaw_diff_a;
    
    //     if (pose_diff < min_pose_diff) {
    //       min_pose_diff = pose_diff;
    //       yaw_diff_best  = yaw_diff_a;
    //       closest_armor = armor;
       
    //     }
       
    //   }
    //   if(yaw_diff_best>=1.55&&yaw_diff_best<=1.60 )
    //   {
    //       updateBestYawdiff(closest_armor, tracked_armor);
    //   }
    
    
    

    // }
    

    if (min_position_diff < max_match_distance_ && yaw_diff < max_match_yaw_diff_) {
      matched = true;
      auto p = tracked_armor.target_pos;
      double measured_yaw = orientationToYaw(tracked_armor.target_ori);
      measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);
      target_state = ekf->update(measurement);
    } else if (same_id_armors_count == 1 && yaw_diff > max_match_yaw_diff_) {
      
      handleArmorJump(same_id_armor);
      
      yaw_diff_=yaw_diff;
    }else {
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

  target_state = Eigen::VectorXd::Zero(X_N);
  double r = 0.24;
  double xc = xa + r * cos(yaw);
  double yc = ya + r * sin(yaw);
  double zc = za;
  d_za = 0, d_zc = 0, another_r = r;
  target_state << xc, 0, yc, 0, zc, 0, yaw, 0, r, d_zc;
  ekf->setState(target_state);
}


void Tracker::handleArmorJump(const Armor &current_armor) noexcept {
  // using clock = std::chrono::steady_clock;
  // auto now = clock::now();

  // // 记录跳变时间
  // armor_jump_timestamps_.emplace_back(now);

  // // 清理超过 1 秒的时间点
  // while (!armor_jump_timestamps_.empty() &&
  //        std::chrono::duration_cast<std::chrono::duration<double>>(now - armor_jump_timestamps_.front()).count() > 1.0) {
  //   armor_jump_timestamps_.pop_front();
  // }

  // // 跳变频率
  // double jump_frequency = static_cast<double>(armor_jump_timestamps_.size());
  // WUST_DEBUG(tracker_logger) << fmt::format("Armor Jump Frequency: {:.1f} Hz", jump_frequency);

  // -------- 原始逻辑保持不变 --------
  double last_yaw = target_state(6);
  double yaw = orientationToYaw(current_armor.target_ori);

  if (std::abs(yaw - last_yaw) > 0.3) {
    target_state(6) = yaw;

    if (tracked_armors_num == ArmorsNum::NORMAL_4) {
      d_za = target_state(4) + target_state(9) - current_armor.target_pos.z;
      std::swap(target_state(8), another_r);
      d_zc = d_zc == 0 ? -d_za : 0;
      target_state(9) = d_zc;
    }
    WUST_DEBUG(tracker_logger) << "Armor Jump!";
  }

  Eigen::Vector3d current_p(current_armor.target_pos.x, current_armor.target_pos.y, current_armor.target_pos.z);
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
void Tracker::updateBestYawdiff(const Armor &armor1, const Armor &armor2)
{
  // 位置向量
  Eigen::Vector3d p1(armor1.target_pos.x, armor1.target_pos.y, armor1.target_pos.z);
  Eigen::Vector3d p2(armor2.target_pos.x, armor2.target_pos.y, armor2.target_pos.z);

  // 获取 armor2 的朝向（假设为真值）
  double yaw2 = orientationToYaw(armor2.target_ori);

  // 从 armor2 朝向的 yaw 反向延长一定距离，用 armor1 的位置反推
  // 方向向量：反向 unit vector（向后）
  Eigen::Vector2d yaw_dir(-cos(yaw2), -sin(yaw2));

  // armor1 和 armor2 之间的距离
  Eigen::Vector2d p1_2d = p1.head<2>();
  Eigen::Vector2d p2_2d = p2.head<2>();

  // 估计圆心的位置
  // 做法：从 p2 出发，沿反 yaw2 方向延长一段距离 r，使其接近 p1
  // 解一个一元方程：||p1 - (p2 + r * dir)|| 最小 -> 最佳 r
  double r_opt = (p1_2d - p2_2d).dot(yaw_dir);  // 最佳延长距离（投影）

  Eigen::Vector2d center = p2_2d + r_opt * yaw_dir;

  // 计算两个半径
  double r1 = (p1_2d - center).norm();
  double r2 = (p2_2d - center).norm();

  // 输出调试信息
  WUST_INFO(tracker_logger) << fmt::format("rotation center: ({:.2f}, {:.2f})", center.x(), center.y());
  WUST_INFO(tracker_logger) << fmt::format("radius1: {:.2f}, radius2: {:.2f}", r1, r2);

  // 可选：保存结果
  // rotation_center_ = Eigen::Vector3d(center.x(), center.y(), (p1.z() + p2.z()) / 2.0);
  // radius1_ = r1;
  // radius2_ = r2;
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
