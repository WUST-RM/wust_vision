// armor_solver.cpp
#include "control/armor_solver.hpp"
<<<<<<< HEAD
#include "common/logger.hpp"
#include "common/gobal.hpp"
=======
#include "common/gobal.hpp"
#include "common/logger.hpp"
>>>>>>> ec64a0b (update nuc)
#include "yaml-cpp/yaml.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
Solver::Solver(const YAML::Node &config) { init(config); }
void Solver::init(const YAML::Node &config) {
  // 1. 加载 YAML

  if (!config["solver"]) {
    throw std::runtime_error("Missing 'solver' node in config");
  }
  auto s = config["solver"];

  // 2. 基本标量参数
<<<<<<< HEAD
  shooting_range_w = s["shooting_range_w"].as<double>(0.135);
  shooting_range_h = s["shooting_range_h"].as<double>(0.135);
=======
  shooting_range_w = s["shooting_range_w"].as<double>(0.12);
  shooting_range_h = s["shooting_range_h"].as<double>(0.12);
>>>>>>> ec64a0b (update nuc)
  max_tracking_v_yaw = s["max_tracking_v_yaw"].as<double>(60.0);
  prediction_delay = s["prediction_delay"].as<double>(0.0);
  controller_delay = s["controller_delay"].as<double>(0.0);
  side_angle = s["side_angle"].as<double>(20.0);
  min_switching_v_yaw = s["min_switching_v_yaw"].as<double>(1.0);

  bullet_speed = s["bullet_speed"].as<double>(25.0);
  gravity = s["gravity"].as<double>(10.0);
  resistance = s["resistance"].as<double>(0.092);
  iteration_times = s["iteration_times"].as<int>(20);

  std::string comp_type = s["compenstator_type"].as<std::string>("ideal");

  // 3. 初始化弹道补偿器
  trajectory_compensator_ = CompensatorFactory::createCompensator(comp_type);
  trajectory_compensator_->iteration_times = iteration_times;
  velocity = bullet_speed;
  trajectory_compensator_->gravity = gravity;
  trajectory_compensator_->resistance = resistance;

  // 4. 手动补偿表（pitch_offset）
  manual_compensator_ = std::make_unique<ManualCompensator>();
<<<<<<< HEAD
  if (s["pitch_offset"]) {
    std::vector<std::string> raw_offsets =
        s["pitch_offset"].as<std::vector<std::string>>();

    if (!manual_compensator_->updateMapFlow(raw_offsets)) {
      WUST_WARN(solver_logger) << "Failed to update manual compensator";
    }
  }
=======
  std::vector<OffsetEntry> entries;

  if (s["pitch_offset"]) {
    for (const auto &node : s["pitch_offset"]) {
      OffsetEntry e;
      e.d_min = node["d_min"].as<double>();
      e.d_max = node["d_max"].as<double>();
      e.h_min = node["h_min"].as<double>();
      e.h_max = node["h_max"].as<double>();
      e.pitch_off = node["pitch_off"].as<double>();
      e.yaw_off = node["yaw_off"].as<double>();
      entries.push_back(e);
    }
  }
  manual_compensator_->updateMapFlow(entries);
>>>>>>> ec64a0b (update nuc)

  // 5. 状态机初值
  state_ = State::TRACKING_ARMOR;
  overflow_count_ = 0;
  transfer_thresh = 5;
}

GimbalCmd Solver::solve(const Target &target,
                        std::chrono::steady_clock::time_point current_time) {
  // 1. 获取最新的云台 RPY
  std::array<double, 3> rpy{};
<<<<<<< HEAD
  Transform tf_gimbal;
  auto now= std::chrono::steady_clock::now();
  if (!tf_tree_.getTransform("gimbal_odom", "gimbal_link",now, tf_gimbal)) {
    throw std::runtime_error("Failed to get gimbal_link transform");
  }
  double r,p,y;
  tf2::Matrix3x3(tf_gimbal.orientation).getRPY(r, p,y);
  // p=p+gimbal2camera_pitch;
  
  // std::cout<<"tf r"<<r/M_PI*180<<"tf p"<<p/M_PI*180<<"tf y"<<y/M_PI*180<<std::endl;

  rpy[0]=last_roll;
  rpy[1]=last_pitch+gimbal2camera_pitch;
  rpy[2]=last_yaw;
  // std::cout<<"roll: "<<rpy[0]/M_PI*180<<"pitch: "<<rpy[1]/M_PI*180<<"yaw: "<<rpy[2]/M_PI*180<<std::endl;
 // std::cout<<"pitch: "<<rpy[1]/M_PI*180<<std::endl;
=======

  rpy[0] = last_roll;
  rpy[1] = last_pitch + gimbal2camera_pitch;
  rpy[2] = last_yaw;

>>>>>>> ec64a0b (update nuc)
  //  2. 预测目标位置与朝向
  Eigen::Vector3d pos(target.position_.x, target.position_.y,
                      target.position_.z);
  double yaw = target.yaw;
<<<<<<< HEAD
  //std::cout<<"yaw: "<<yaw/M_PI*180<<std::endl;
=======

>>>>>>> ec64a0b (update nuc)
  using namespace std::chrono;

  double fly_t = trajectory_compensator_->getFlyingTime(pos);
  auto dt_seconds = duration<double>(fly_t + prediction_delay);
  auto dt = duration_cast<steady_clock::duration>(dt_seconds);
  auto total_dt = (current_time - target.timestamp) + dt;
  double dt_seconds_double = duration<double>(total_dt).count();
  pos += dt_seconds_double * Eigen::Vector3d(target.velocity_.x,
                                             target.velocity_.y,
                                             target.velocity_.z);
  yaw += dt_seconds_double * target.v_yaw;

  // 3. 选装甲板并计算原始 yaw/pitch
  auto armors = getArmorPositions(pos, yaw, target.radius_1, target.radius_2,
                                  target.d_zc, target.d_za, target.armors_num);
  int idx = selectBestArmor(armors, pos, yaw, target.v_yaw, target.armors_num);

  Eigen::Vector3d chosen = armors.at(idx);
  if (chosen.norm() < 0.1) {
    throw std::runtime_error("No valid armor to shoot");
  }
  double raw_yaw, raw_pitch;
  calcYawAndPitch(chosen, rpy, raw_yaw, raw_pitch);
  double distance = chosen.norm();
<<<<<<< HEAD

=======
  std::vector<double> offs;
  double pitch_off;
  double yaw_off;
  double fire_yaw;
  double fire_pitch;
  double raw_yaw_, raw_pitch_;
>>>>>>> ec64a0b (update nuc)
  // 4. 状态机逻辑
  bool fire_advice = false;
  switch (state_) {
  case TRACKING_ARMOR:
    if (std::abs(target.v_yaw) > max_tracking_v_yaw) {
      ++overflow_count_;
    } else {
      overflow_count_ = 0;
    }
    if (overflow_count_ > transfer_thresh) {
      state_ = TRACKING_CENTER;
    }
    // 如果一直没对上，也加 controller_delay 预测
    if (controller_delay != 0.0) {
      pos += controller_delay * Eigen::Vector3d(target.velocity_.x,
                                                target.velocity_.y,
                                                target.velocity_.z);
      yaw += controller_delay * target.v_yaw;
      auto tmp = getArmorPositions(pos, yaw, target.radius_1, target.radius_2,
                                   target.d_zc, target.d_za, target.armors_num)
                     .at(idx);
      if (tmp.norm() < 0.1) {
        throw std::runtime_error("No valid armor after controller delay");
      }
      calcYawAndPitch(tmp, rpy, raw_yaw, raw_pitch);
      distance = tmp.norm();
    }
<<<<<<< HEAD
    fire_advice = isOnTarget(rpy[2], rpy[1], raw_yaw, raw_pitch, distance);
=======
    // calcYawAndPitch(pos, rpy, raw_yaw_, raw_pitch);
    offs = manual_compensator_->angleHardCorrect(distance, chosen.z());
    yaw_off = offs[1] * M_PI / 180.0;
    pitch_off = offs[0] * M_PI / 180.0;
    fire_yaw = raw_yaw + yaw_off;
    fire_pitch = raw_pitch + pitch_off;
    fire_advice = isOnTarget(rpy[2], rpy[1], fire_yaw, fire_pitch, distance);
>>>>>>> ec64a0b (update nuc)
    break;

  case TRACKING_CENTER:
    if (std::abs(target.v_yaw) < max_tracking_v_yaw) {
      ++overflow_count_;
    } else {
      overflow_count_ = 0;
    }
    if (overflow_count_ > transfer_thresh) {
      state_ = TRACKING_ARMOR;
      overflow_count_ = 0;
    }
<<<<<<< HEAD
    fire_advice = true;
    calcYawAndPitch(pos, rpy, raw_yaw, raw_pitch);
    distance = pos.norm();
=======

    calcYawAndPitch(chosen, rpy, raw_yaw_, raw_pitch);
    if (controller_delay != 0.0) {
      pos += controller_delay * Eigen::Vector3d(target.velocity_.x,
                                                target.velocity_.y,
                                                target.velocity_.z);
      yaw += controller_delay * target.v_yaw;
      auto tmp = getArmorPositions(pos, yaw, target.radius_1, target.radius_2,
                                   target.d_zc, target.d_za, target.armors_num)
                     .at(idx);
      if (tmp.norm() < 0.1) {
        throw std::runtime_error("No valid armor after controller delay");
      }
      calcYawAndPitch(tmp, rpy, raw_yaw_, raw_pitch);
      distance = tmp.norm();
    }
    // fire_advice = true;
    calcYawAndPitch(pos, rpy, raw_yaw, raw_pitch_);
    distance = pos.norm();
    offs = manual_compensator_->angleHardCorrect(distance, chosen.z());
    yaw_off = offs[1] * M_PI / 180.0;
    pitch_off = offs[0] * M_PI / 180.0;

    fire_yaw = raw_yaw_ + yaw_off;
    fire_pitch = raw_pitch + pitch_off;
    fire_advice = isOnTarget(rpy[2], rpy[1], fire_yaw, fire_pitch, distance);
>>>>>>> ec64a0b (update nuc)
    break;
  }

  // 5. 弹道+手动补偿
<<<<<<< HEAD
  auto offs = manual_compensator_->angleHardCorrect(distance, chosen.z());
  double pitch_off = offs[0] * M_PI / 180.0;
  double yaw_off = offs[1] * M_PI / 180.0;
=======

>>>>>>> ec64a0b (update nuc)
  double cmd_pitch = raw_pitch + pitch_off;
  double cmd_yaw = normalize_angle(raw_yaw + yaw_off);

  // 6. 填充输出
  GimbalCmd cmd;
  cmd.timestamp = current_time;
  cmd.distance = distance;
  cmd.fire_advice = fire_advice;
  cmd.yaw = cmd_yaw * 180.0 / M_PI;
  cmd.pitch = cmd_pitch * 180.0 / M_PI;
  cmd.yaw_diff = (cmd_yaw - rpy[2]) * 180.0 / M_PI;
  cmd.pitch_diff = (cmd_pitch - rpy[1]) * 180.0 / M_PI;
  cmd.select_id = idx;
  return cmd;
}

std::vector<std::pair<double, double>> Solver::getTrajectory() const noexcept {
  auto traj = trajectory_compensator_->getTrajectory(15, rpy_[1]);
  for (auto &p : traj) {
    double x = p.first, y = p.second;
    p.first = x * std::cos(rpy_[1]) + y * std::sin(rpy_[1]);
    p.second = -x * std::sin(rpy_[1]) + y * std::cos(rpy_[1]);
  }
  return traj;
}

bool Solver::isOnTarget(const double cur_yaw, const double cur_pitch,
                        const double target_yaw, const double target_pitch,
                        const double distance) const noexcept {
  // Judge whether to shoot
  double shooting_range_yaw = std::abs(atan2(shooting_range_w / 2, distance));
  double shooting_range_pitch = std::abs(atan2(shooting_range_h / 2, distance));
  // Limit the shooting area to 1 degree to avoid not shooting when distance is
  // too large
  shooting_range_yaw = std::max(shooting_range_yaw, 1.0 * M_PI / 180);
  shooting_range_pitch = std::max(shooting_range_pitch, 1.0 * M_PI / 180);
  if (std::abs(cur_yaw - target_yaw) < shooting_range_yaw &&
      std::abs(cur_pitch - target_pitch) < shooting_range_pitch) {
    return true;
  }

  return false;
}
std::vector<Eigen::Vector3d>
Solver::getArmorPositions(const Eigen::Vector3d &target_center,
                          const double target_yaw, const double r1,
                          const double r2, const double d_zc, const double d_za,
                          const size_t armors_num) const noexcept {
  auto armor_positions =
      std::vector<Eigen::Vector3d>(armors_num, Eigen::Vector3d::Zero());
  // Calculate the position of each armor
  bool is_current_pair = true;
  double r = 0., target_dz = 0.;
  for (size_t i = 0; i < armors_num; i++) {
    double temp_yaw = target_yaw + i * (2 * M_PI / armors_num);
    if (armors_num == 4) {
      r = is_current_pair ? r1 : r2;
      target_dz = d_zc + (is_current_pair ? 0 : d_za);
      is_current_pair = !is_current_pair;
    } else {
      r = r1;
      target_dz = d_zc;
    }
    armor_positions[i] =
        target_center +
        Eigen::Vector3d(-r * cos(temp_yaw), -r * sin(temp_yaw), target_dz);
  }
  return armor_positions;
}
void Solver::calcYawAndPitch(const Eigen::Vector3d &p,
                             const std::array<double, 3> &rpy, double &yaw,
                             double &pitch) const noexcept {
  // Calculate yaw and pitch
  yaw = atan2(p.y(), p.x());
  pitch = atan2(p.z(), p.head(2).norm());

  if (double temp_pitch = pitch;
      trajectory_compensator_->compensate(p, temp_pitch)) {
    pitch = temp_pitch;
  }
<<<<<<< HEAD
=======
  // std::cout << "yaw: " << yaw << " pitch: " << pitch << std::endl;
>>>>>>> ec64a0b (update nuc)
}
int Solver::selectBestArmor(const std::vector<Eigen::Vector3d> &armor_positions,
                            const Eigen::Vector3d &target_center,
                            const double target_yaw, const double target_v_yaw,
                            const size_t armors_num) const noexcept {
  // Angle between the car's center and the X-axis
  double alpha = std::atan2(target_center.y(), target_center.x());
  // Angle between the front of observed armor and the X-axis
  double beta = target_yaw;

  // clang-format off
<<<<<<< HEAD
Eigen::Matrix2d R_odom2center;
Eigen::Matrix2d R_odom2armor;
R_odom2center << std::cos(alpha), std::sin(alpha), 
-std::sin(alpha), std::cos(alpha);
R_odom2armor << std::cos(beta), std::sin(beta), 
-std::sin(beta), std::cos(beta);
=======
  Eigen::Matrix2d R_odom2center;
  Eigen::Matrix2d R_odom2armor;
  R_odom2center << std::cos(alpha), std::sin(alpha), 
  -std::sin(alpha), std::cos(alpha);
  R_odom2armor << std::cos(beta), std::sin(beta), 
  -std::sin(beta), std::cos(beta);
>>>>>>> ec64a0b (update nuc)
  // clang-format on
  Eigen::Matrix2d R_center2armor = R_odom2center.transpose() * R_odom2armor;

  // Equal to (alpha - beta) in most cases
  double decision_angle = -std::asin(R_center2armor(0, 1));

  // Angle thresh of the armor jump
  double theta = (target_v_yaw > 0 ? side_angle : -side_angle) / 180.0 * M_PI;

  // Avoid the frequent switch between two armor
  if (std::abs(target_v_yaw) < min_switching_v_yaw) {
    theta = 0;
  }

  double temp_angle = decision_angle + M_PI / armors_num - theta;

  if (temp_angle < 0) {
    temp_angle += 2 * M_PI;
  }

  int selected_id = static_cast<int>(temp_angle / (2 * M_PI / armors_num));
  return selected_id;
}