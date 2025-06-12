// Maintained by Chengfu Zou, Labor
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

#ifndef RUNE_SOLVER_RUNE_SOLVER_HPP_
#define RUNE_SOLVER_RUNE_SOLVER_HPP_

// std
#include <algorithm>
#include <array>
#include <deque>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

// project

#include "common/logger.hpp"
#include "control/manual_compensator.hpp"
#include "control/trajectory_compensator.hpp"
#include "detect/pnp_solver.hpp"
#include "tracker/curve_fitter.hpp"
#include "tracker/extended_kalman_filter.hpp"
#include "tracker/motion_modelr.hpp"
#include "type/type.hpp"

// Usage:
//   1. init(msg), if tracker_state == LOST
//   2. update(msg), if tracker_state == DETECTING or TRACKING
//   3. p = predictTarget(timestamp), to get the predicted position
//   4. cmd = solveGimbalCmd(p), to get the gimbal command
class RuneSolver {
public:
  struct RuneSolverParams {
    std::string compensator_type;
    double gravity;
    double bullet_speed;
    double angle_offset_thres;
    double lost_time_thres;
    bool auto_type_determined;
  };

  enum State {
    LOST,
    DETECTING,
    TRACKING,
  } tracker_state;

  RuneSolver(const RuneSolverParams &sr_params);

  // Return: initial angle
  double init(const Rune received_target);

  // Return: normalized angle
  double update(const Rune receive_target);

  // Return: normalized predicted angle
  double predictTarget(Eigen::Vector3d &predicted_position, double timestamp);

  // Return: transormation matrix from rune to odom
  // Throws: tf2::TransformException or std::runtime_error
  Eigen::Matrix4d solvePose(const Rune &target);

  GimbalCmd solveGimbalCmd(const Eigen::Vector3d &target);

  // Return: 3d position of R tag
  Eigen::Vector3d getCenterPosition() const;

  // Param: angle_diff: how much the angle target should prerotate, 0 for no
  // prediction Return: 3d position of target to be aimed at
  Eigen::Vector3d getTargetPosition(double angle_diff) const;

  double getCurAngle() const;

  GimbalCmd solve();

  // Solvers
  std::unique_ptr<PnPSolver> pnp_solver;
  std::unique_ptr<TrajectoryCompensator> trajectory_compensator;
  std::unique_ptr<CurveFitter> curve_fitter;
  std::unique_ptr<rune_motion_model::RuneCenterEKF> ekf;
  std::unique_ptr<ManualCompensator> manual_compensator;

  RuneSolverParams rune_solver_params;
  double predict_offset_;

private:
  double getNormalAngle(const Rune received_target);

  double getObservedAngle(double normal_angle);

  // Return the centroid of the input armor points
  cv::Point2f getCenterPoint(
      const std::array<cv::Point2f, ARMOR_KEYPOINTS_NUM> &armor_points);

  // Return ekf state
  Eigen::Vector4d getStateFromTransform(const Eigen::Matrix4d &transform) const;

  // Observation data

  // last_observed_angle_ is continuously increasing (or decreasing)
  // from the first detection (call init()) of the target without
  // any abrupt change in between.
  double last_observed_angle_;

  // last_angle_ would change (N * DEG_72) when the target jumps
  double last_angle_;
  double start_time_;
  double last_time_;

  Eigen::Vector4d ekf_state_;
};

#endif // RUNE_SOLVER_SOLVER_HPP_
