#pragma once

#include "common/tf.hpp"
#include "detect/mono_measure_tool.hpp"

extern TfTree tf_tree_;
extern std::unique_ptr<MonoMeasureTool> measure_tool_;
extern int detect_color_;
extern bool debug_mode_;
extern int debug_w;
extern int debug_h;
extern double controller_delay;
extern double velocity;
extern bool if_manual_reset;
extern int control_rate;
extern double last_roll;
extern double last_pitch;
extern double last_yaw;
extern double gimbal2camera_yaw, gimbal2camera_roll, gimbal2camera_pitch;
extern double odom2gimbal_yaw, odom2gimbal_roll, odom2gimbal_pitch;