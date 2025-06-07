#pragma once

#include "common/tf.hpp"
#include "detect/mono_measure_tool.hpp"
#include <thread>

extern std::vector<std::pair<double, double>> target_yaw_log_;
extern std::mutex yaw_log_mutex_;
extern std::chrono::steady_clock::time_point start_time_;
extern std::thread target_yaw_plot_thread_;
extern std::thread robot_cmd_plot_thread_;

extern std::mutex robot_cmd_mutex_;
extern std::vector<double> time_log_;
extern std::vector<double> cmd_yaw_log_;
extern std::vector<double> cmd_pitch_log_;