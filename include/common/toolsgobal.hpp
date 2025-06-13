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
extern std::vector<double> armor_dis_log_;
extern size_t img_recv_count_;
extern size_t detect_finish_count_;
extern size_t fire_count_;
extern std::chrono::steady_clock::time_point last_stat_time_steady_;

extern double latency_ms;

extern double debug_show_dt_;

extern GimbalCmd last_cmd_;
extern double last_distance;