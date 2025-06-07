#include "common/toolsgobal.hpp"

std::vector<std::pair<double, double>> target_yaw_log_;
std::mutex yaw_log_mutex_;
std::chrono::steady_clock::time_point start_time_ =
    std::chrono::steady_clock::now();
std::thread target_yaw_plot_thread_;
std::thread robot_cmd_plot_thread_;
std::mutex robot_cmd_mutex_;
std::vector<double> time_log_;
std::vector<double> cmd_yaw_log_;
std::vector<double> cmd_pitch_log_;
