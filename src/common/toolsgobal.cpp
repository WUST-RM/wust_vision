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
std::vector<double> armor_dis_log_;
size_t img_recv_count_ = 0;
size_t detect_finish_count_ = 0;
size_t fire_count_ = 0;
std::chrono::steady_clock::time_point last_stat_time_steady_;
double latency_ms;
double debug_show_dt_;
imgframe imgframe_;
std::mutex img_mutex_;
GimbalCmd last_cmd_;
double last_distance;