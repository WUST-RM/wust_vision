#pragma once
#include "common/tf.hpp"
#include "detect/mono_measure_tool.hpp"
#include "driver/packet_typedef.hpp"
#include "opencv2/opencv.hpp"
#include "tracker/tracker.hpp"
#include "type/type.hpp"
#include <opencv2/core/mat.hpp>
#include <optional>
void drawresult(const cv::Mat &src_img, const std::vector<ArmorObject> &objs,
                int64_t timestamp_nanosec);
void drawresult(const imgframe &src_img, const Armors &armors);
void drawreprojec(const cv::Mat &src_img,
                  const std::vector<std::vector<cv::Point2f>> all_pts,
                  const Target target, const Tracker::State state);
void drawreprojec(const cv::Mat &src_img, const Target_info target_info,
                  const Target target, const Tracker::State state);
void drawreprojec(const imgframe &src_img, const Target_info target_info,
                  const Target target, const Tracker::State state);
void drawreprojec(const imgframe &src_img, const Target_info target_info,
                  const Target target, const Tracker::State state,
                  GimbalCmd gimbal_cmd);
// cv::Mat drawreprojec(const imgframe &src_img, const Target_info
// target_info,const Target target,const Tracker::State state);
void dumpTargetToFile(const Target &target,
                      const std::string &path = "/tmp/target_status.txt");
void drawGimbalDirection(cv::Mat &debug_img, const GimbalCmd &gimbal_cmd);
void updatePlot(const std::vector<Armor> &armors);
void dumpImuToFile(const ReceiveImuData &imu,
                   const std::string &path = "/tmp/imu_status.txt");
void dumpAimToFile(const ReceiveAimINFO &aim,
                   const std::string &path = "/tmp/aim_status.txt");
std::string formatAimInfo(const ReceiveAimINFO &aim);
std::string formatImuInfo(const ReceiveImuData &imu);
void draw_debug_overlay(
    const imgframe &src_img, const Armors *armors = nullptr,
    const Target_info *target_info = nullptr, const Target *target = nullptr,
    const std::optional<Tracker::State> &state = std::nullopt,
    const std::optional<GimbalCmd> &gimbal_cmd = std::nullopt);
cv::Mat draw_debug_overlayMat(const imgframe &src_img, const Armors *armors,
                              const Target_info *target_info,
                              const Target *target,
                              const std::optional<Tracker::State> &state,
                              const std::optional<GimbalCmd> &gimbal_cmd);
void draw_debug_overlaywrite(const imgframe &src_img, const Armors *armors,
                             const Target_info *target_info,
                             const Target *target,
                             const std::optional<Tracker::State> &state,
                             const std::optional<GimbalCmd> &gimbal_cmd);
void write_target_log_to_json(const Target &target);
void write_aim_log_to_json(const ReceiveAimINFO &aim);
void drawRune(cv::Mat &src_img, const std::vector<RuneObject> &objs,
              std::chrono::steady_clock::time_point timestamp);
void drawRuneandpre(cv::Mat &src_img, const std::vector<RuneObject> &objs,
                    std::chrono::steady_clock::time_point timestamp,
                    double predict_angle);
void drawRuneandprewrite(cv::Mat &src_img, const std::vector<RuneObject> &objs,
                         std::chrono::steady_clock::time_point timestamp,
                         double predict_angle);
std::string GetUniqueVideoFilename(const std::string &folder,
                                   const std::string &prefix = "output");
cv::Point2f normalize(const cv::Point2f &v);