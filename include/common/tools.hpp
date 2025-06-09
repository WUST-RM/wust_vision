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
<<<<<<< HEAD
    const std::optional<GimbalCmd> &gimbal_cmd = std::nullopt);
=======
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
>>>>>>> ec64a0b (update nuc)
