#include "common/tools.hpp"
#include "common/gobal.hpp"
#include "common/matplotlibcpp.h"
#include "common/tf.hpp"
#include "detect/mono_measure_tool.hpp"
#include "fmt/format.h"
#include "tracker/tracker.hpp"
#include "type/type.hpp"
#include <chrono>
#include <cstddef>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>
void drawGimbalDirection(cv::Mat &debug_img, const GimbalCmd &gimbal_cmd) {
  // 1. 云台坐标系下的方向向量（右手系：Z 前，X 右，Y 下）
  Eigen::Vector3f dir_gimbal;
  dir_gimbal << std::cos(gimbal_cmd.pitch) * std::sin(gimbal_cmd.yaw), // X
      -std::sin(gimbal_cmd.pitch),                                     // Y
      std::cos(gimbal_cmd.pitch) * std::cos(gimbal_cmd.yaw);           // Z

  // 2. 云台坐标系 → 相机坐标系变换矩阵
  // 例：Z前Y下X右 (gimbal) → 相机 (X右Y下Z前)，可能需要根据具体坐标系调整
  Eigen::Matrix3f R;
  R << 0, 0, 1, // gimbal Z → camera X
      -1, 0, 0, // gimbal X → camera Y
      0, -1, 0; // gimbal Y → camera Z

  Eigen::Vector3f dir_cam = R * dir_gimbal;

  // 3. 投影到图像平面
  if (dir_cam.z() <= 0.01f)
    return; // 防止除零或背向相机

  float fx =
      static_cast<float>(measure_tool_->camera_intrinsic_.at<double>(0, 0));
  float fy =
      static_cast<float>(measure_tool_->camera_intrinsic_.at<double>(1, 1));
  float cx =
      static_cast<float>(measure_tool_->camera_intrinsic_.at<double>(0, 2));
  float cy =
      static_cast<float>(measure_tool_->camera_intrinsic_.at<double>(1, 2));

  float u = fx * (dir_cam.x() / dir_cam.z()) + cx;
  float v = fy * (dir_cam.y() / dir_cam.z()) + cy;

  // 4. 绘制圆点（限制在图像边界内）
  if (u >= 0 && u < debug_img.cols && v >= 0 && v < debug_img.rows) {
    cv::Point2f pt(u, v);
    cv::circle(debug_img, pt, 8, cv::Scalar(0, 255, 0), -1);
    cv::putText(debug_img, "Gimbal", pt + cv::Point2f(10, -10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
  }
}

void drawresult(const cv::Mat &src_img, const std::vector<ArmorObject> &objs,
                int64_t timestamp_nanosec) {
  static auto last_show_time = std::chrono::steady_clock::now();
  static bool window_initialized = false;
  static int brightness_slider = 200;
  if (src_img.empty()) {
    return;
  }
  if (!window_initialized) {
    cv::namedWindow("debug_armorA", cv::WINDOW_NORMAL);
    cv::resizeWindow("debug_armorA", debug_w, debug_h);
    cv::createTrackbar("Brightness", "debug_armorA", &brightness_slider, 400);
    window_initialized = true;
  }

  auto now = std::chrono::steady_clock::now();
  constexpr double min_interval_ms = 1000.0 / 60.0;

  double elapsed_ms =
      std::chrono::duration<double, std::milli>(now - last_show_time).count();
  if (elapsed_ms < min_interval_ms) {
    return;
  }
  last_show_time = now;

  // 调整亮度
  cv::Mat debug_img;
  double brightness_factor = brightness_slider / 100.0;
  src_img.convertTo(debug_img, -1, brightness_factor, 0);

  cv::cvtColor(debug_img, debug_img, cv::COLOR_BGR2RGB);

  static const int next_indices[] = {2, 0, 3, 1};
  for (auto &obj : objs) {
    for (size_t i = 0; i < 4; ++i) {
      cv::line(debug_img, obj.pts[i], obj.pts[(i + 1) % 4],
               cv::Scalar(48, 48, 255), 1);
      if (obj.is_ok) {
        cv::line(debug_img, obj.pts_binary[i], obj.pts_binary[next_indices[i]],
                 cv::Scalar(0, 255, 0), 1);
        cv::putText(debug_img, fmt::format("{}", i),
                    cv::Point2i(obj.pts_binary[i]), cv::FONT_HERSHEY_SIMPLEX,
                    0.8, cv::Scalar(255, 255, 0), 2);
      }
    }

    std::string armor_color;
    switch (obj.color) {
    case ArmorColor::BLUE:
      armor_color = "B";
      break;
    case ArmorColor::RED:
      armor_color = "R";
      break;
    case ArmorColor::NONE:
      armor_color = "N";
      break;
    case ArmorColor::PURPLE:
      armor_color = "P";
      break;
    default:
      armor_color = "UNKNOWN";
      break;
    }

    std::string armor_key =
        fmt::format("{} {}", armor_color, static_cast<int>(obj.number));
    cv::putText(debug_img, armor_key, cv::Point2i(obj.pts[0]),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
  }
  cv::circle(debug_img, cv::Point2i(1440 / 2., 1080 / 2.), 5,
             cv::Scalar(0, 0, 255), 1);

  auto timestamp_tp = std::chrono::steady_clock::time_point() +
                      std::chrono::nanoseconds(timestamp_nanosec);
  auto latency_nano =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now - timestamp_tp)
          .count();
  double latency_ms = static_cast<double>(latency_nano) / 1e6;

  std::string latency = fmt::format("Latency: {:.3f}ms", latency_ms);
  cv::putText(debug_img, latency, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
              0.8, cv::Scalar(255, 255, 255), 2);

  cv::imshow("debug_armorA", debug_img);
  cv::waitKey(1);
}
void drawresult(const imgframe &src_img, const Armors &armors) {
  static auto last_show_time = std::chrono::steady_clock::now();
  static bool window_initialized = false;
  static int brightness_slider = 200;
  static float yaw_diff;

  if (src_img.img.empty())
    return;

  if (!window_initialized) {
    cv::namedWindow("debug_armor", cv::WINDOW_NORMAL);
    cv::resizeWindow("debug_armor", debug_w, debug_h);
    cv::createTrackbar("Brightness", "debug_armor", &brightness_slider, 400);
    window_initialized = true;
  }

  auto now = std::chrono::steady_clock::now();
  constexpr double min_interval_ms = 1000.0 / 60.0;

  double elapsed_ms =
      std::chrono::duration<double, std::milli>(now - last_show_time).count();
  if (elapsed_ms < min_interval_ms) {
    return;
  }
  last_show_time = now;

  // 调整亮度
  cv::Mat debug_img;

  double brightness_factor = brightness_slider / 100.0;
  src_img.img.convertTo(debug_img, -1, brightness_factor, 0);

  cv::cvtColor(debug_img, debug_img, cv::COLOR_BGR2RGB);

  static const int next_indices[] = {2, 0, 3, 1};
  for (auto &armor : armors.armors) {
    // std::cout<<"cc\n";
    std::vector<cv::Point2f> pts;

    if (!measure_tool_->reprojectArmorCorners_raw(armor, pts))
      continue;
    for (size_t i = 0; i < 4; ++i) {
      cv::line(debug_img, pts[i], pts[(i + 1) % 4], cv::Scalar(255, 100, 0), 2);
    }
    std::string yaw_info = fmt::format("Yaw: {:.3f}", armor.yaw / M_PI * 180);
    cv::putText(debug_img, yaw_info, cv::Point(pts[0].x, pts[0].y - 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 200, 200), 2);
    std::string armor_info;
    switch (armor.number) {
    case ArmorNumber::SENTRY:
      armor_info = "SENTRY";
      break;
    case ArmorNumber::BASE:
      armor_info = "BASE";
      break;
    case ArmorNumber::OUTPOST:
      armor_info = "OUTPOST";
      break;
    case ArmorNumber::NO1:
      armor_info = "NO1";
      break;
    case ArmorNumber::NO2:
      armor_info = "NO2";
      break;
    case ArmorNumber::NO3:
      armor_info = "NO3";
      break;
    case ArmorNumber::NO4:
      armor_info = "NO4";
      break;
    case ArmorNumber::NO5:
      armor_info = "NO5";
      break;
    case ArmorNumber::UNKNOWN:
      armor_info = "UNKNOWN";
      break;
    }
    cv::putText(debug_img, armor_info, cv::Point(pts[1].x, pts[1].y + 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 200, 200), 2);
  }

  if (armors.armors.size() == 2) {
    double yaw1 = armors.armors[0].yaw;
    double yaw2 = armors.armors[1].yaw;

    // 计算周期性最小差值，结果范围在 [-π, π]
    double diff = yaw1 - yaw2;
    while (diff > M_PI)
      diff -= 2 * M_PI;
    while (diff < -M_PI)
      diff += 2 * M_PI;

    yaw_diff = std::abs(diff); // 始终是非负、最小差值

  } else {
    // yaw_diff = 0;
  }
  std::string yaw_diff_info =
      "Yaw_diff: " + std::to_string(yaw_diff / M_PI * 180);
  cv::putText(debug_img, yaw_diff_info, cv::Point(100, 100),
              cv::FONT_HERSHEY_SIMPLEX, 2.7, cv::Scalar(40, 255, 40), 2);
  cv::circle(debug_img, cv::Point2i(1440 / 2., 1080 / 2.), 5,
             cv::Scalar(0, 0, 255), 1);

  auto latency_nano = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          now - armors.timestamp)
                          .count();
  double latency_ms = static_cast<double>(latency_nano) / 1e6;

  std::string latency = fmt::format("Latency: {:.3f}ms", latency_ms);
  cv::putText(debug_img, latency, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
              0.8, cv::Scalar(255, 255, 255), 2);

  cv::imshow("debug_armor", debug_img);
  cv::waitKey(1);
}

void drawreprojec(const cv::Mat &src_img,
                  const std::vector<std::vector<cv::Point2f>> all_pts,
                  const Target target, const Tracker::State state) {
  static auto last_show_time = std::chrono::steady_clock::now();
  static bool window_initialized = false;
  static int brightness_slider = 200;
  static cv::Mat debug_img;
  if (src_img.empty()) {
    return;
  }

  if (!window_initialized) {
    cv::namedWindow("debug_target", cv::WINDOW_NORMAL);
    cv::resizeWindow("debug_target", debug_w, debug_h);
    cv::createTrackbar("Brightness", "debug_target", &brightness_slider, 400);
    window_initialized = true;
  }
  auto now = std::chrono::steady_clock::now();
  constexpr double min_interval_ms = 1000.0 / 60.0;

  double elapsed_ms =
      std::chrono::duration<double, std::milli>(now - last_show_time).count();
  if (elapsed_ms < min_interval_ms) {
    return;
  }
  last_show_time = now;

  // 调整亮度

  double brightness_factor = brightness_slider / 100.0;
  src_img.convertTo(debug_img, -1, brightness_factor, 0);

  cv::cvtColor(debug_img, debug_img, cv::COLOR_BGR2RGB);
  std::vector<cv::Point2f> all_corners;

  for (auto &pts : all_pts) {
    for (size_t i = 0; i < 4; ++i) {
      cv::line(debug_img, pts[i], pts[(i + 1) % 4], cv::Scalar(48, 48, 255), 1);
    }
    all_corners.insert(all_corners.end(), pts.begin(), pts.end());
  }
  cv::circle(debug_img, cv::Point2i(1440 / 2., 1080 / 2.), 5,
             cv::Scalar(255, 255, 255), 2);

  if (!all_corners.empty()) {
    cv::Point2f center(0.f, 0.f);
    for (const auto &pt : all_corners) {
      center += pt;
    }
    center *= 1.0f / all_corners.size();
    cv::circle(debug_img, center, 5, cv::Scalar(0, 255, 0), -1);
  }

  auto latency_duration = now - target.timestamp;
  auto latency_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(latency_duration)
          .count() /
      1000.0;

  std::string latency = fmt::format("Latency: {:.3f}ms", latency_ms);
  cv::putText(debug_img, latency, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
              0.8, cv::Scalar(100, 255, 0), 2);
  std::string state_str;
  switch (state) {
  case Tracker::LOST:
    state_str = "LOST";
    break;
  case Tracker::DETECTING:
    state_str = "DETECTING";
    break;
  case Tracker::TRACKING:
    state_str = "TRACKING";
    break;
  case Tracker::TEMP_LOST:
    state_str = "TEMP_LOST";
    break;
  default:
    state_str = "UNKNOWN";
    break;
  }
  int baseline = 0;
  cv::Size text_size =
      cv::getTextSize(state_str, cv::FONT_HERSHEY_SIMPLEX, 2.8, 2, &baseline);
  cv::Point text_org(debug_img.cols - text_size.width - 10,
                     text_size.height + 10);
  cv::putText(debug_img, state_str, text_org, cv::FONT_HERSHEY_SIMPLEX, 2.8,
              cv::Scalar(0, 0, 255), 2);

  cv::imshow("debug_target", debug_img);
  cv::waitKey(1);
}

void drawreprojec(const cv::Mat &src_img, const Target_info target_info,
                  const Target target, const Tracker::State state) {
  static auto last_show_time = std::chrono::steady_clock::now();
  static bool window_initialized = false;
  static int brightness_slider = 200;
  static cv::Mat debug_img;

  if (src_img.empty())
    return;

  if (!window_initialized) {
    cv::namedWindow("debug_target", cv::WINDOW_NORMAL);
    cv::resizeWindow("debug_target", debug_w, debug_h);
    cv::createTrackbar("Brightness", "debug_target", &brightness_slider, 400);
    window_initialized = true;
  }

  auto now = std::chrono::steady_clock::now();
  constexpr double min_interval_ms = 1000.0 / 60.0;
  double elapsed_ms =
      std::chrono::duration<double, std::milli>(now - last_show_time).count();
  if (elapsed_ms < min_interval_ms)
    return;
  last_show_time = now;

  double brightness_factor = brightness_slider / 100.0;
  src_img.convertTo(debug_img, -1, brightness_factor, 0);
  cv::cvtColor(debug_img, debug_img, cv::COLOR_BGR2RGB);

  std::vector<cv::Point2f> all_corners;

  for (size_t i = 0; i < target_info.pts.size(); ++i) {
    const auto &pts = target_info.pts[i];
    const auto &position = target_info.pos[i];
    const auto &orientation = target_info.ori[i];

    for (size_t j = 0; j < pts.size(); ++j) {
      cv::line(debug_img, pts[j], pts[(j + 1) % 4], cv::Scalar(255, 0, 0), 2);
    }

    all_corners.insert(all_corners.end(), pts.begin(), pts.end());

    double yaw = getYawFromQuaternion(orientation);
    double distance =
        std::sqrt(position.x * position.x + position.y * position.y +
                  position.z * position.z);

    std::vector<std::string> info_lines = {
        fmt::format("Dis: {:.3f}", distance * 100),
        fmt::format("X: {:.3f}", position.x),
        fmt::format("Y: {:.3f}", position.y),
        fmt::format("Z: {:.3f}", position.z),
        fmt::format("Yaw: {:.3f}", yaw * 180.0 / M_PI)};

    cv::Point2f text_org = pts[0] + cv::Point2f(0, 200);
    for (int k = 0; k < info_lines.size(); ++k) {
      cv::putText(debug_img, info_lines[k],
                  text_org + cv::Point2f(0, -10 - 20 * k),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(50, 255, 255), 1);
    }
  }

  // 绘制全局图像中心
  cv::circle(debug_img, cv::Point2i(1440 / 2., 1080 / 2.), 5,
             cv::Scalar(255, 255, 255), 2);

  // 绘制 target 角点中心
  if (!all_corners.empty()) {
    cv::Point2f center(0.f, 0.f);
    for (const auto &pt : all_corners) {
      center += pt;
    }
    center *= 1.0f / all_corners.size();
    cv::circle(debug_img, center, 5, cv::Scalar(0, 255, 0), -1); // 绿色实心圆
  }

  auto latency_duration = now - target.timestamp;
  double latency_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(latency_duration)
          .count() /
      1000.0;
  std::string latency = fmt::format("Latency: {:.3f}ms", latency_ms);
  cv::putText(debug_img, latency, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
              0.8, cv::Scalar(255, 255, 255), 2);

  std::string state_str;
  switch (state) {
  case Tracker::LOST:
    state_str = "LOST";
    break;
  case Tracker::DETECTING:
    state_str = "DETECTING";
    break;
  case Tracker::TRACKING:
    state_str = "TRACKING";
    break;
  case Tracker::TEMP_LOST:
    state_str = "TEMP_LOST";
    break;
  default:
    state_str = "UNKNOWN";
    break;
  }

  int baseline = 0;
  cv::Size state_text_size =
      cv::getTextSize(state_str, cv::FONT_HERSHEY_SIMPLEX, 2.8, 2, &baseline);
  cv::Point state_text_org(debug_img.cols - state_text_size.width - 10,
                           state_text_size.height + 10);
  cv::putText(debug_img, state_str, state_text_org, cv::FONT_HERSHEY_SIMPLEX,
              2.8, cv::Scalar(0, 0, 255), 2);

  auto armorNumberToString = [](ArmorNumber num) -> std::string {
    switch (num) {
    case ArmorNumber::SENTRY:
      return "SENTRY";
    case ArmorNumber::NO1:
      return "NO1";
    case ArmorNumber::NO2:
      return "NO2";
    case ArmorNumber::NO3:
      return "NO3";
    case ArmorNumber::NO4:
      return "NO4";
    case ArmorNumber::NO5:
      return "NO5";
    case ArmorNumber::OUTPOST:
      return "OUTPOST";
    case ArmorNumber::BASE:
      return "BASE";
    default:
      return "UNKNOWN";
    }
  };

  std::string armor_str = "Attack: " + armorNumberToString(target.id);
  cv::Size armor_text_size =
      cv::getTextSize(armor_str, cv::FONT_HERSHEY_SIMPLEX, 1.6, 2, &baseline);
  cv::Point armor_text_org(debug_img.cols - armor_text_size.width - 10,
                           state_text_org.y + state_text_size.height + 20);
  cv::putText(debug_img, armor_str, armor_text_org, cv::FONT_HERSHEY_SIMPLEX,
              1.6, cv::Scalar(255, 0, 255), 2);

  cv::imshow("debug_target", debug_img);
  cv::waitKey(1);
}

void drawreprojec(const imgframe &src_img, const Target_info target_info,
                  const Target target, const Tracker::State state) {
  static auto last_show_time = std::chrono::steady_clock::now();
  static bool window_initialized = false;
  static int brightness_slider = 200;
  static cv::Mat debug_img;
  if (src_img.img.empty()) {
    return;
  }

  if (!window_initialized) {
    cv::namedWindow("debug_target", cv::WINDOW_NORMAL);
    cv::resizeWindow("debug_target", debug_w, debug_h);
    cv::createTrackbar("Brightness", "debug_target", &brightness_slider, 400);
    window_initialized = true;
  }
  auto now = std::chrono::steady_clock::now();
  constexpr double min_interval_ms = 1000.0 / 60.0;

  double elapsed_ms =
      std::chrono::duration<double, std::milli>(now - last_show_time).count();
  if (elapsed_ms < min_interval_ms) {
    return;
  }
  last_show_time = now;

  double brightness_factor = brightness_slider / 100.0;
  src_img.img.convertTo(debug_img, -1, brightness_factor, 0);

  cv::cvtColor(debug_img, debug_img, cv::COLOR_BGR2RGB);
  std::vector<cv::Point2f> all_corners;

  for (size_t i = 0; i < target_info.pts.size(); ++i) {
    const auto &pts = target_info.pts[i];
    const auto &position = target_info.pos[i];
    const auto &orientation = target_info.ori[i];

    for (size_t j = 0; j < pts.size(); ++j) {

      cv::line(debug_img, pts[j], pts[(j + 1) % 4], cv::Scalar(255, 0, 0), 2);
    }
    all_corners.insert(all_corners.end(), pts.begin(), pts.end());
    // 计算 yaw
    double yaw = getYawFromQuaternion(orientation);
    double distance =
        std::sqrt(position.x * position.x + position.y * position.y +
                  position.z * position.z);

    std::vector<std::string> info_lines = {
        fmt::format("Dis: {:.3f}", distance * 100),
        fmt::format("X: {:.3f}", position.x),
        fmt::format("Y: {:.3f}", position.y),
        fmt::format("Z: {:.3f}", position.z),
        fmt::format("Yaw: {:.3f}", yaw * 180.0 / M_PI)};
    cv::Point2f text_org = pts[0] + cv::Point2f(0, 200);
    for (int k = 0; k < info_lines.size(); ++k) {
      cv::putText(debug_img, info_lines[k],
                  text_org + cv::Point2f(0, -10 - 20 * k),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(50, 255, 255), 1);
    }
  }
  if (target_info.select_id != -1) {
    if (!target_info.pts[target_info.select_id].empty()) {
      cv::Point pt = target_info.pts[target_info.select_id][0];
      cv::circle(debug_img, pt, 10, cv::Scalar(0, 0, 255), 2);
    }
  }
  cv::circle(debug_img, cv::Point2i(1440 / 2., 1080 / 2.), 5,
             cv::Scalar(255, 255, 255), 2);

  if (!all_corners.empty()) {
    cv::Point2f center(0.f, 0.f);
    for (const auto &pt : all_corners) {
      center += pt;
    }
    center *= 1.0f / all_corners.size();
    cv::circle(debug_img, center, 5, cv::Scalar(0, 255, 0), -1);
  }
  auto latency_duration = now - target.timestamp;
  auto latency_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(latency_duration)
          .count() /
      1000.0;

  std::string latency = fmt::format("Latency: {:.3f}ms", latency_ms);
  cv::putText(debug_img, latency, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
              0.8, cv::Scalar(255, 255, 255), 2);

  auto latency_target_img =
      std::chrono::duration_cast<std::chrono::microseconds>(src_img.timestamp -
                                                            target.timestamp)
          .count() /
      1000.0;
  std::string latency_t_i =
      fmt::format("Latency of img-target: {:.2f}ms", latency_target_img);
  cv::putText(debug_img, latency_t_i, cv::Point(10, 60),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

  int baseline = 0;

  std::string state_str;
  switch (state) {
  case Tracker::LOST:
    state_str = "LOST";
    break;
  case Tracker::DETECTING:
    state_str = "DETECTING";
    break;
  case Tracker::TRACKING:
    state_str = "TRACKING";
    break;
  case Tracker::TEMP_LOST:
    state_str = "TEMP_LOST";
    break;
  default:
    state_str = "UNKNOWN";
    break;
  }
  cv::Size state_text_size =
      cv::getTextSize(state_str, cv::FONT_HERSHEY_SIMPLEX, 2.8, 2, &baseline);
  cv::Point state_text_org(debug_img.cols - state_text_size.width - 10,
                           state_text_size.height + 10);
  cv::putText(debug_img, state_str, state_text_org, cv::FONT_HERSHEY_SIMPLEX,
              2.8, cv::Scalar(0, 0, 255), 2);

  // 右上角再往下显示 armor_str（与 state_str 平行对齐）
  auto armorNumberToString = [](ArmorNumber num) -> std::string {
    switch (num) {
    case ArmorNumber::SENTRY:
      return "SENTRY";
    case ArmorNumber::NO1:
      return "NO1";
    case ArmorNumber::NO2:
      return "NO2";
    case ArmorNumber::NO3:
      return "NO3";
    case ArmorNumber::NO4:
      return "NO4";
    case ArmorNumber::NO5:
      return "NO5";
    case ArmorNumber::OUTPOST:
      return "OUTPOST";
    case ArmorNumber::BASE:
      return "BASE";
    default:
      return "UNKNOWN";
    }
  };
  std::string armor_str = "Attack: " + armorNumberToString(target.id);
  cv::Size armor_text_size =
      cv::getTextSize(armor_str, cv::FONT_HERSHEY_SIMPLEX, 1.6, 2, &baseline);
  cv::Point armor_text_org(debug_img.cols - armor_text_size.width - 10,
                           state_text_org.y + state_text_size.height + 20);
  cv::putText(debug_img, armor_str, armor_text_org, cv::FONT_HERSHEY_SIMPLEX,
              1.6, cv::Scalar(255, 0, 255), 2);

  cv::imshow("debug_target", debug_img);
  cv::waitKey(1);
}
void drawreprojec(const imgframe &src_img, const Target_info target_info,
                  const Target target, const Tracker::State state,
                  GimbalCmd gimbal_cmd) {
  static auto last_show_time = std::chrono::steady_clock::now();
  static bool window_initialized = false;
  static int brightness_slider = 200;
  static cv::Mat debug_img;
  if (src_img.img.empty()) {
    return;
  }

  if (!window_initialized) {
    cv::namedWindow("debug_target", cv::WINDOW_NORMAL);
    cv::resizeWindow("debug_target", debug_w, debug_h);
    cv::createTrackbar("Brightness", "debug_target", &brightness_slider, 400);
    window_initialized = true;
  }
  auto now = std::chrono::steady_clock::now();
  constexpr double min_interval_ms = 1000.0 / 60.0;

  double elapsed_ms =
      std::chrono::duration<double, std::milli>(now - last_show_time).count();
  if (elapsed_ms < min_interval_ms) {
    return;
  }
  last_show_time = now;

  double brightness_factor = brightness_slider / 100.0;
  src_img.img.convertTo(debug_img, -1, brightness_factor, 0);

  cv::cvtColor(debug_img, debug_img, cv::COLOR_BGR2RGB);
  std::vector<cv::Point2f> all_corners;

  for (size_t i = 0; i < target_info.pts.size(); ++i) {
    const auto &pts = target_info.pts[i];
    const auto &position = target_info.pos[i];
    const auto &orientation = target_info.ori[i];

    for (size_t j = 0; j < pts.size(); ++j) {

      cv::line(debug_img, pts[j], pts[(j + 1) % 4], cv::Scalar(255, 0, 0), 2);
    }
    all_corners.insert(all_corners.end(), pts.begin(), pts.end());
    // 计算 yaw
    double yaw = getYawFromQuaternion(orientation);
    double distance =
        std::sqrt(position.x * position.x + position.y * position.y +
                  position.z * position.z);

    std::vector<std::string> info_lines = {
        fmt::format("Dis: {:.3f}", distance * 100),
        fmt::format("X: {:.3f}", position.x),
        fmt::format("Y: {:.3f}", position.y),
        fmt::format("Z: {:.3f}", position.z),
        fmt::format("Yaw: {:.3f}", yaw * 180.0 / M_PI)};
    cv::Point2f text_org = pts[0] + cv::Point2f(0, 200);
    for (int k = 0; k < info_lines.size(); ++k) {
      cv::putText(debug_img, info_lines[k],
                  text_org + cv::Point2f(0, -10 - 20 * k),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(50, 255, 255), 1);
    }
  }
  if (target_info.select_id != -1) {
    if (!target_info.pts[target_info.select_id].empty()) {
      cv::Point2f center(0.f, 0.f);
      for (int i = 0; i < 4; ++i) {
        center += target_info.pts[target_info.select_id][i];
      }
      center *= 0.25f; // 四个点取平均值
      cv::Point2f pt = center + cv::Point2f(0, -200);
      cv::circle(debug_img, pt, 20, cv::Scalar(0, 0, 255), 5);
    }
  }
  cv::circle(debug_img, cv::Point2i(1440 / 2., 1080 / 2.), 5,
             cv::Scalar(255, 255, 255), 2);

  if (!all_corners.empty()) {
    cv::Point2f center(0.f, 0.f);
    for (const auto &pt : all_corners) {
      center += pt;
    }
    center *= 1.0f / all_corners.size();
    cv::circle(debug_img, center, 5, cv::Scalar(0, 255, 0), -1);
  }
  auto latency_duration = now - target.timestamp;
  auto latency_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(latency_duration)
          .count() /
      1000.0;

  std::string latency = fmt::format("Latency: {:.3f}ms", latency_ms);
  cv::putText(debug_img, latency, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
              0.8, cv::Scalar(255, 255, 255), 2);

  auto latency_target_img =
      std::chrono::duration_cast<std::chrono::microseconds>(src_img.timestamp -
                                                            target.timestamp)
          .count() /
      1000.0;
  std::string latency_t_i =
      fmt::format("Latency of img-target: {:.2f}ms", latency_target_img);
  cv::putText(debug_img, latency_t_i, cv::Point(10, 60),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

  int baseline = 0;

  std::string state_str;
  switch (state) {
  case Tracker::LOST:
    state_str = "LOST";
    break;
  case Tracker::DETECTING:
    state_str = "DETECTING";
    break;
  case Tracker::TRACKING:
    state_str = "TRACKING";
    break;
  case Tracker::TEMP_LOST:
    state_str = "TEMP_LOST";
    break;
  default:
    state_str = "UNKNOWN";
    break;
  }
  cv::Size state_text_size =
      cv::getTextSize(state_str, cv::FONT_HERSHEY_SIMPLEX, 2.8, 2, &baseline);
  cv::Point state_text_org(debug_img.cols - state_text_size.width - 10,
                           state_text_size.height + 10);
  cv::putText(debug_img, state_str, state_text_org, cv::FONT_HERSHEY_SIMPLEX,
              2.8, cv::Scalar(0, 0, 255), 2);

  // 右上角再往下显示 armor_str（与 state_str 平行对齐）
  auto armorNumberToString = [](ArmorNumber num) -> std::string {
    switch (num) {
    case ArmorNumber::SENTRY:
      return "SENTRY";
    case ArmorNumber::NO1:
      return "NO1";
    case ArmorNumber::NO2:
      return "NO2";
    case ArmorNumber::NO3:
      return "NO3";
    case ArmorNumber::NO4:
      return "NO4";
    case ArmorNumber::NO5:
      return "NO5";
    case ArmorNumber::OUTPOST:
      return "OUTPOST";
    case ArmorNumber::BASE:
      return "BASE";
    default:
      return "UNKNOWN";
    }
  };
  std::string armor_str = "Attack: " + armorNumberToString(target.id);
  cv::Size armor_text_size =
      cv::getTextSize(armor_str, cv::FONT_HERSHEY_SIMPLEX, 1.6, 2, &baseline);
  cv::Point armor_text_org(debug_img.cols - armor_text_size.width - 10,
                           state_text_org.y + state_text_size.height + 20);
  cv::putText(debug_img, armor_str, armor_text_org, cv::FONT_HERSHEY_SIMPLEX,
              1.6, cv::Scalar(255, 0, 255), 2);
  std::string gimbal_info = fmt::format("Pitch: {:.2f}, Yaw: {:.2f}",
                                        gimbal_cmd.pitch, gimbal_cmd.yaw);
  std::string gimbal_diff_info =
      fmt::format("Pitch_Diff: {:.2f}, Yaw_Diff: {:.2f}", gimbal_cmd.pitch_diff,
                  gimbal_cmd.yaw_diff);
  cv::putText(debug_img, gimbal_info, cv::Point(10, 90),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
  cv::putText(debug_img, gimbal_diff_info, cv::Point(10, 120),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
  if (gimbal_cmd.fire_advice) {
    std::string shoot_text = "SHOOT";
    int baseline = 0;
    cv::Size shoot_text_size =
        cv::getTextSize(shoot_text, cv::FONT_HERSHEY_DUPLEX, 3.5, 5, &baseline);
    cv::Point shoot_text_org(debug_img.cols - shoot_text_size.width - 20,
                             armor_text_org.y + armor_text_size.height + 40);
    cv::putText(debug_img, shoot_text, shoot_text_org, cv::FONT_HERSHEY_DUPLEX,
                3.5, cv::Scalar(0, 0, 255), 5); // 红色加粗
  }

  // drawGimbalDirection(debug_img, gimbal_cmd);

  cv::imshow("debug_target", debug_img);
  cv::waitKey(1);
}
void draw_debug_overlay(const imgframe &src_img, const Armors *armors,
                        const Target_info *target_info, const Target *target,
                        const std::optional<Tracker::State> &state,
                        const std::optional<GimbalCmd> &gimbal_cmd) {
  static auto last_show_time = std::chrono::steady_clock::now();
  static bool window_initialized = false;
  static int brightness_slider = 200;

  if (src_img.img.empty())
    return;

  if (!window_initialized) {
    cv::namedWindow("debug_overlay", cv::WINDOW_NORMAL);
    cv::resizeWindow("debug_overlay", debug_w, debug_h);
    cv::createTrackbar("Brightness", "debug_overlay", &brightness_slider, 400);
    window_initialized = true;
  }

  auto now = std::chrono::steady_clock::now();
  constexpr double min_interval_ms = 1000.0 / 45.0;
  double elapsed_ms =
      std::chrono::duration<double, std::milli>(now - last_show_time).count();
  if (elapsed_ms < min_interval_ms)
    return;
  last_show_time = now;

  // 图像亮度调整
  double brightness_factor = brightness_slider / 100.0;
  cv::Mat debug_img;
  src_img.img.convertTo(debug_img, -1, brightness_factor, 0);
  cv::cvtColor(debug_img, debug_img, cv::COLOR_BGR2RGB);

  // ========= 绘制 Armors =========
  static float yaw_diff = 0;
  if (armors) {
    static const int next_indices[] = {2, 0, 3, 1};

    for (const auto &armor : armors->armors) {
      std::vector<cv::Point2f> pts;
      if (!measure_tool_->reprojectArmorCorners_raw(armor, pts))
        continue;

      for (size_t i = 0; i < 4; ++i)
        cv::line(debug_img, pts[i], pts[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);

      std::string yaw_info =
          fmt::format("Yaw: {:.2f}", armor.yaw * 180.0 / M_PI);
      cv::putText(debug_img, yaw_info, pts[0] + cv::Point2f(0, -50),
                  cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 200, 200), 2);

      std::string armor_str;
      switch (armor.number) {
      case ArmorNumber::SENTRY:
        armor_str = "SENTRY";
        break;
      case ArmorNumber::BASE:
        armor_str = "BASE";
        break;
      case ArmorNumber::OUTPOST:
        armor_str = "OUTPOST";
        break;
      case ArmorNumber::NO1:
        armor_str = "NO1";
        break;
      case ArmorNumber::NO2:
        armor_str = "NO2";
        break;
      case ArmorNumber::NO3:
        armor_str = "NO3";
        break;
      case ArmorNumber::NO4:
        armor_str = "NO4";
        break;
      case ArmorNumber::NO5:
        armor_str = "NO5";
        break;
      default:
        armor_str = "UNKNOWN";
        break;
      }

      cv::putText(debug_img, armor_str, pts[1] + cv::Point2f(0, 50),
                  cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 200, 200), 2);
    }

    if (armors->armors.size() == 2) {
      double diff = armors->armors[0].yaw - armors->armors[1].yaw;
      while (diff > M_PI)
        diff -= 2 * M_PI;
      while (diff < -M_PI)
        diff += 2 * M_PI;
      yaw_diff = std::abs(diff);
    }

    std::string yaw_diff_str =
        fmt::format("Yaw_diff: {:.2f}", yaw_diff * 180.0 / M_PI);
    cv::putText(debug_img, yaw_diff_str, cv::Point(100, 150),
                cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(40, 255, 40), 2);

    double latency =
        std::chrono::duration<double, std::milli>(now - armors->timestamp)
            .count();
    std::string latency_str = fmt::format("Latency: {:.2f}ms", latency);
    cv::putText(debug_img, latency_str, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
  }

  // ========= 绘制 Target =========
  std::vector<cv::Point2f> all_corners;
  if (target_info && target) {
    for (size_t i = 0; i < target_info->pts.size(); ++i) {
      const auto &pts = target_info->pts[i];
      const auto &pos = target_info->pos[i];
      const auto &ori = target_info->ori[i];

      for (size_t j = 0; j < 4; ++j)
        cv::line(debug_img, pts[j], pts[(j + 1) % 4], cv::Scalar(255, 0, 0), 2);

      all_corners.insert(all_corners.end(), pts.begin(), pts.end());

      double yaw = getYawFromQuaternion(ori);
      double distance =
          std::sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);

      std::vector<std::string> info_lines = {
          fmt::format("Dis: {:.1f}cm", distance * 100),
          fmt::format("X: {:.2f}", pos.x), fmt::format("Y: {:.2f}", pos.y),
          fmt::format("Z: {:.2f}", pos.z),
          fmt::format("Yaw: {:.2f}", yaw * 180.0 / M_PI)};

      cv::Point2f text_org = pts[0] + cv::Point2f(0, 200);
      for (int k = 0; k < info_lines.size(); ++k) {
        cv::putText(debug_img, info_lines[k],
                    text_org + cv::Point2f(0, -10 - 20 * k),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(50, 255, 255), 1);
      }
    }

    if (target_info->select_id != -1 &&
        !target_info->pts[target_info->select_id].empty()) {
      cv::Point2f center(0.f, 0.f);
      for (int i = 0; i < 4; ++i)
        center += target_info->pts[target_info->select_id][i];
      center *= 0.25f;
      cv::circle(debug_img, center + cv::Point2f(0, -200), 20,
                 cv::Scalar(0, 0, 255), 5);
    }

    if (!all_corners.empty()) {
      cv::Point2f avg(0.f, 0.f);
      for (const auto &pt : all_corners)
        avg += pt;
      avg *= 1.0f / all_corners.size();
      cv::circle(debug_img, avg, 5, cv::Scalar(0, 255, 0), -1);
    }

    auto latency = std::chrono::duration_cast<std::chrono::microseconds>(
                       now - target->timestamp)
                       .count() /
                   1000.0;
    auto latency_img_target =
        std::chrono::duration_cast<std::chrono::microseconds>(
            src_img.timestamp - target->timestamp)
            .count() /
        1000.0;

    cv::putText(debug_img,
                fmt::format("Img-Frame Delay: {:.2f}ms", latency_img_target),
                cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(255, 255, 255), 2);
  }

  int baseline = 0;
  if (state.has_value()) {
    std::string state_str;
    switch (state.value()) {
    case Tracker::LOST:
      state_str = "LOST";
      break;
    case Tracker::DETECTING:
      state_str = "DETECTING";
      break;
    case Tracker::TRACKING:
      state_str = "TRACKING";
      break;
    case Tracker::TEMP_LOST:
      state_str = "TEMP_LOST";
      break;
    default:
      state_str = "UNKNOWN";
      break;
    }
    cv::Size state_size =
        cv::getTextSize(state_str, cv::FONT_HERSHEY_SIMPLEX, 2.5, 2, &baseline);
    int x = std::max(0, debug_img.cols - state_size.width - 10);
    int y = std::min(debug_img.rows - 1, state_size.height + 10);
    cv::putText(debug_img, state_str, {x, y}, cv::FONT_HERSHEY_SIMPLEX, 2.5,
                cv::Scalar(0, 0, 255), 2);
  }

  if (target) {
    auto armorName = [](ArmorNumber num) {
      switch (num) {
      case ArmorNumber::SENTRY:
        return "SENTRY";
      case ArmorNumber::BASE:
        return "BASE";
      case ArmorNumber::OUTPOST:
        return "OUTPOST";
      case ArmorNumber::NO1:
        return "NO1";
      case ArmorNumber::NO2:
        return "NO2";
      case ArmorNumber::NO3:
        return "NO3";
      case ArmorNumber::NO4:
        return "NO4";
      case ArmorNumber::NO5:
        return "NO5";
      default:
        return "UNKNOWN";
      }
    };
    std::string id_str = fmt::format("Attack: {}", armorName(target->id));
    cv::Size id_size =
        cv::getTextSize(id_str, cv::FONT_HERSHEY_SIMPLEX, 1.6, 2, &baseline);
    int x = std::max(0, debug_img.cols - id_size.width - 10);
    int y = std::min(debug_img.rows - 1, 100);
    cv::putText(debug_img, id_str, {x, y}, cv::FONT_HERSHEY_SIMPLEX, 1.6,
                cv::Scalar(255, 0, 255), 2);
  }
  std::string fire_str = gimbal_cmd && gimbal_cmd->fire_advice ? "Fire!" : "";
  cv::Size fire_size =
      cv::getTextSize(fire_str, cv::FONT_HERSHEY_SIMPLEX, 1.2, 2, &baseline);
  int fire_x = 1440 / 2 - fire_size.width - 10;
  int fire_y = 200;

  cv::putText(debug_img, fire_str, {fire_x, fire_y}, cv::FONT_HERSHEY_SIMPLEX,
              2.85, cv::Scalar(0, 0, 255), 2);

  if (gimbal_cmd.has_value()) {
    std::string gimbal_str = fmt::format(
        "Pitch: {:.2f}, Yaw: {:.2f}, Pitch_diff: {:.2f}, Yaw_diff: {:.2f}",
        gimbal_cmd->pitch, gimbal_cmd->yaw, gimbal_cmd->pitch_diff,
        gimbal_cmd->yaw_diff);
    cv::putText(debug_img, gimbal_str, {10, debug_img.rows - 30},
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
  }

  cv::circle(debug_img, cv::Point2i(1440 / 2, 1080 / 2), 5,
             cv::Scalar(255, 255, 255), 2);
  cv::imshow("debug_overlay", debug_img);
  cv::waitKey(1);
}
void draw_debug_overlaywrite(const imgframe &src_img, const Armors *armors,
                             const Target_info *target_info,
                             const Target *target,
                             const std::optional<Tracker::State> &state,
                             const std::optional<GimbalCmd> &gimbal_cmd) {
  static auto last_show_time = std::chrono::steady_clock::now();
  // static bool window_initialized = false;

  if (src_img.img.empty())
    return;

  // if (!window_initialized) {
  //   cv::namedWindow("debug_overlay", cv::WINDOW_NORMAL);
  //   cv::resizeWindow("debug_overlay", debug_w, debug_h);
  //   cv::createTrackbar("Brightness", "debug_overlay", &brightness_slider,
  //   400); window_initialized = true;
  // }

  auto now = std::chrono::steady_clock::now();
  constexpr double min_interval_ms = 1000.0 / 45.0;
  double elapsed_ms =
      std::chrono::duration<double, std::milli>(now - last_show_time).count();
  if (elapsed_ms < min_interval_ms)
    return;
  last_show_time = now;

  // 图像亮度调整

  cv::Mat debug_img;
  src_img.img.convertTo(debug_img, -1, 1, 0);
  cv::cvtColor(debug_img, debug_img, cv::COLOR_BGR2RGB);

  // ========= 绘制 Armors =========
  static float yaw_diff = 0;
  if (armors) {
    static const int next_indices[] = {2, 0, 3, 1};

    for (const auto &armor : armors->armors) {
      std::vector<cv::Point2f> pts;
      if (!measure_tool_->reprojectArmorCorners_raw(armor, pts))
        continue;

      for (size_t i = 0; i < 4; ++i)
        cv::line(debug_img, pts[i], pts[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);

      std::string yaw_info =
          fmt::format("Yaw: {:.2f}", armor.yaw * 180.0 / M_PI);
      cv::putText(debug_img, yaw_info, pts[0] + cv::Point2f(0, -50),
                  cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 200, 200), 2);

      std::string armor_str;
      switch (armor.number) {
      case ArmorNumber::SENTRY:
        armor_str = "SENTRY";
        break;
      case ArmorNumber::BASE:
        armor_str = "BASE";
        break;
      case ArmorNumber::OUTPOST:
        armor_str = "OUTPOST";
        break;
      case ArmorNumber::NO1:
        armor_str = "NO1";
        break;
      case ArmorNumber::NO2:
        armor_str = "NO2";
        break;
      case ArmorNumber::NO3:
        armor_str = "NO3";
        break;
      case ArmorNumber::NO4:
        armor_str = "NO4";
        break;
      case ArmorNumber::NO5:
        armor_str = "NO5";
        break;
      default:
        armor_str = "UNKNOWN";
        break;
      }

      cv::putText(debug_img, armor_str, pts[1] + cv::Point2f(0, 50),
                  cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 200, 200), 2);
    }

    if (armors->armors.size() == 2) {
      double diff = armors->armors[0].yaw - armors->armors[1].yaw;
      while (diff > M_PI)
        diff -= 2 * M_PI;
      while (diff < -M_PI)
        diff += 2 * M_PI;
      yaw_diff = std::abs(diff);
    }

    std::string yaw_diff_str =
        fmt::format("Yaw_diff: {:.2f}", yaw_diff * 180.0 / M_PI);
    cv::putText(debug_img, yaw_diff_str, cv::Point(100, 150),
                cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(40, 255, 40), 2);

    double latency =
        std::chrono::duration<double, std::milli>(now - armors->timestamp)
            .count();
    std::string latency_str = fmt::format("Latency: {:.2f}ms", latency);
    cv::putText(debug_img, latency_str, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
  }

  // ========= 绘制 Target =========
  std::vector<cv::Point2f> all_corners;
  if (target_info && target) {
    for (size_t i = 0; i < target_info->pts.size(); ++i) {
      const auto &pts = target_info->pts[i];
      const auto &pos = target_info->pos[i];
      const auto &ori = target_info->ori[i];

      for (size_t j = 0; j < 4; ++j)
        cv::line(debug_img, pts[j], pts[(j + 1) % 4], cv::Scalar(255, 0, 0), 2);

      all_corners.insert(all_corners.end(), pts.begin(), pts.end());

      double yaw = getYawFromQuaternion(ori);
      double distance =
          std::sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);

      std::vector<std::string> info_lines = {
          fmt::format("Dis: {:.1f}cm", distance * 100),
          fmt::format("X: {:.2f}", pos.x), fmt::format("Y: {:.2f}", pos.y),
          fmt::format("Z: {:.2f}", pos.z),
          fmt::format("Yaw: {:.2f}", yaw * 180.0 / M_PI)};

      cv::Point2f text_org = pts[0] + cv::Point2f(0, 200);
      for (int k = 0; k < info_lines.size(); ++k) {
        cv::putText(debug_img, info_lines[k],
                    text_org + cv::Point2f(0, -10 - 20 * k),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(50, 255, 255), 1);
      }
    }

    if (target_info->select_id != -1 &&
        !target_info->pts[target_info->select_id].empty()) {
      cv::Point2f center(0.f, 0.f);
      for (int i = 0; i < 4; ++i)
        center += target_info->pts[target_info->select_id][i];
      center *= 0.25f;
      cv::circle(debug_img, center + cv::Point2f(0, -200), 20,
                 cv::Scalar(0, 0, 255), 5);
    }

    if (!all_corners.empty()) {
      cv::Point2f avg(0.f, 0.f);
      for (const auto &pt : all_corners)
        avg += pt;
      avg *= 1.0f / all_corners.size();
      cv::circle(debug_img, avg, 5, cv::Scalar(0, 255, 0), -1);
    }

    auto latency = std::chrono::duration_cast<std::chrono::microseconds>(
                       now - target->timestamp)
                       .count() /
                   1000.0;
    auto latency_img_target =
        std::chrono::duration_cast<std::chrono::microseconds>(
            src_img.timestamp - target->timestamp)
            .count() /
        1000.0;

    cv::putText(debug_img,
                fmt::format("Img-Frame Delay: {:.2f}ms", latency_img_target),
                cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(255, 255, 255), 2);
  }

  int baseline = 0;
  if (state.has_value()) {
    std::string state_str;
    switch (state.value()) {
    case Tracker::LOST:
      state_str = "LOST";
      break;
    case Tracker::DETECTING:
      state_str = "DETECTING";
      break;
    case Tracker::TRACKING:
      state_str = "TRACKING";
      break;
    case Tracker::TEMP_LOST:
      state_str = "TEMP_LOST";
      break;
    default:
      state_str = "UNKNOWN";
      break;
    }
    cv::Size state_size =
        cv::getTextSize(state_str, cv::FONT_HERSHEY_SIMPLEX, 2.5, 2, &baseline);
    int x = std::max(0, debug_img.cols - state_size.width - 10);
    int y = std::min(debug_img.rows - 1, state_size.height + 10);
    cv::putText(debug_img, state_str, {x, y}, cv::FONT_HERSHEY_SIMPLEX, 2.5,
                cv::Scalar(0, 0, 255), 2);
  }

  if (target) {
    auto armorName = [](ArmorNumber num) {
      switch (num) {
      case ArmorNumber::SENTRY:
        return "SENTRY";
      case ArmorNumber::BASE:
        return "BASE";
      case ArmorNumber::OUTPOST:
        return "OUTPOST";
      case ArmorNumber::NO1:
        return "NO1";
      case ArmorNumber::NO2:
        return "NO2";
      case ArmorNumber::NO3:
        return "NO3";
      case ArmorNumber::NO4:
        return "NO4";
      case ArmorNumber::NO5:
        return "NO5";
      default:
        return "UNKNOWN";
      }
    };
    std::string id_str = fmt::format("Attack: {}", armorName(target->id));
    cv::Size id_size =
        cv::getTextSize(id_str, cv::FONT_HERSHEY_SIMPLEX, 1.6, 2, &baseline);
    int x = std::max(0, debug_img.cols - id_size.width - 10);
    int y = std::min(debug_img.rows - 1, 100);
    cv::putText(debug_img, id_str, {x, y}, cv::FONT_HERSHEY_SIMPLEX, 1.6,
                cv::Scalar(255, 0, 255), 2);
  }
  std::string fire_str = gimbal_cmd && gimbal_cmd->fire_advice ? "Fire!" : "";
  cv::Size fire_size =
      cv::getTextSize(fire_str, cv::FONT_HERSHEY_SIMPLEX, 1.2, 2, &baseline);
  int fire_x = 1440 / 2 - fire_size.width - 10;
  int fire_y = 200;

  cv::putText(debug_img, fire_str, {fire_x, fire_y}, cv::FONT_HERSHEY_SIMPLEX,
              2.85, cv::Scalar(0, 0, 255), 2);

  if (gimbal_cmd.has_value()) {
    std::string gimbal_str = fmt::format(
        "Pitch: {:.2f}, Yaw: {:.2f}, Pitch_diff: {:.2f}, Yaw_diff: {:.2f}",
        gimbal_cmd->pitch, gimbal_cmd->yaw, gimbal_cmd->pitch_diff,
        gimbal_cmd->yaw_diff);
    cv::putText(debug_img, gimbal_str, {10, debug_img.rows - 30},
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
  }

  cv::circle(debug_img, cv::Point2i(1440 / 2, 1080 / 2), 5,
             cv::Scalar(255, 255, 255), 2);
  std::vector<uchar> buf;
  cv::imencode(".jpg", debug_img, buf);
  std::ofstream ofs("/dev/shm/debug_frame.jpg.tmp", std::ios::binary);
  ofs.write(reinterpret_cast<const char *>(buf.data()), buf.size());
  ofs.close();
  std::rename("/dev/shm/debug_frame.jpg.tmp", "/dev/shm/debug_frame.jpg");
}
cv::Mat draw_debug_overlayMat(const imgframe &src_img, const Armors *armors,
                              const Target_info *target_info,
                              const Target *target,
                              const std::optional<Tracker::State> &state,
                              const std::optional<GimbalCmd> &gimbal_cmd) {
  static auto last_show_time = std::chrono::steady_clock::now();
  static bool window_initialized = false;
  static int brightness_slider = 200;

  if (src_img.img.empty())
    return cv::Mat();

  // if (!window_initialized) {
  // cv::namedWindow("debug_overlay", cv::WINDOW_NORMAL);
  // cv::resizeWindow("debug_overlay", debug_w, debug_h);
  // cv::createTrackbar("Brightness", "debug_overlay", &brightness_slider, 400);
  // window_initialized = true;
  // }

  auto now = std::chrono::steady_clock::now();
  // constexpr double min_interval_ms = 1000.0 / 45.0;
  // double elapsed_ms =
  // std::chrono::duration<double, std::milli>(now - last_show_time).count();
  // if (elapsed_ms < min_interval_ms)
  // return cv::Mat();
  // last_show_time = now;

  // 图像亮度调整
  // double brightness_factor = brightness_slider / 100.0;
  cv::Mat debug_img;
  src_img.img.convertTo(debug_img, -1, 1, 0);
  cv::cvtColor(debug_img, debug_img, cv::COLOR_BGR2RGB);

  // ========= 绘制 Armors =========
  static float yaw_diff = 0;
  if (armors) {
    static const int next_indices[] = {2, 0, 3, 1};

    for (const auto &armor : armors->armors) {
      std::vector<cv::Point2f> pts;
      if (!measure_tool_->reprojectArmorCorners_raw(armor, pts))
        continue;

      for (size_t i = 0; i < 4; ++i)
        cv::line(debug_img, pts[i], pts[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);

      std::string yaw_info =
          fmt::format("Yaw: {:.2f}", armor.yaw * 180.0 / M_PI);
      cv::putText(debug_img, yaw_info, pts[0] + cv::Point2f(0, -50),
                  cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 200, 200), 2);

      std::string armor_str;
      switch (armor.number) {
      case ArmorNumber::SENTRY:
        armor_str = "SENTRY";
        break;
      case ArmorNumber::BASE:
        armor_str = "BASE";
        break;
      case ArmorNumber::OUTPOST:
        armor_str = "OUTPOST";
        break;
      case ArmorNumber::NO1:
        armor_str = "NO1";
        break;
      case ArmorNumber::NO2:
        armor_str = "NO2";
        break;
      case ArmorNumber::NO3:
        armor_str = "NO3";
        break;
      case ArmorNumber::NO4:
        armor_str = "NO4";
        break;
      case ArmorNumber::NO5:
        armor_str = "NO5";
        break;
      default:
        armor_str = "UNKNOWN";
        break;
      }

      cv::putText(debug_img, armor_str, pts[1] + cv::Point2f(0, 50),
                  cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 200, 200), 2);
    }

    if (armors->armors.size() == 2) {
      double diff = armors->armors[0].yaw - armors->armors[1].yaw;
      while (diff > M_PI)
        diff -= 2 * M_PI;
      while (diff < -M_PI)
        diff += 2 * M_PI;
      yaw_diff = std::abs(diff);
    }

    std::string yaw_diff_str =
        fmt::format("Yaw_diff: {:.2f}", yaw_diff * 180.0 / M_PI);
    cv::putText(debug_img, yaw_diff_str, cv::Point(100, 150),
                cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(40, 255, 40), 2);

    double latency =
        std::chrono::duration<double, std::milli>(now - armors->timestamp)
            .count();
    std::string latency_str = fmt::format("Latency: {:.2f}ms", latency);
    cv::putText(debug_img, latency_str, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
  }

  // ========= 绘制 Target =========
  std::vector<cv::Point2f> all_corners;
  if (target_info && target) {
    for (size_t i = 0; i < target_info->pts.size(); ++i) {
      const auto &pts = target_info->pts[i];
      const auto &pos = target_info->pos[i];
      const auto &ori = target_info->ori[i];

      for (size_t j = 0; j < 4; ++j)
        cv::line(debug_img, pts[j], pts[(j + 1) % 4], cv::Scalar(255, 0, 0), 2);

      all_corners.insert(all_corners.end(), pts.begin(), pts.end());

      double yaw = getYawFromQuaternion(ori);
      double distance =
          std::sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);

      std::vector<std::string> info_lines = {
          fmt::format("Dis: {:.1f}cm", distance * 100),
          fmt::format("X: {:.2f}", pos.x), fmt::format("Y: {:.2f}", pos.y),
          fmt::format("Z: {:.2f}", pos.z),
          fmt::format("Yaw: {:.2f}", yaw * 180.0 / M_PI)};

      cv::Point2f text_org = pts[0] + cv::Point2f(0, 200);
      for (int k = 0; k < info_lines.size(); ++k) {
        cv::putText(debug_img, info_lines[k],
                    text_org + cv::Point2f(0, -10 - 20 * k),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(50, 255, 255), 1);
      }
    }

    if (target_info->select_id != -1 &&
        !target_info->pts[target_info->select_id].empty()) {
      cv::Point2f center(0.f, 0.f);
      for (int i = 0; i < 4; ++i)
        center += target_info->pts[target_info->select_id][i];
      center *= 0.25f;
      cv::circle(debug_img, center + cv::Point2f(0, -200), 20,
                 cv::Scalar(0, 0, 255), 5);
    }

    if (!all_corners.empty()) {
      cv::Point2f avg(0.f, 0.f);
      for (const auto &pt : all_corners)
        avg += pt;
      avg *= 1.0f / all_corners.size();
      cv::circle(debug_img, avg, 5, cv::Scalar(0, 255, 0), -1);
    }

    auto latency = std::chrono::duration_cast<std::chrono::microseconds>(
                       now - target->timestamp)
                       .count() /
                   1000.0;
    auto latency_img_target =
        std::chrono::duration_cast<std::chrono::microseconds>(
            src_img.timestamp - target->timestamp)
            .count() /
        1000.0;

    cv::putText(debug_img,
                fmt::format("Img-Frame Delay: {:.2f}ms", latency_img_target),
                cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(255, 255, 255), 2);
  }

  int baseline = 0;
  if (state.has_value()) {
    std::string state_str;
    switch (state.value()) {
    case Tracker::LOST:
      state_str = "LOST";
      break;
    case Tracker::DETECTING:
      state_str = "DETECTING";
      break;
    case Tracker::TRACKING:
      state_str = "TRACKING";
      break;
    case Tracker::TEMP_LOST:
      state_str = "TEMP_LOST";
      break;
    default:
      state_str = "UNKNOWN";
      break;
    }
    cv::Size state_size =
        cv::getTextSize(state_str, cv::FONT_HERSHEY_SIMPLEX, 2.5, 2, &baseline);
    int x = std::max(0, debug_img.cols - state_size.width - 10);
    int y = std::min(debug_img.rows - 1, state_size.height + 10);
    cv::putText(debug_img, state_str, {x, y}, cv::FONT_HERSHEY_SIMPLEX, 2.5,
                cv::Scalar(0, 0, 255), 2);
  }

  if (target) {
    auto armorName = [](ArmorNumber num) {
      switch (num) {
      case ArmorNumber::SENTRY:
        return "SENTRY";
      case ArmorNumber::BASE:
        return "BASE";
      case ArmorNumber::OUTPOST:
        return "OUTPOST";
      case ArmorNumber::NO1:
        return "NO1";
      case ArmorNumber::NO2:
        return "NO2";
      case ArmorNumber::NO3:
        return "NO3";
      case ArmorNumber::NO4:
        return "NO4";
      case ArmorNumber::NO5:
        return "NO5";
      default:
        return "UNKNOWN";
      }
    };
    std::string id_str = fmt::format("Attack: {}", armorName(target->id));
    cv::Size id_size =
        cv::getTextSize(id_str, cv::FONT_HERSHEY_SIMPLEX, 1.6, 2, &baseline);
    int x = std::max(0, debug_img.cols - id_size.width - 10);
    int y = std::min(debug_img.rows - 1, 100);
    cv::putText(debug_img, id_str, {x, y}, cv::FONT_HERSHEY_SIMPLEX, 1.6,
                cv::Scalar(255, 0, 255), 2);
  }
  std::string fire_str = gimbal_cmd && gimbal_cmd->fire_advice ? "Fire!" : "";
  cv::Size fire_size =
      cv::getTextSize(fire_str, cv::FONT_HERSHEY_SIMPLEX, 1.2, 2, &baseline);
  int fire_x = 1440 / 2 - fire_size.width - 10;
  int fire_y = 200;

  cv::putText(debug_img, fire_str, {fire_x, fire_y}, cv::FONT_HERSHEY_SIMPLEX,
              2.85, cv::Scalar(0, 0, 255), 2);

  if (gimbal_cmd.has_value()) {
    std::string gimbal_str = fmt::format(
        "Pitch: {:.2f}, Yaw: {:.2f}, Pitch_diff: {:.2f}, Yaw_diff: {:.2f}",
        gimbal_cmd->pitch, gimbal_cmd->yaw, gimbal_cmd->pitch_diff,
        gimbal_cmd->yaw_diff);
    cv::putText(debug_img, gimbal_str, {10, debug_img.rows - 30},
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
  }

  cv::circle(debug_img, cv::Point2i(1440 / 2, 1080 / 2), 5,
             cv::Scalar(255, 255, 255), 2);
  return debug_img;
  // cv::imshow("debug_overlay", debug_img);
  // cv::waitKey(1);
}

std::string formatTargetInfo(const Target &target) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);

  auto now = std::chrono::steady_clock::now();
  auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
                 now - target.timestamp)
                 .count();

  oss << "=== Target Info ===\n";
  oss << "Frame ID      : " << target.frame_id << "\n";
  oss << "Type          : " << target.type << "\n";
  oss << "Tracking      : " << (target.tracking ? "Yes" : "No") << "\n";
  oss << "ID            : " << static_cast<int>(target.id) << "\n";
  oss << "Armors Num    : " << target.armors_num << "\n";
  oss << "Timestamp Age : " << age << " ms ago\n";

  oss << "\n-- Position --\n";
  oss << "x: " << target.position_.x << ", y: " << target.position_.y
      << ", z: " << target.position_.z << "\n";

  oss << "\n-- Velocity --\n";
  oss << "vx: " << target.velocity_.x << ", vy: " << target.velocity_.y
      << ", vz: " << target.velocity_.z << "\n";

  oss << "\n-- Yaw Info --\n";
  oss << "Yaw      : " << target.yaw << "\n";
  oss << "v_yaw    : " << target.v_yaw << "\n";
  oss << "Yaw Diff : " << target.yaw_diff << "\n";

  oss << "\n-- Radii  --\n";
  oss << "Radius 1       : " << target.radius_1 << "\n";
  oss << "Radius 2       : " << target.radius_2 << "\n";
  oss << "d_za           : " << target.d_za << "\n";
  oss << "d_zc           : " << target.d_zc << "\n";
  oss << "Position Diff  : " << target.position_diff << "\n";
  oss << "z_diff         : " << std::abs(target.d_za) + std::abs(target.d_zc)
      << "\n";

  return oss.str();
}
void dumpTargetToFile(const Target &target, const std::string &path) {
  std::ofstream file(path);
  if (file.is_open()) {
    file << formatTargetInfo(target);
    file.close();
  }
}

std::string formatImuInfo(const ReceiveImuData &imu) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3); // 设置输出精度

  // 静态变量用于统计帧率
  static int frame_count = 0;
  static double fps = 0.0;
  static auto last_time = std::chrono::steady_clock::now();

  // 每帧计数
  ++frame_count;

  // 时间间隔
  auto now = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(now - last_time).count();

  if (elapsed >= 1.0) {
    fps = frame_count / elapsed;
    frame_count = 0;
    last_time = now;
  }

  oss << "=== IMU Info ===\n";
  oss << "Timestamp     : " << imu.time_stamp << "\n";

  oss << "\n-- Orientation (rad) --\n";
  oss << "Yaw   : " << imu.data.yaw << "\n";
  oss << "Pitch : " << imu.data.pitch << "\n";
  oss << "Roll  : " << imu.data.roll << "\n";

  oss << "\n-- Angular Velocity (rad/s) --\n";
  oss << "Yaw_vel   : " << imu.data.yaw_vel << "\n";
  oss << "Pitch_vel : " << imu.data.pitch_vel << "\n";
  oss << "Roll_vel  : " << imu.data.roll_vel << "\n";

  oss << "\n-- Orientation (deg) --\n";
  oss << "Yaw   : " << imu.data.yaw * 180 / M_PI << "\n";
  oss << "Pitch : " << imu.data.pitch * 180 / M_PI << "\n";
  oss << "Roll  : " << imu.data.roll * 180 / M_PI << "\n";

  oss << "\nCRC           : 0x" << std::hex << imu.crc << std::dec << "\n";

  // 显示最近一次统计得到的 FPS
  oss << "Frame Rate (FPS): " << std::setprecision(1) << fps << "\n";

  return oss.str();
}

void dumpImuToFile(const ReceiveImuData &imu, const std::string &path) {
  std::ofstream file(path);
  if (file.is_open()) {
    file << formatImuInfo(imu);
    file.close();
  }
}
std::string formatAimInfo(const ReceiveAimINFO &aim) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3);

  // 静态变量用于帧率统计
  static int frame_count = 0;
  static double fps = 0.0;
  static auto last_time = std::chrono::steady_clock::now();

  ++frame_count;
  auto now = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(now - last_time).count();

  if (elapsed >= 1.0) {
    fps = frame_count / elapsed;
    frame_count = 0;
    last_time = now;
  }

  oss << "=== Aim Info ===\n";
  oss << "Timestamp     : " << aim.time_stamp << "\n";

  oss << "\n-- Orientation (rad) --\n";
  oss << "Yaw   : " << aim.yaw << "\n";
  oss << "Pitch : " << aim.pitch << "\n";
  oss << "Roll  : " << aim.roll << "\n";

  oss << "\n-- Angular Velocity (rad/s) --\n";
  oss << "Yaw_vel   : " << aim.yaw_vel << "\n";
  oss << "Pitch_vel : " << aim.pitch_vel << "\n";
  oss << "Roll_vel  : " << aim.roll_vel << "\n";

  oss << "\n-- Orientation (deg) --\n";
  oss << "Yaw   : " << aim.yaw * 180 / M_PI << "\n";
  oss << "Pitch : " << aim.pitch * 180 / M_PI << "\n";
  oss << "Roll  : " << aim.roll * 180 / M_PI << "\n";

  oss << "\n-- System Info --\n";
  oss << "Bullet Speed     : " << aim.bullet_speed << " m/s\n";
  oss << "Controller Delay : " << aim.controller_delay << " s\n";
  oss << "Detect Color     : " << (aim.detect_color == 0 ? "Red" : "Blue")
      << "\n";

  oss << "Frame Rate (FPS) : " << std::setprecision(1) << fps << "\n";

  return oss.str();
}
void dumpAimToFile(const ReceiveAimINFO &aim, const std::string &path) {
  std::ofstream file(path);
  if (file.is_open()) {
    file << formatAimInfo(aim);
    file.close();
  }
}
void write_aim_log_to_json(const ReceiveAimINFO &aim) {
  nlohmann::json j;

  j["timestamp"] = aim.time_stamp;
  j["yaw"] = aim.yaw;
  j["pitch"] = aim.pitch;
  j["roll"] = aim.roll;

  j["yaw_vel"] = aim.yaw_vel;
  j["pitch_vel"] = aim.pitch_vel;
  j["roll_vel"] = aim.roll_vel;

  j["yaw_deg"] = aim.yaw * 180.0 / M_PI;
  j["pitch_deg"] = aim.pitch * 180.0 / M_PI;
  j["roll_deg"] = aim.roll * 180.0 / M_PI;

  j["bullet_speed"] = aim.bullet_speed;
  j["controller_delay"] = aim.controller_delay;
  j["detect_color"] = (aim.detect_color == 0 ? "Red" : "Blue");

  // FPS 统计
  static int frame_count = 0;
  static double fps = 0.0;
  static auto last_time = std::chrono::steady_clock::now();

  ++frame_count;
  auto now = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(now - last_time).count();
  if (elapsed >= 1.0) {
    fps = frame_count / elapsed;
    frame_count = 0;
    last_time = now;
  }
  j["fps"] = fps;

  std::ofstream file("/dev/shm/aim_log.json");
  if (file.is_open()) {
    file << j.dump(2);
  }
}
void write_target_log_to_json(const Target &target) {
  nlohmann::json j;

  j["frame_id"] = target.frame_id;
  j["type"] = target.type;
  j["tracking"] = target.tracking;
  j["id"] = static_cast<int>(target.id);
  j["armors_num"] = target.armors_num;

  auto now = std::chrono::steady_clock::now();
  auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - target.timestamp)
                    .count();
  j["timestamp_age_ms"] = age_ms;

  j["position"] = {{"x", target.position_.x},
                   {"y", target.position_.y},
                   {"z", target.position_.z}};

  j["velocity"] = {{"x", target.velocity_.x},
                   {"y", target.velocity_.y},
                   {"z", target.velocity_.z}};

  j["yaw"] = target.yaw;
  j["v_yaw"] = target.v_yaw;
  j["yaw_diff"] = target.yaw_diff;

  j["radius_1"] = target.radius_1;
  j["radius_2"] = target.radius_2;
  j["d_za"] = target.d_za;
  j["d_zc"] = target.d_zc;
  j["position_diff"] = target.position_diff;
  j["z_diff"] = std::abs(target.d_za) + std::abs(target.d_zc);

  std::ofstream file("/dev/shm/target_log.json");
  if (file.is_open()) {
    file << j.dump(2);
  }
}
void drawRune(cv::Mat &src_img, const std::vector<RuneObject> &objs,
              std::chrono::steady_clock::time_point timestamp) {

  static auto last_show_time = std::chrono::steady_clock::now();
  static bool window_initialized;
  auto now = std::chrono::steady_clock::now();
  constexpr double min_interval_ms = 1000.0 / 60.0;
  double elapsed_ms =
      std::chrono::duration<double, std::milli>(now - last_show_time).count();
  if (elapsed_ms < min_interval_ms)
  {
    return;
  }


  if (!window_initialized) {
      cv::namedWindow("debug_rune", cv::WINDOW_NORMAL);
      cv::resizeWindow("debug_rune", debug_w, debug_h);
  
      window_initialized = true;
    }
  last_show_time = now;
  cv::Mat debug_img;
  src_img.convertTo(debug_img, -1, 1, 0);
  cv::cvtColor(debug_img, debug_img, cv::COLOR_BGR2RGB);
  
  for (const auto &obj : objs) {
    auto pts = obj.pts.toVector2f();
    // 计算中心点，这里你原代码是从 pts.begin()+1 到 pts.end()
    // 累加后除以4，感觉更合理的是平均所有点或明确写个除数
    cv::Point2f aim_point =
        std::accumulate(pts.begin() + 1, pts.end(), cv::Point2f(0, 0)) / 4.0f;

    cv::Scalar line_color = obj.type == RuneType::INACTIVATED
                                ? cv::Scalar(50, 255, 50)
                                : cv::Scalar(255, 50, 255);

    cv::putText(debug_img, fmt::format("{:.2f}", obj.prob), cv::Point2i(pts[1]),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2);
    cv::polylines(debug_img, obj.pts.toVector2i(), true, line_color, 2);
    cv::circle(debug_img, aim_point, 5, line_color, -1);

    std::string rune_type = obj.type == RuneType::INACTIVATED ? "_HIT" : "_OK";
    std::string rune_color = enemyColorToString(obj.color);
    cv::putText(debug_img, rune_color + rune_type, cv::Point2i(pts[2]),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2);
  }

  double latency =
      std::chrono::duration<double, std::milli>(now - timestamp).count();
  std::string latency_str = fmt::format("Latency: {:.2f}ms", latency);
  cv::putText(debug_img, latency_str, cv::Point2i(10, 30),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

  cv::imshow("debug_rune", debug_img);
  cv::waitKey(1);
}
