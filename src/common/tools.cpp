#include "common/tools.hpp"
#include "fmt/format.h"
#include <chrono>
#include <opencv2/core/types.hpp>
#include <vector>
#include "common/tf.hpp"
#include "detect/mono_measure_tool.hpp"
#include "tracker/tracker.hpp"
#include "type/type.hpp"
void drawresult(const cv::Mat &src_img, const std::vector<ArmorObject> &objs, int64_t timestamp_nanosec)
{
    static auto last_show_time = std::chrono::steady_clock::now();
    static bool window_initialized = false;
    static int brightness_slider = 200;  

    if (!window_initialized) {
        cv::namedWindow("debug", cv::WINDOW_NORMAL); 
        cv::resizeWindow("debug", 640, 480); 
        cv::createTrackbar("Brightness", "debug", &brightness_slider, 400); 
        window_initialized = true;
    }

    auto now = std::chrono::steady_clock::now();
    constexpr double min_interval_ms = 1000.0 / 60.0;

    double elapsed_ms = std::chrono::duration<double, std::milli>(now - last_show_time).count();
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
            cv::line(debug_img, obj.pts[i], obj.pts[(i + 1) % 4], cv::Scalar(48, 48, 255), 1);
            if (obj.is_ok) {
                cv::line(debug_img, obj.pts_binary[i], obj.pts_binary[next_indices[i]], cv::Scalar(0, 255, 0), 1);
            }
        }

        std::string armor_color;
        switch (obj.color) {
            case ArmorColor::BLUE:   armor_color = "B"; break;
            case ArmorColor::RED:    armor_color = "R"; break;
            case ArmorColor::NONE:   armor_color = "N"; break;
            case ArmorColor::PURPLE: armor_color = "P"; break;
            default:                 armor_color = "UNKNOWN"; break;
        }

        std::string armor_key = fmt::format("{} {}", armor_color, static_cast<int>(obj.number));
        cv::putText(debug_img, armor_key, cv::Point2i(obj.pts[0]), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(255, 255, 0), 2);
    }
    cv::circle(
      debug_img, cv::Point2i(1440 / 2., 1080 / 2.), 5,
      cv::Scalar(0, 0, 255), 1);

    auto timestamp_tp = std::chrono::steady_clock::time_point() + std::chrono::nanoseconds(timestamp_nanosec);
    auto latency_nano = std::chrono::duration_cast<std::chrono::nanoseconds>(now - timestamp_tp).count();
    double latency_ms = static_cast<double>(latency_nano) / 1e6;

    std::string latency = fmt::format("Latency: {:.3f}ms", latency_ms);
    cv::putText(debug_img, latency, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

    cv::imshow("debug", debug_img);
    cv::waitKey(1);
}

void drawreprojec(const cv::Mat &src_img, const std::vector<std::vector<cv::Point2f>> all_pts,const Target target,const Tracker::State state)
{
    static auto last_show_time = std::chrono::steady_clock::now();
    static bool window_initialized = false;
    static int brightness_slider = 200;  
    static cv::Mat debug_img;
    if(src_img.empty())
    {
        return;
    }

    if (!window_initialized) {
        cv::namedWindow("debug", cv::WINDOW_NORMAL); 
        cv::resizeWindow("debug", 640, 480); 
        cv::createTrackbar("Brightness", "debug", &brightness_slider, 400); 
        window_initialized = true;
    }
    auto now = std::chrono::steady_clock::now();
    constexpr double min_interval_ms = 1000.0 / 60.0;

    double elapsed_ms = std::chrono::duration<double, std::milli>(now - last_show_time).count();
    if (elapsed_ms < min_interval_ms) {
        return;
    }
    last_show_time = now;



    // 调整亮度
    
    double brightness_factor = brightness_slider / 100.0;  
    src_img.convertTo(debug_img, -1, brightness_factor, 0);  

    cv::cvtColor(debug_img, debug_img, cv::COLOR_BGR2RGB);

    static const int next_indices[] = {2, 0, 3, 1};
    for (auto &pts : all_pts) {
        for (size_t i = 0; i < 4; ++i) {
            cv::line(debug_img, pts[i], pts[(i + 1) % 4], cv::Scalar(48, 48, 255), 1);
           
        }

      
    }
    cv::circle(
      debug_img, cv::Point2i(1440 / 2., 1080 / 2.), 5,
      cv::Scalar(255,255, 255), 2);

      auto latency_duration = now - target.timestamp;
      auto latency_ms = std::chrono::duration_cast<std::chrono::microseconds>(latency_duration).count() / 1000.0;
  
      std::string latency = fmt::format("Latency: {:.3f}ms", latency_ms);
      cv::putText(debug_img, latency, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(100, 255, 0), 2);
      std::string state_str;
    switch (state) {
        case Tracker::LOST:        state_str = "LOST"; break;
        case Tracker::DETECTING:   state_str = "DETECTING"; break;
        case Tracker::TRACKING:    state_str = "TRACKING"; break;
        case Tracker::TEMP_LOST:   state_str = "TEMP_LOST"; break;
        default:                   state_str = "UNKNOWN"; break;
    }
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(state_str, cv::FONT_HERSHEY_SIMPLEX, 2.8, 2, &baseline);
    cv::Point text_org(debug_img.cols - text_size.width - 10, text_size.height + 10);
    cv::putText(debug_img, state_str, text_org, cv::FONT_HERSHEY_SIMPLEX, 2.8, cv::Scalar(0, 0, 255), 2);

   
    cv::imshow("debug", debug_img);
    cv::waitKey(1);
}

void drawreprojec(const cv::Mat &src_img, const Target_info target_info,const Target target,const Tracker::State state)
{
    static auto last_show_time = std::chrono::steady_clock::now();
    static bool window_initialized = false;
    static int brightness_slider = 200;  
    static cv::Mat debug_img;
    if(src_img.empty())
    {
        return;
    }

    if (!window_initialized) {
        cv::namedWindow("debug", cv::WINDOW_NORMAL); 
        cv::resizeWindow("debug", 640, 480); 
        cv::createTrackbar("Brightness", "debug", &brightness_slider, 400); 
        window_initialized = true;
    }
    auto now = std::chrono::steady_clock::now();
    constexpr double min_interval_ms = 1000.0 / 60.0;

    double elapsed_ms = std::chrono::duration<double, std::milli>(now - last_show_time).count();
    if (elapsed_ms < min_interval_ms) {
        return;
    }
    last_show_time = now;

    double brightness_factor = brightness_slider / 100.0;  
    src_img.convertTo(debug_img, -1, brightness_factor, 0);  

    cv::cvtColor(debug_img, debug_img, cv::COLOR_BGR2RGB);

    static const int next_indices[] = {2, 0, 3, 1};
    for (size_t i = 0; i < target_info.pts.size(); ++i) {
        const auto &pts = target_info.pts[i];
        const auto &position = target_info.pos[i];
        const auto &orientation = target_info.ori[i];
    
        for (size_t j = 0; j < pts.size(); ++j) {
            cv::line(debug_img, pts[j], pts[(j + 1) % 4], cv::Scalar(255, 0, 0), 2);
        }
    
        // 计算 yaw
        double yaw = getYawFromQuaternion(orientation);
        double distance = std::sqrt(position.x * position.x + position.y * position.y + position.z * position.z);
    
        std::vector<std::string> info_lines = {
            fmt::format("Dis: {:.3f}", distance*100),
            fmt::format("X: {:.3f}", position.x),
            fmt::format("Y: {:.3f}", position.y),
            fmt::format("Z: {:.3f}", position.z),
            fmt::format("Yaw: {:.3f}", yaw * 180.0 / M_PI)
        };
        cv::Point2f text_org = pts[0] + cv::Point2f(0, 200);
        for (int k = 0; k < info_lines.size(); ++k) {
            cv::putText(debug_img, info_lines[k], text_org + cv::Point2f(0, -10 - 20 * k),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(50, 255, 255), 1);
        }
    }
    
    cv::circle(
      debug_img, cv::Point2i(1440 / 2., 1080 / 2.), 5,
      cv::Scalar(255,255, 255), 2);

      auto latency_duration = now - target.timestamp;
      auto latency_ms = std::chrono::duration_cast<std::chrono::microseconds>(latency_duration).count() / 1000.0;
  
      std::string latency = fmt::format("Latency: {:.3f}ms", latency_ms);
      cv::putText(debug_img, latency, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
     
      int baseline = 0;

    
      std::string state_str;
      switch (state) {
          case Tracker::LOST:        state_str = "LOST"; break;
          case Tracker::DETECTING:   state_str = "DETECTING"; break;
          case Tracker::TRACKING:    state_str = "TRACKING"; break;
          case Tracker::TEMP_LOST:   state_str = "TEMP_LOST"; break;
          default:                   state_str = "UNKNOWN"; break;
      }
      cv::Size state_text_size = cv::getTextSize(state_str, cv::FONT_HERSHEY_SIMPLEX, 2.8, 2, &baseline);
      cv::Point state_text_org(debug_img.cols - state_text_size.width - 10, state_text_size.height + 10);
      cv::putText(debug_img, state_str, state_text_org, cv::FONT_HERSHEY_SIMPLEX, 2.8, cv::Scalar(0, 0, 255), 2);
      
      // 右上角再往下显示 armor_str（与 state_str 平行对齐）
      auto armorNumberToString = [](ArmorNumber num) -> std::string {
          switch (num) {
              case ArmorNumber::SENTRY:   return "SENTRY";
              case ArmorNumber::NO1:      return "NO1";
              case ArmorNumber::NO2:      return "NO2";
              case ArmorNumber::NO3:      return "NO3";
              case ArmorNumber::NO4:      return "NO4";
              case ArmorNumber::NO5:      return "NO5";
              case ArmorNumber::OUTPOST:  return "OUTPOST";
              case ArmorNumber::BASE:     return "BASE";
              default:                    return "UNKNOWN";
          }
      };
      std::string armor_str = "Attack: " + armorNumberToString(target.id);
      cv::Size armor_text_size = cv::getTextSize(armor_str, cv::FONT_HERSHEY_SIMPLEX, 1.6, 2, &baseline);
      cv::Point armor_text_org(debug_img.cols - armor_text_size.width - 10,
                               state_text_org.y + state_text_size.height + 20);
      cv::putText(debug_img, armor_str, armor_text_org, cv::FONT_HERSHEY_SIMPLEX, 1.6, cv::Scalar(255, 0, 255), 2);
      
    

   
    cv::imshow("debug", debug_img);
    cv::waitKey(1);
}