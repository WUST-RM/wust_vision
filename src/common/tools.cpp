#include "common/tools.hpp"
#include "fmt/format.h"
#include <chrono>
#include "common/tf.hpp"

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

