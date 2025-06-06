#ifndef HIK_HPP
#define HIK_HPP

#include "MvCameraControl.h"
#include "common/ThreadPool.h"
#include "type/image.hpp"
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <opencv2/videoio.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

class HikCamera {
public:
  HikCamera();
  ~HikCamera();
  void setFrameCallback(std::function<void(const ImageFrame &)> cb) {
    on_frame_callback_ = std::move(cb);
  }

  bool initializeCamera(const std::string &video_path);
  void setParameters(double acquisition_frame_rate, double exposure_time,
                     double gain, const std::string &adc_bit_depth,
                     const std::string &pixel_format);
  void startCamera();
  bool restartCamera();
  void stopCamera();

private:
  void hikCaptureLoop();
  void videoCaptureLoop();

  void *camera_handle_;
  int fail_count_;
  MV_IMAGE_BASIC_INFO img_info_;
  MV_CC_PIXEL_CONVERT_PARAM convert_param_;
  std::thread capture_thread_;
  std::string hik_logger = "hik_camera";
  double last_frame_rate_, last_exposure_time_, last_gain_;
  std::string last_adc_bit_depth_, last_pixel_format_;
  bool in_low_frame_rate_state_;
  std::chrono::steady_clock::time_point low_frame_rate_start_time_;
  std::atomic<bool> stop_signal_{false};
  bool is_video_mode_;
  cv::VideoCapture video_cap_;
  int video_fps_;
  int expected_width_ = 0;
  int expected_height_ = 0;
  std::function<void(const ImageFrame &)> on_frame_callback_;
};

#endif // HIK_HPP