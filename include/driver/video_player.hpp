#pragma once
#include "type/image.hpp"
#include <atomic>
#include <functional>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

class VideoPlayer {
public:
  using FrameCallback = std::function<void(const ImageFrame &)>;

  VideoPlayer(const std::string &video_path, int frame_rate = 30,
              int start_frame = 0, bool loop = true);

  void setCallback(FrameCallback cb);
  bool start();
  void stop();
  ~VideoPlayer();

private:
  void run(); // 后台线程函数

  std::string path_;
  int frame_rate_;
  int start_frame_;
  bool loop_;
  std::atomic<bool> running_;
  cv::VideoCapture cap_;
  std::thread worker_;
  FrameCallback on_frame_callback_;
};
