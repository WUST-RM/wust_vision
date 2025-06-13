#include "driver/video_player.hpp"
#include <chrono>
#include <iostream>
#include <thread>

VideoPlayer::VideoPlayer(const std::string &video_path, int frame_rate,
                         int start_frame, bool loop)
    : path_(video_path), frame_rate_(frame_rate), start_frame_(start_frame),
      loop_(loop), running_(false) {}

void VideoPlayer::setCallback(FrameCallback cb) {
  on_frame_callback_ = std::move(cb);
}

bool VideoPlayer::start() {
  cap_.open(path_);
  if (!cap_.isOpened()) {
    std::cerr << "Failed to open video: " << path_ << std::endl;
    return false;
  }

  cap_.set(cv::CAP_PROP_POS_FRAMES, start_frame_);
  running_ = true;
  worker_ = std::thread(&VideoPlayer::run, this);
  return true;
}

void VideoPlayer::stop() {
  running_ = false;
  if (worker_.joinable()) {
    worker_.join();
  }
  cap_.release();
}

VideoPlayer::~VideoPlayer() { stop(); }

void VideoPlayer::run() {
  const int wait_ms = static_cast<int>(1000.0 / frame_rate_);

  while (running_) {
    cv::Mat frame_bgr;
    cap_ >> frame_bgr;
    if (frame_bgr.empty()) {
      if (loop_) {
        cap_.set(cv::CAP_PROP_POS_FRAMES, start_frame_);
        continue;
      } else {
        break;
      }
    }

    ImageFrame frame;
    frame.width = frame_bgr.cols;
    frame.height = frame_bgr.rows;
    frame.step = frame.width * 3;
    frame.data.assign(frame_bgr.data,
                      frame_bgr.data + frame.step * frame.height);
    frame.timestamp = std::chrono::steady_clock::now();

    if (on_frame_callback_) {
      on_frame_callback_(frame);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
  }
}
