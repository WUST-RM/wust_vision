#ifndef IMAGE_HPP_
#define IMAGE_HPP_
#include "MvCameraControl.h"
#include "opencv2/opencv.hpp"

struct ImageFrame {
  std::vector<uint8_t> data;
  int width;
  int height;
  int step;
  std::chrono::steady_clock::time_point timestamp;
};
// struct ImageFrame {
//     int width = 0, height = 0, step = 0;
//     std::vector<uint8_t> data;
//     std::chrono::steady_clock::time_point timestamp;

//     // 新增原始图像信息
//     unsigned char* raw_data = nullptr;
//     int raw_len = 0;

//     MvGvspPixelType         pixel_type  = PixelType_Gvsp_RGB8_Packed;
// };

inline cv::Mat convertToMat(const ImageFrame &frame) {
  if (frame.data.empty()) {
    return cv::Mat();
  }
  cv::Mat rgb(frame.height, frame.width, CV_8UC3);
  memcpy(rgb.data, frame.data.data(), frame.height * frame.step);
  return rgb;
}
inline cv::Mat convertToMatrgb(const ImageFrame &frame) {
  if (frame.data.empty()) {
    return cv::Mat();
  }
  cv::Mat img(frame.height, frame.width, CV_8UC3);
  memcpy(img.data, frame.data.data(), frame.height * frame.step);
  return img;
}
inline cv::Mat convertToMatbgr(const ImageFrame &frame) {
  if (frame.data.empty()) {
    return cv::Mat();
  }
  cv::Mat img(frame.height, frame.width, CV_8UC3);
  memcpy(img.data, frame.data.data(), frame.height * frame.step);

  cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
  return img;
}
// inline cv::Mat convertToMat(const ImageFrame& frame) {
//     if (frame.data.empty()) {
//         return cv::Mat();
//     }
//     // 1) 直接用 frame.data 作为 Mat 的数据指针，不再 memcpy
//     // 2) 指定 step（每行字节数）以支持不同行宽或 padding
//     cv::Mat rgb(
//         frame.height,
//         frame.width,
//         CV_8UC3,
//         const_cast<uint8_t*>(frame.data.data()),
//         frame.step
//     );

//     // 3) OpenCV 默认使用 BGR 顺序，若你的后续算法需要
//     BGR，直接在这里转换一次
//     // cv::Mat bgr;
//     // cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);

//     // 注意：这里不再 clone，bgr.data 依然指向 frame.data。
//     // 如果你希望 bgr 拥有自己的内存（frame.data 生命周期结束后依然有效），
//     // 可在返回前加上一句： return bgr.clone();
//     return rgb.clone();
// }

#endif