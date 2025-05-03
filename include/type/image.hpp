#ifndef IMAGE_HPP_
#define IMAGE_HPP_
#include "opencv2/opencv.hpp"

struct ImageFrame {
    std::vector<uint8_t> data;
    int width;
    int height;
    int step;
    std::chrono::steady_clock::time_point timestamp;
};

// inline cv::Mat convertToMat(const ImageFrame& frame) {
//     // 注意，这里 frame.step 通常是 width * 3（RGB）
//     cv::Mat rgb(frame.height, frame.width, CV_8UC3, const_cast<uint8_t*>(frame.data.data()), frame.step);
//     // cv::Mat bgr;
//     // cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
//     return rgb;
// }
inline cv::Mat convertToMat(const ImageFrame& frame) {
    cv::Mat rgb(frame.height, frame.width, CV_8UC3);
    memcpy(rgb.data, frame.data.data(), frame.height * frame.step);
    return rgb;
}


#endif
