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


inline cv::Mat convertToMat(const ImageFrame& frame) {
    cv::Mat rgb(frame.height, frame.width, CV_8UC3);
    memcpy(rgb.data, frame.data.data(), frame.height * frame.step);
    return rgb;
}


#endif
