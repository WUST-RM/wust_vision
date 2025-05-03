#pragma once
#include "detect/mono_measure_tool.hpp"
#include "opencv2/opencv.hpp"
#include "type/type.hpp"
#include "common/tf.hpp"
#include "tracker/tracker.hpp"
void drawresult(const cv::Mat &src_img, const std::vector<ArmorObject> &objs,int64_t timestamp_nanosec);
void drawreprojec(const cv::Mat &src_img, const std::vector<std::vector<cv::Point2f>> all_pts,const Target target,const Tracker::State state);
void drawreprojec(const cv::Mat &src_img, const Target_info target_info,const Target target,const Tracker::State state);

