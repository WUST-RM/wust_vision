#pragma once
#include "detect/mono_measure_tool.hpp"
#include "opencv2/opencv.hpp"
#include "type/type.hpp"
#include "common/tf.hpp"
#include "tracker/tracker.hpp"
#include <opencv2/core/mat.hpp>
void drawresult(const cv::Mat &src_img, const std::vector<ArmorObject> &objs,int64_t timestamp_nanosec);
void drawresult(const imgframe &src_img, const Armors &armors);
void drawreprojec(const cv::Mat &src_img, const std::vector<std::vector<cv::Point2f>> all_pts,const Target target,const Tracker::State state);
void drawreprojec(const cv::Mat &src_img, const Target_info target_info,const Target target,const Tracker::State state);
void drawreprojec(const imgframe &src_img, const Target_info target_info,const Target target,const Tracker::State state);
//cv::Mat drawreprojec(const imgframe &src_img, const Target_info target_info,const Target target,const Tracker::State state);
void dumpTargetToFile(const Target& target, const std::string& path = "/tmp/target_status.txt") ;

void updatePlot(const std::vector<Armor>& armors);