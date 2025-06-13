#pragma once

#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class Labeler {
public:
  Labeler();
  ~Labeler();

  void save(const cv::Mat &image, const std::vector<float> &data_row);

private:
  std::string image_dir_;
  std::string csv_dir_;
  std::string temp_csv_path_;

  std::ofstream csv_file_;
  int image_index_ = 1;
  int start_index_ = 1;
  bool csv_header_written_ = false;

  int getLastImageIndex() const;
};
