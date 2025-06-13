#include "driver/labeler.hpp"
#include <filesystem>
#include <iostream>
#include <pwd.h>
#include <regex>
#include <stdexcept>
#include <unistd.h>
Labeler::Labeler() {
  const char *home = nullptr;

  // 尝试从 SUDO_USER 获取真实用户 home
  const char *sudo_user = std::getenv("SUDO_USER");
  if (sudo_user) {
    struct passwd *pw = getpwnam(sudo_user);
    if (pw) {
      home = pw->pw_dir;
    }
  }

  // 如果不是 sudo，使用 getuid 获取 home
  if (!home) {
    struct passwd *pw = getpwuid(getuid());
    if (pw) {
      home = pw->pw_dir;
    }
  }

  if (!home) {
    throw std::runtime_error("HOME environment variable not set.");
  }

  namespace fs = std::filesystem;
  image_dir_ = fs::path(home) / "wust_data/images";
  csv_dir_ = fs::path(home) / "wust_data/labels";

  fs::create_directories(image_dir_);
  fs::create_directories(csv_dir_);

  image_index_ = getLastImageIndex() + 1;
  start_index_ = image_index_;

  std::cout << "Labeler initialized.\nImage dir: " << image_dir_
            << "\nCSV dir: " << csv_dir_ << "\nStart index: " << start_index_
            << std::endl;
}

Labeler::~Labeler() {
  if (csv_file_.is_open()) {
    csv_file_.close();

    std::ostringstream new_name;
    new_name << std::setw(6) << std::setfill('0') << start_index_ << "-"
             << std::setw(6) << std::setfill('0') << (image_index_ - 1)
             << ".csv";
    std::string new_path = csv_dir_ + "/" + new_name.str();

    std::filesystem::rename(temp_csv_path_, new_path);
  }
}

void Labeler::save(const cv::Mat &image, const std::vector<float> &data_row) {
  if (!csv_file_.is_open()) {
    temp_csv_path_ = csv_dir_ + "/temp.csv";
    csv_file_.open(temp_csv_path_, std::ios::out);
  }

  if (!csv_header_written_) {
    csv_file_ << "index,pts0x,pts0y,pts1x,pts1y,pts2x,pts2y,pts3x,pts3y,number,"
                 "color\n";
    csv_header_written_ = true;
  }

  std::ostringstream img_name;
  img_name << std::setw(6) << std::setfill('0') << image_index_ << ".jpg";
  std::string img_path = image_dir_ + "/" + img_name.str();
  cv::imwrite(img_path, image);

  csv_file_ << image_index_;
  for (float val : data_row) {
    csv_file_ << "," << val;
  }
  csv_file_ << "\n";
  csv_file_.flush();

  ++image_index_;
}
int Labeler::getLastImageIndex() const {
  int max_index = -1;
  std::regex pattern(R"((\d{6})\.jpg)");

  namespace fs = std::filesystem;
  for (const auto &entry : fs::directory_iterator(image_dir_)) {
    if (!entry.is_regular_file())
      continue;

    std::smatch match;
    std::string filename = entry.path().filename().string();
    if (std::regex_match(filename, match, pattern)) {
      int index = std::stoi(match[1]);
      if (index > max_index) {
        max_index = index;
      }
    }
  }

  return max_index;
}
