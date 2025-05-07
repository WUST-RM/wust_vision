
#pragma once

#include "opencv2/opencv.hpp"
#include <numeric>
#include <string>
#include <fmt/core.h>
#include "common/tf.hpp"
struct Light : public cv::RotatedRect {
    Light() = default;
    explicit Light(const std::vector<cv::Point> &contour)
    : cv::RotatedRect(cv::minAreaRect(contour)) {
     
  
      center = std::accumulate(
        contour.begin(),
        contour.end(),
        cv::Point2f(0, 0),
        [n = static_cast<float>(contour.size())](const cv::Point2f &a, const cv::Point &b) {
          return a + cv::Point2f(b.x, b.y) / n;
        });
  
      cv::Point2f p[4];
      this->points(p);
      std::sort(p, p + 4, [](const cv::Point2f &a, const cv::Point2f &b) { return a.y < b.y; });
      top = (p[0] + p[1]) / 2;
      bottom = (p[2] + p[3]) / 2;


  
      length = cv::norm(top - bottom);
      width = cv::norm(p[0] - p[1]);
  
      axis = top - bottom;
      axis = axis / cv::norm(axis);
  
      // Calculate the tilt angle
      // The angle is the angle between the light bar and the horizontal line
      tilt_angle = std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y));
      tilt_angle = tilt_angle / CV_PI * 180;
    }
   
    cv::Point2f top, bottom, center;
    cv::Point2f axis;
    double length;
    double width;
    float tilt_angle;
  };
  struct LightParams {
    // width / height
    double min_ratio;
    double max_ratio;
    // vertical angle
    double max_angle;
   
  };
struct GridAndStride
{
  int grid0;
  int grid1;
  int stride;
};

enum class ArmorColor { BLUE = 0, RED, NONE, PURPLE };

enum class ArmorNumber { SENTRY = 0, NO1, NO2, NO3, NO4, NO5, OUTPOST, BASE ,UNKNOWN};
template <>
struct fmt::formatter<ArmorNumber> {
    constexpr auto parse(fmt::format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(ArmorNumber num, FormatContext& ctx) -> decltype(ctx.out()) {
        const char* names[] = {"SENTRY", "NO1", "NO2", "NO3", "NO4", "NO5", "OUTPOST", "BASE"};
        int index = static_cast<int>(num);
        if (index >= 0 && index < sizeof(names) / sizeof(names[0])) {
            return fmt::format_to(ctx.out(), "{}", names[index]);
        } else {
            return fmt::format_to(ctx.out(), "UNKNOWN");
        }
      }
    };

typedef struct ArmorObject
{
  ArmorColor color;
  ArmorNumber number;
  float prob;
  std::vector<cv::Point2f> pts;
  std::vector<cv::Point2f> pts_binary;
  cv::Rect box;


  cv::Mat number_img;  

  double confidence;   

  cv::Mat whole_binary_img;
  cv::Mat whole_rgb_img;
  cv::Mat whole_gray_img;

  std::vector<Light> lights;

  double new_x;
  double new_y;
  bool is_ok =false;

  //std::unique_ptr<LightCornerCorrector> corner_corrector;

} ArmorObject;

constexpr const char * K_ARMOR_NAMES[] = {"guard", "1", "2", "3", "4", "5", "outpost", "base"};


struct Armor
{ 
  ArmorNumber number;
  std::string type;
  Position pos;
  tf2::Quaternion ori;
  Position target_pos;
  tf2::Quaternion target_ori;
  float distance_to_image_center;

};
struct Armors
{
  std::vector<Armor> armors;
  std::chrono::steady_clock::time_point timestamp;
  std::string frame_id;
};
struct Target
{ std::chrono::steady_clock::time_point timestamp;
  std::string frame_id;
  std::string type;
  bool tracking=false;
  ArmorNumber id=ArmorNumber::UNKNOWN;
  int armors_num=0;


  Position position_;
  Position velocity_;
  float yaw;
  float v_yaw;
  float radius_1;
  float radius_2;
  float d_za;
  float d_zc;
  // float yaw_diff;
  // float position_diff;

  void clear() {
    id = ArmorNumber::UNKNOWN;
    tracking = false;
    armors_num = 0;
    type = "";
    position_ = Position();  
    velocity_ = Position(); 
    yaw = 0.0;
    v_yaw = 0.0;
    radius_1 = 0.0;
    radius_2 = 0.0;
    d_zc = 0.0;
    d_za = 0.0;
    // yaw_diff = 0.0;
    // position_diff = 0.0;
}

};
struct imgframe {
  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
};
struct SyncedData {
  Target target;
  imgframe image;
};