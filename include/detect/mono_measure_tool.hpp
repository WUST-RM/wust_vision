// Copyright 2023 Yunlong Feng
// Copyright 2025 Lihan Chen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ARMOR_DETECTOR_TENSORRT__MONO_MEASURE_TOOL_HPP_
#define ARMOR_DETECTOR_TENSORRT__MONO_MEASURE_TOOL_HPP_

#include <string>
#include <vector>

#include "common/tf.hpp"

#include "opencv2/opencv.hpp"
#include "type/type.hpp"
struct Target_info
{
  std::vector<std::vector<cv::Point2f>> pts;
  std::vector<Position> pos;
  std::vector<tf2::Quaternion> ori;
  int select_id;
};

class MonoMeasureTool
{
public:
  MonoMeasureTool() = default;
  MonoMeasureTool(const std::string &yaml_path);

  explicit MonoMeasureTool(
    std::vector<double> camera_intrinsic, std::vector<double> camera_distortion);

  /**
   * @brief Set the camera intrinsic parameter
   *
   * @param camera_intrinsic camera intrinsic in 3x3 matrix flat in line stretch
   * @param camera_distortion camera distortion parameter in plumb_bob distortion model
   * @return true
   * @return false
   */
  bool setCameraInfo(std::vector<double> camera_intrinsic, std::vector<double> camera_distortion);
  /**
   * @brief Solve Perspective-n-Point problem in camera
   * 3d点坐标求解（use solve pnp）
   * @param points2d a list of points in image frame
   * @param points3d a list of points correspondent to points2d
   * @param position output position of the origin point of 3d coordinate system
   * @return true
   * @return false
   */
  bool solvePnp(
    const std::vector<cv::Point2f> & points2d, const std::vector<cv::Point3f> & points3d,
    cv::Point3f & position, cv::Mat & rvec, cv::SolvePnPMethod pnp_method = cv::SOLVEPNP_ITERATIVE);
  /**
   * @brief 逆投影，已知深度，2d->3d点求解
   *
   * @param p 图像上点坐标
   * @param distance 已知的真实距离
   * @return cv::Point3f 对应的真实3d点坐标
   */
  cv::Point3f unproject(cv::Point2f p, double distance);
  /**
   * @brief 视角求解
   *
   * @param p 图像上点坐标
   * @param pitch 视角pitch
   * @param yaw 视角yaw
   */
  void calcViewAngle(cv::Point2f p, float & pitch, float & yaw);

  /**
   * @brief 装甲板目标位姿求解
   *
   * @param obj 装甲板目标
   * @param position 返回的坐标
   * @param rvec 相对旋转向量
   * @return true
   * @return false
   */
  bool calcArmorTarget(
    const ArmorObject & obj, cv::Point3f & position, cv::Mat & rvec, std::string & armor_type);

  float calcDistanceToCenter(const ArmorObject & obj);
  
  bool reprojectArmorsCorners(
    Armors & armors,
    Target_info & target_info);

  
  bool reprojectArmorCorners(
    const Armor & armor,
    std::vector<cv::Point2f> & image_points);
    bool reprojectArmorCorners_raw(
      const Armor & armor,
      std::vector<cv::Point2f> & image_points);
  void processDetectedArmors(
        const std::vector<ArmorObject>& objs,
        int detect_color,
        Armors& armors_out);



    static std::vector<cv::Point3f> SMALL_ARMOR_3D_POINTS;
    static std::vector<cv::Point3f> BIG_ARMOR_3D_POINTS;
    static std::vector<cv::Point3f> SMALL_ARMOR_3D_POINTS_NET;
    static std::vector<cv::Point3f> BIG_ARMOR_3D_POINTS_NET;
    cv::Mat camera_intrinsic_;   // 相机内参3*3
    cv::Mat camera_distortion_;  // 相机畸变参数1*5

private:
  // 相机参数
  cv::Mat prev_rvec_, prev_tvec_;
  bool has_prev_{false};
  

  

  std::string mono_logger="mono_measure_tool";

  double fx_{0}, fy_{0}, u0_{0}, v0_{0};

};



#endif  // ARMOR_DETECTOR_TENSORRT__MONO_MEASURE_TOOL_HPP_
