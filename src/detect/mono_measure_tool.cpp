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

#include "detect/mono_measure_tool.hpp"
#include "common/logger.hpp"
#include <yaml-cpp/yaml.h>
#include <numeric>
#include <cmath>

std::vector<cv::Point3f> MonoMeasureTool::SMALL_ARMOR_3D_POINTS = {
  {0, 0.025, -0.066},
  {0, -0.025, -0.066},
  {0, -0.025, 0.066},
  {0, 0.025, 0.066},
};

std::vector<cv::Point3f> MonoMeasureTool::BIG_ARMOR_3D_POINTS = {
  {0, 0.025, -0.1125},
  {0, -0.025, -0.1125},
  {0, -0.025, 0.1125},
  {0, 0.025, 0.1125},
};
std::vector<cv::Point3f> MonoMeasureTool::SMALL_ARMOR_3D_POINTS_NET = {
  {0, 0.027, -0.066},
  {0, -0.027, -0.066},
  {0, -0.027, 0.066},
  {0, 0.027, 0.066},
};

std::vector<cv::Point3f> MonoMeasureTool::BIG_ARMOR_3D_POINTS_NET = {
  {0, 0.027, -0.1125},
  {0, -0.027, -0.1125},
  {0, -0.027, 0.1125},
  {0, 0.027, 0.1125},
};

bool is_big_armor(const ArmorObject & obj)
{
  switch (obj.number) {
    case ArmorNumber::NO1:
    case ArmorNumber::BASE:
      return true;
    default:
      return false;
  }
}

MonoMeasureTool::MonoMeasureTool(
  std::vector<double> camera_intrinsic, std::vector<double> camera_distortion)
{
  setCameraInfo(camera_intrinsic, camera_distortion);
}

MonoMeasureTool::MonoMeasureTool(const std::string &yaml_path)
{
  YAML::Node config = YAML::LoadFile(yaml_path);

  std::vector<double> camera_k = config["camera_matrix"]["data"].as<std::vector<double>>();
  std::vector<double> camera_d = config["distortion_coefficients"]["data"].as<std::vector<double>>();

  if (!setCameraInfo(camera_k, camera_d)) {
    WUST_ERROR(mono_logger) << "Failed to set camera info from YAML file: " << yaml_path;
  }
}

bool MonoMeasureTool::setCameraInfo(
  std::vector<double> camera_intrinsic, std::vector<double> camera_distortion)
{
  if (camera_intrinsic.size() != 9) {
    WUST_ERROR(mono_logger) << "Camera intrinsic size != 9";
    return false;
  }

  cv::Mat camera_intrinsic_mat(camera_intrinsic, true);
  camera_intrinsic_mat = camera_intrinsic_mat.reshape(0, 3);
  camera_intrinsic_ = camera_intrinsic_mat.clone();

  cv::Mat camera_distortion_mat(camera_distortion, true);
  camera_distortion_mat = camera_distortion_mat.reshape(0, 1);
  camera_distortion_ = camera_distortion_mat.clone();

  fx_ = camera_intrinsic_.at<double>(0, 0);
  fy_ = camera_intrinsic_.at<double>(1, 1);
  u0_ = camera_intrinsic_.at<double>(0, 2);
  v0_ = camera_intrinsic_.at<double>(1, 2);

  return true;
}

bool MonoMeasureTool::solvePnp(
  const std::vector<cv::Point2f> & points2d, const std::vector<cv::Point3f> & points3d,
  cv::Point3f & position, cv::Mat & rvec, cv::SolvePnPMethod pnp_method)
{
  if (points2d.size() != points3d.size()) {
    //WUST_WARN(mono_logger) << "2D-3D point size mismatch: " << points2d.size() << " vs " << points3d.size();
    return false;
  }

  if (camera_intrinsic_.empty() || camera_distortion_.empty()) {
    //WUST_ERROR(mono_logger) << "Camera parameters not initialized.";
    return false;
  }

  cv::Mat trans = cv::Mat::zeros(3, 1, CV_64FC1);
  cv::Mat r;
  bool res = cv::solvePnP(
    points3d, points2d, camera_intrinsic_, camera_distortion_, r, trans, false, pnp_method);

  if (!res || !cv::checkRange(r) || !cv::checkRange(trans)) {
    //WUST_WARN(mono_logger) << "solvePnP failed or produced invalid result.";
    return false;
  }

  rvec = r.clone();
  position = cv::Point3f(trans);
  return true;
}
// refer to :http://www.cnblogs.com/singlex/p/pose_estimation_1_1.html
// 根据输入的参数将图像坐标转换到相机坐标中
// 输入为图像上的点坐标
// double distance 物距
// 输出3d点坐标的单位与distance（物距）的单位保持一致
cv::Point3f MonoMeasureTool::unproject(cv::Point2f p, double distance) 
{
  double zc = distance;
  double xc = (p.x - u0_) * distance / fx_;
  double yc = (p.y - v0_) * distance / fy_;
  return cv::Point3f(xc, yc, zc);
}
// 获取image任意点的视角，pitch，yaw（相对相机坐标系）。
// 与相机坐标系保持一致。
void MonoMeasureTool::calcViewAngle(cv::Point2f p, float & pitch, float & yaw)
{
  pitch = atan2((p.y - v0_), fy_);
  yaw = atan2((p.x - u0_), fx_);
}

bool MonoMeasureTool::calcArmorTarget(
  const ArmorObject & obj, cv::Point3f & position, cv::Mat & rvec, std::string & armor_type)
{
  if(obj.is_ok)
  { 
    
    if (is_big_armor(obj)) {
      armor_type = "large";
      return solvePnp(obj.pts_binary, BIG_ARMOR_3D_POINTS, position, rvec, cv::SOLVEPNP_IPPE);
    } else {
      armor_type = "small";
      return solvePnp(obj.pts_binary, SMALL_ARMOR_3D_POINTS, position, rvec, cv::SOLVEPNP_IPPE);
    }
   

  }
  else {
  if (is_big_armor(obj)) {
    armor_type = "large";
    return solvePnp(obj.pts, BIG_ARMOR_3D_POINTS_NET, position, rvec, cv::SOLVEPNP_IPPE);
  } else {
    armor_type = "small";
    return solvePnp(obj.pts, SMALL_ARMOR_3D_POINTS_NET, position, rvec, cv::SOLVEPNP_IPPE);
 }
}
}
// bool MonoMeasureTool::calcArmorTarget(
//   const ArmorObject & obj,
//   cv::Point3f & position,
//   cv::Mat & rvec,
//   std::string & armor_type)
// {
//   // Determine armor size
//   const std::vector<cv::Point3f> *model_points = nullptr;
//   const std::vector<cv::Point2f> *image_points = nullptr;

//   if (obj.is_ok) {
//     image_points = &obj.pts_binary;
//     if (is_big_armor(obj)) {
//       armor_type = "large";
//       model_points = &BIG_ARMOR_3D_POINTS;
//     } else {
//       armor_type = "small";
//       model_points = &SMALL_ARMOR_3D_POINTS;
//     }
//   } else {
//     image_points = &obj.pts;
//     if (is_big_armor(obj)) {
//       armor_type = "large";
//       model_points = &BIG_ARMOR_3D_POINTS_NET;
//     } else {
//       armor_type = "small";
//       model_points = &SMALL_ARMOR_3D_POINTS_NET;
//     }
//   }

//   // Ensure camera parameters initialized
//   if (camera_intrinsic_.empty() || camera_distortion_.empty()) {
//     WUST_ERROR(mono_logger) << "Camera parameters not initialized.";
//     return false;
//   }

//   // Use solvePnPGeneric to get all candidate solutions
//   std::vector<cv::Mat> rvecs, tvecs;
//   bool generic_ok = cv::solvePnPGeneric(
//     *model_points, *image_points,
//     camera_intrinsic_, camera_distortion_,
//     rvecs, tvecs,
//     false, /* useExtrinsicGuess */
//     cv::SOLVEPNP_IPPE
//   );
//   if (!generic_ok || rvecs.empty()) {
//     WUST_WARN(mono_logger) << "solvePnPGeneric failed.";
//     return false;
//   }

//   // Select solution with minimum reprojection error
//   double best_err = std::numeric_limits<double>::max();
//   size_t best_idx = 0;
//   for (size_t i = 0; i < rvecs.size(); ++i) {
//     std::vector<cv::Point2f> reproj;
//     cv::projectPoints(
//       *model_points, rvecs[i], tvecs[i],
//       camera_intrinsic_, camera_distortion_, reproj
//     );
//     double err = 0.0;
//     for (size_t k = 0; k < reproj.size(); ++k) {
//       err += cv::norm(reproj[k] - (*image_points)[k]);
//     }
//     if (err < best_err) {
//       best_err = err;
//       best_idx = i;
//     }
//   }

//   // Retrieve best solution
//   rvec = rvecs[best_idx].clone();
//   cv::Mat tvec = tvecs[best_idx].clone();
//   position = cv::Point3f(tvec);

//   // Store for next-frame iterative refinement
//   prev_rvec_ = rvec;
//   prev_tvec_ = tvec;
//   has_prev_ = true;

//   return true;
// }
float MonoMeasureTool::calcDistanceToCenter(const ArmorObject & obj) 
{
  cv::Point2f img_center(
    this->camera_intrinsic_.at<double>(0, 2), this->camera_intrinsic_.at<double>(1, 2));
  cv::Point2f armor_center;
  if(obj.is_ok)
  {
  armor_center.x = (obj.pts_binary[0].x + obj.pts_binary[1].x + obj.pts_binary[2].x + obj.pts_binary[3].x) / 4.;
  armor_center.y = (obj.pts_binary[0].y + obj.pts_binary[1].y + obj.pts_binary[2].y + obj.pts_binary[3].y) / 4.;
  }
  else {
  armor_center.x = (obj.pts[0].x + obj.pts[1].x + obj.pts[2].x + obj.pts[3].x) / 4.;
  armor_center.y = (obj.pts[0].y + obj.pts[1].y + obj.pts[2].y + obj.pts[3].y) / 4.;
}
  auto dis_vec = img_center - armor_center;
  return sqrt(dis_vec.dot(dis_vec));
}
bool MonoMeasureTool::reprojectArmorCorners(
  const Armor & armor,
  std::vector<cv::Point2f> & image_points)
{
  if (camera_intrinsic_.empty() || camera_distortion_.empty()) {
    WUST_ERROR(mono_logger) << "Camera parameters not initialized.";
    return false;
  }

  // 获取装甲板的模板角点
  const std::vector<cv::Point3f> * model_points;
  if (armor.type == "large") {
    model_points = &BIG_ARMOR_3D_POINTS;
  } else if (armor.type == "small") {
    model_points = &SMALL_ARMOR_3D_POINTS;
  } else {
    WUST_ERROR(mono_logger) << "Unknown armor type: " << armor.type;
    return false;
  }

  // 四元数 -> 旋转矩阵
  tf2::Matrix3x3 tf_rot(armor.target_ori);
  cv::Mat rot_mat = (cv::Mat_<double>(3, 3) <<
    tf_rot[0][0], tf_rot[0][1], tf_rot[0][2],
    tf_rot[1][0], tf_rot[1][1], tf_rot[1][2],
    tf_rot[2][0], tf_rot[2][1], tf_rot[2][2]);

  // 旋转矩阵 -> 旋转向量
  cv::Mat rvec;
  cv::Rodrigues(rot_mat, rvec);

  // 平移向量
  cv::Mat tvec = (cv::Mat_<double>(3,1) << armor.target_pos.x, armor.target_pos.y, armor.target_pos.z);

  // 反投影
  cv::projectPoints(*model_points, rvec, tvec, camera_intrinsic_, camera_distortion_, image_points);

  return true;
}
bool MonoMeasureTool::reprojectArmorCorners_raw(
  const Armor & armor,
  std::vector<cv::Point2f> & image_points)
{
  if (camera_intrinsic_.empty() || camera_distortion_.empty()) {
    WUST_ERROR(mono_logger) << "Camera parameters not initialized.";
    return false;
  }

  // 获取装甲板的模板角点
  const std::vector<cv::Point3f> * model_points;
  if (armor.type == "large") {
    model_points = &BIG_ARMOR_3D_POINTS;
  } else if (armor.type == "small") {
    model_points = &SMALL_ARMOR_3D_POINTS;
  } else {
    WUST_ERROR(mono_logger) << "Unknown armor type: " << armor.type;
    return false;
  }

  // 四元数 -> 旋转矩阵
  tf2::Matrix3x3 tf_rot(armor.ori);
  cv::Mat rot_mat = (cv::Mat_<double>(3, 3) <<
    tf_rot[0][0], tf_rot[0][1], tf_rot[0][2],
    tf_rot[1][0], tf_rot[1][1], tf_rot[1][2],
    tf_rot[2][0], tf_rot[2][1], tf_rot[2][2]);

  // 旋转矩阵 -> 旋转向量
  cv::Mat rvec;
  cv::Rodrigues(rot_mat, rvec);

  // 平移向量
  cv::Mat tvec = (cv::Mat_<double>(3,1) << armor.pos.x, armor.pos.y, armor.pos.z);

  // 反投影
  cv::projectPoints(*model_points, rvec, tvec, camera_intrinsic_, camera_distortion_, image_points);

  return true;
}
bool MonoMeasureTool::reprojectArmorsCorners(
  Armors & armors,
  Target_info & target_info)
  {
    if (camera_intrinsic_.empty() || camera_distortion_.empty()) {
      WUST_ERROR(mono_logger) << "Camera parameters not initialized.";
      return false;
    }
    for (auto & armor : armors.armors) { 
      std::vector<cv::Point2f> pts;
      
      if(!reprojectArmorCorners(armor, pts))return false;
      target_info.pts.push_back(pts);
      target_info.pos.push_back(armor.pos);
      target_info.ori.push_back(armor.ori);
    }
    return true;
  }
