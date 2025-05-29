// Maintained by Shenglin Qin, Chengfu Zou
// Copyright (C) FYT Vision Group. All rights reserved.
// Copyright 2025 XiaoJian Wu
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

#include "detect/light_corner_corrector.hpp"
#include "common/logger.hpp"

#include <iostream>
#include <numeric>
#include <ostream>

void LightCornerCorrector::correctCorners(ArmorObject &armor) noexcept {
  constexpr int PASS_OPTIMIZE_WIDTH = 3;
  if (armor.lights.empty())
    return;
  if (armor.whole_gray_img.empty()) {
    return;
  }

  double zero_x = armor.new_x;
  double zero_y = armor.new_y;

  for (auto &light : armor.lights) {
    if (light.width <= PASS_OPTIMIZE_WIDTH) {
      light.top.x += zero_x;
      light.top.y += zero_y;
      light.center.x += zero_x;
      light.center.y += zero_y;
      light.bottom.x += zero_x;
      light.bottom.y += zero_y;

      continue;
    }

    SymmetryAxis axis = findSymmetryAxis(armor.whole_gray_img, light);

    light.center = axis.centroid;
    light.axis = axis.direction;

    if (cv::Point2f t = findCorner(armor.whole_gray_img, light, axis, "top");
        t.x > 0) {
      light.top = t;
    }
    if (cv::Point2f b = findCorner(armor.whole_gray_img, light, axis, "bottom");
        b.x > 0) {
      light.bottom = b;
    }
    light.top.x += zero_x;
    light.top.y += zero_y;
    light.center.x += zero_x;
    light.center.y += zero_y;

    light.bottom.x += zero_x;
    light.bottom.y += zero_y;
  }

  armor.pts_binary.clear();
  cv::Point2f armor_center =
      (armor.pts[0] + armor.pts[1] + armor.pts[2] + armor.pts[3]) * 0.25;

  // Step 2: 计算每个灯条中心与装甲板中心的距离
  std::vector<std::pair<const Light *, double>> light_distances;
  for (const auto &light : armor.lights) {
    double dist = cv::norm(light.center - armor_center);
    light_distances.emplace_back(&light, dist);
  }

  // Step 3: 按距离排序，选择最近两个灯条
  std::sort(light_distances.begin(), light_distances.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; });
  if (light_distances.size() >= 2) {
    const Light *l1 = light_distances[0].first;
    const Light *l2 = light_distances[1].first;
  }

  if (light_distances.size() >= 2) {
    const Light *l1 = light_distances[0].first;
    const Light *l2 = light_distances[1].first;

    // 判断哪个灯条在左侧，哪个在右侧
    if (l1->center.x < l2->center.x) {
      armor.lights[0] = *l1; // 解引用指针，赋值对象
      armor.lights[1] = *l2;
    } else {
      armor.lights[0] = *l2;
      armor.lights[1] = *l1;
    }
  }

  // Step 4: 构建 candidates，只保留两个灯条的 top/bottom
  std::vector<cv::Point2f> candidates;
  for (int i = 0; i < std::min(2, (int)light_distances.size()); ++i) {
    const auto *light = light_distances[i].first;
    candidates.push_back(light->top);
    candidates.push_back(light->bottom);
  }

  double w = cv::norm(armor.pts[0] - armor.pts[1]);
  double h = cv::norm(armor.pts[0] - armor.pts[3]);
  double size_scale = w + h;

  std::vector<cv::Point2f> selected_pts(4, cv::Point2f(-1, -1));
  std::vector<int> selected_indices(4, -1);

  for (int i = 0; i < 4; ++i) {
    double min_dist = DBL_MAX;
    int best_match = -1;

    double test_result = cv::pointPolygonTest(armor.pts, armor.pts[i], false);
    double dist_threshold =
        (test_result >= 0) ? (0.15 * size_scale) : (0.25 * size_scale);

    for (size_t j = 0; j < candidates.size(); ++j) {
      double dist = cv::norm(armor.pts[i] - candidates[j]);
      if (dist < min_dist) {
        min_dist = dist;
        best_match = static_cast<int>(j);
      }
    }

    if (best_match != -1 && min_dist < dist_threshold) {
      selected_pts[i] = candidates[best_match];
      selected_indices[i] = best_match;
    }
  }

  for (const auto &pt : selected_pts) {
    if (pt.x >= 0 && pt.y >= 0) {
      auto it = std::find(candidates.begin(), candidates.end(), pt);
      if (it != candidates.end())
        candidates.erase(it);
    }
    armor.pts_binary.push_back(pt);
  }

  armor.is_ok = true;
  for (const auto &pt : armor.pts_binary) {
    if (pt.x < 0 || pt.y < 0) {
      armor.is_ok = false;
      break;
    }
  }

  if (std::count_if(
          armor.pts_binary.begin(), armor.pts_binary.end(),
          [](const cv::Point2f &p) { return p.x >= 0 && p.y >= 0; }) != 4) {
    armor.is_ok = false;
  }

  if (!armor.is_ok) {
    armor.pts_binary.clear();
  }
  if (armor.is_ok) {
    armor.center = (armor.lights[0].center + armor.lights[1].center) / 2;
  } else {
    armor.center = armor_center;
  }
}

SymmetryAxis LightCornerCorrector::findSymmetryAxis(const cv::Mat &gray_img,
                                                    const Light &light) {
  constexpr float MAX_BRIGHTNESS = 25;
  constexpr float SCALE = 0.07;

  // Scale the bounding box
  cv::Rect light_box = light.boundingRect();
  light_box.x -= light_box.width * SCALE;
  light_box.y -= light_box.height * SCALE;
  light_box.width += light_box.width * SCALE * 2;
  light_box.height += light_box.height * SCALE * 2;

  // Check boundary
  light_box.x = std::max(light_box.x, 0);
  light_box.x = std::min(light_box.x, gray_img.cols - 1);
  light_box.y = std::max(light_box.y, 0);
  light_box.y = std::min(light_box.y, gray_img.rows - 1);
  light_box.width = std::min(light_box.width, gray_img.cols - light_box.x);
  light_box.height = std::min(light_box.height, gray_img.rows - light_box.y);

  // Get normalized light image
  cv::Mat roi = gray_img(light_box);
  float mean_val = cv::mean(roi)[0];
  roi.convertTo(roi, CV_32F);
  cv::normalize(roi, roi, 0, MAX_BRIGHTNESS, cv::NORM_MINMAX);

  // Calculate the centroid
  cv::Moments moments = cv::moments(roi, false);
  cv::Point2f centroid =
      cv::Point2f(moments.m10 / moments.m00, moments.m01 / moments.m00) +
      cv::Point2f(light_box.x, light_box.y);

  // Initialize the PointCloud
  std::vector<cv::Point2f> points;
  for (int i = 0; i < roi.rows; i++) {
    for (int j = 0; j < roi.cols; j++) {
      for (int k = 0; k < std::round(roi.at<float>(i, j)); k++) {
        points.emplace_back(cv::Point2f(j, i));
      }
    }
  }
  cv::Mat points_mat = cv::Mat(points).reshape(1);

  // PCA (Principal Component Analysis)
  auto pca = cv::PCA(points_mat, cv::Mat(), cv::PCA::DATA_AS_ROW);

  // Get the symmetry axis
  cv::Point2f axis = cv::Point2f(pca.eigenvectors.at<float>(0, 0),
                                 pca.eigenvectors.at<float>(0, 1));

  // Normalize the axis
  axis = axis / cv::norm(axis);

  if (axis.y > 0) {
    axis = -axis;
  }

  return SymmetryAxis{
      .centroid = centroid, .direction = axis, .mean_val = mean_val};
}
//   SymmetryAxis LightCornerCorrector::findSymmetryAxis(const cv::Mat
//   &gray_img, const Light &light) {
//     constexpr float MAX_BRIGHTNESS = 25.0f;
//     constexpr float SCALE = 0.07f;

//     // Scale bounding box
//     cv::Rect light_box = light.boundingRect();
//     light_box.x -= light_box.width * SCALE;
//     light_box.y -= light_box.height * SCALE;
//     light_box.width += light_box.width * SCALE * 2;
//     light_box.height += light_box.height * SCALE * 2;

//     // Clip ROI within image bounds
//     light_box &= cv::Rect(0, 0, gray_img.cols, gray_img.rows);
//     if (light_box.empty()) {
//         return SymmetryAxis{.centroid = cv::Point2f(-1, -1), .direction = {0,
//         -1}, .mean_val = 0};
//     }

//     // Get ROI and normalize
//     cv::Mat roi = gray_img(light_box);
//     if (roi.empty()) {
//         return SymmetryAxis{.centroid = cv::Point2f(-1, -1), .direction = {0,
//         -1}, .mean_val = 0};
//     }

//     cv::Mat roi_float;
//     roi.convertTo(roi_float, CV_32F);
//     float mean_val = static_cast<float>(cv::mean(roi_float)[0]);
//     cv::normalize(roi_float, roi_float, 0.0f, MAX_BRIGHTNESS,
//     cv::NORM_MINMAX);

//     // Weighted centroid and covariance calculation
//     double m00 = 0, m10 = 0, m01 = 0;
//     double mu20 = 0, mu02 = 0, mu11 = 0;

//     for (int y = 0; y < roi_float.rows; ++y) {
//         for (int x = 0; x < roi_float.cols; ++x) {
//             float w = roi_float.at<float>(y, x);
//             if (w < 1e-3f) continue;

//             m00 += w;
//             m10 += x * w;
//             m01 += y * w;
//         }
//     }

//     if (m00 < 1e-5) {
//         return SymmetryAxis{.centroid = cv::Point2f(-1, -1), .direction = {0,
//         -1}, .mean_val = mean_val};
//     }

//     double cx = m10 / m00;
//     double cy = m01 / m00;

//     for (int y = 0; y < roi_float.rows; ++y) {
//         for (int x = 0; x < roi_float.cols; ++x) {
//             float w = roi_float.at<float>(y, x);
//             if (w < 1e-3f) continue;

//             double dx = x - cx;
//             double dy = y - cy;
//             mu20 += w * dx * dx;
//             mu02 += w * dy * dy;
//             mu11 += w * dx * dy;
//         }
//     }

//     // Compute orientation from covariance matrix
//     double theta = 0.5 * std::atan2(2 * mu11, mu20 - mu02);
//     cv::Point2f dir(std::cos(theta), std::sin(theta));
//     if (dir.y > 0) dir = -dir;

//     cv::Point2f centroid(cx + light_box.x, cy + light_box.y);
//     return SymmetryAxis{.centroid = centroid, .direction = dir, .mean_val =
//     mean_val};
//  }

cv::Point2f LightCornerCorrector::findCorner(const cv::Mat &gray_img,
                                             const Light &light,
                                             const SymmetryAxis &axis,
                                             std::string order) {
  constexpr float START = 0.8 / 2;
  constexpr float END = 1.2 / 2;

  auto inImage = [&gray_img](const cv::Point &point) -> bool {
    return point.x >= 0 && point.x < gray_img.cols && point.y >= 0 &&
           point.y < gray_img.rows;
  };

  auto distance = [](float x0, float y0, float x1, float y1) -> float {
    return std::sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
  };

  int oper = order == "top" ? 1 : -1;
  float L = light.length;
  float dx = axis.direction.x * oper;
  float dy = axis.direction.y * oper;

  std::vector<cv::Point2f> candidates;

  // Select multiple corner candidates and take the average as the final corner
  int n = light.width - 2;
  int half_n = std::round(n / 2);
  for (int i = -half_n; i <= half_n; i++) {
    float x0 = axis.centroid.x + L * START * dx + i;
    float y0 = axis.centroid.y + L * START * dy;

    cv::Point2f prev = cv::Point2f(x0, y0);
    cv::Point2f corner = cv::Point2f(x0, y0);
    float max_brightness_diff = 0;
    bool has_corner = false;

    // Search along the symmetry axis to find the corner that has the maximum
    // brightness difference
    for (float x = x0 + dx, y = y0 + dy;
         distance(x, y, x0, y0) < L * (END - START); x += dx, y += dy) {
      cv::Point2f cur = cv::Point2f(x, y);
      if (!inImage(cv::Point(cur))) {
        break;
      }

      float brightness_diff =
          gray_img.at<uchar>(prev) - gray_img.at<uchar>(cur);
      if (brightness_diff > max_brightness_diff &&
          gray_img.at<uchar>(prev) > axis.mean_val) {
        max_brightness_diff = brightness_diff;
        corner = prev;
        has_corner = true;
      }

      prev = cur;
    }

    if (has_corner) {
      candidates.emplace_back(corner);
    }
  }
  if (!candidates.empty()) {
    cv::Point2f result = std::accumulate(candidates.begin(), candidates.end(),
                                         cv::Point2f(0, 0)) /
                         static_cast<float>(candidates.size());
    return result;
  } else {
    return cv::Point2f(-1, -1);
  }
}
