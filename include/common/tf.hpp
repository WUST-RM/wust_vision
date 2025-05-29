#ifndef TF2_HPP
#define TF2_HPP

#include "Eigen/Dense"
#include "fmt/format.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <opencv2/core.hpp>
#include <set>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
struct rpy {
  float roll;
  float pitch;
  float yaw;
};
namespace tf2 {

class Quaternion {
public:
  float x, y, z, w;

  Quaternion() : x(0.0), y(0.0), z(0.0), w(1.0) {}
  Quaternion(float x_, float y_, float z_, float w_)
      : x(x_), y(y_), z(z_), w(w_) {}

  cv::Matx33d toRotationMatrix() const {
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float xw = x * w, yw = y * w, zw = z * w;

    return cv::Matx33d(1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw),
                       2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw),
                       2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy));
  }
  void setRPY(double roll, double pitch, double yaw) {
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);

    w = cr * cp * cy + sr * sp * sy;
    x = sr * cp * cy - cr * sp * sy;
    y = cr * sp * cy + sr * cp * sy;
    z = cr * cp * sy - sr * sp * cy;
  }
  void getRPY(double &roll, double &pitch, double &yaw) const {
    // roll (x-axis rotation)
    double sinr_cosp = 2.0 * (w * x + y * z);
    double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
    roll = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = 2.0 * (w * y - z * x);
    if (std::abs(sinp) >= 1)
      pitch = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
      pitch = std::asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2.0 * (w * z + x * y);
    double cos_y_cosp = 1.0 - 2.0 * (y * y + z * z);
    yaw = std::atan2(siny_cosp, cos_y_cosp);
  }
  Quaternion normalized() const {
    double norm = std::sqrt(x * x + y * y + z * z + w * w);
    if (norm == 0.0)
      return Quaternion(0, 0, 0, 1); // fallback
    return Quaternion(x / norm, y / norm, z / norm, w / norm);
  }
  Quaternion slerp(const Quaternion &other, double t) const {
    // 归一化输入（可选，假设已归一化也行）
    Quaternion q1 = normalized();
    Quaternion q2 = other.normalized();

    // 计算点积
    double dot = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;

    // 如果点积为负，取-q2，避免走长路径
    if (dot < 0.0) {
      q2 = Quaternion(-q2.x, -q2.y, -q2.z, -q2.w);
      dot = -dot;
    }

    const double DOT_THRESHOLD = 0.9995;
    if (dot > DOT_THRESHOLD) {
      // 角度太小，用线性插值代替
      Quaternion result(q1.x + t * (q2.x - q1.x), q1.y + t * (q2.y - q1.y),
                        q1.z + t * (q2.z - q1.z), q1.w + t * (q2.w - q1.w));
      return result.normalized();
    }

    // 真正的 SLERP
    double theta_0 = std::acos(dot); // 起始角
    double theta = theta_0 * t;      // 插值角
    double sin_theta = std::sin(theta);
    double sin_theta_0 = std::sin(theta_0);

    double s0 = std::cos(theta) - dot * sin_theta / sin_theta_0;
    double s1 = sin_theta / sin_theta_0;

    return Quaternion((q1.x * s0) + (q2.x * s1), (q1.y * s0) + (q2.y * s1),
                      (q1.z * s0) + (q2.z * s1), (q1.w * s0) + (q2.w * s1));
  }
};

inline double getYawFromQuaternion(const Quaternion &q) {
  auto R = q.toRotationMatrix();
  return std::atan2(R(1, 0), R(0, 0)); // yaw = atan2(r21, r11)
}
inline rpy getRPYFromQuaternion(const Quaternion &q) {
  rpy result;

  // roll (x-axis rotation)
  double sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z);
  double cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
  result.roll = static_cast<float>(std::atan2(sinr_cosp, cosr_cosp));

  // pitch (y-axis rotation)
  double sinp = 2.0 * (q.w * q.y - q.z * q.x);
  if (std::abs(sinp) >= 1.0)
    result.pitch = static_cast<float>(std::copysign(M_PI / 2, sinp));
  else
    result.pitch = static_cast<float>(std::asin(sinp));

  // yaw (z-axis rotation)
  double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  result.yaw = static_cast<float>(std::atan2(siny_cosp, cosy_cosp));

  return result;
}
class Matrix3x3 {
public:
  float m[3][3];

  Matrix3x3() {
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        m[i][j] = (i == j) ? 1.0f : 0.0f;
  }
  Matrix3x3(float m00, float m01, float m02, float m10, float m11, float m12,
            float m20, float m21, float m22) {
    m[0][0] = m00;
    m[0][1] = m01;
    m[0][2] = m02;
    m[1][0] = m10;
    m[1][1] = m11;
    m[1][2] = m12;
    m[2][0] = m20;
    m[2][1] = m21;
    m[2][2] = m22;
  }

  Matrix3x3(const Quaternion &q) {
    cv::Matx33d rot = q.toRotationMatrix();
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        m[i][j] = static_cast<float>(rot(i, j));
  }
  const float *operator[](int i) const { return m[i]; }
  float *operator[](int i) { return m[i]; }

  void getRotation(Quaternion &q) const {
    float trace = m[0][0] + m[1][1] + m[2][2];
    if (trace > 0.0) {
      float s = std::sqrt(trace + 1.0f) * 2.0f;
      q.w = 0.25f * s;
      q.x = (m[2][1] - m[1][2]) / s;
      q.y = (m[0][2] - m[2][0]) / s;
      q.z = (m[1][0] - m[0][1]) / s;
    } else {
      int i = (m[0][0] > m[1][1]) ? ((m[0][0] > m[2][2]) ? 0 : 2)
                                  : ((m[1][1] > m[2][2]) ? 1 : 2);
      float s;
      if (i == 0) {
        s = std::sqrt(1.0f + m[0][0] - m[1][1] - m[2][2]) * 2.0f;
        q.w = (m[2][1] - m[1][2]) / s;
        q.x = 0.25f * s;
        q.y = (m[0][1] + m[1][0]) / s;
        q.z = (m[0][2] + m[2][0]) / s;
      } else if (i == 1) {
        s = std::sqrt(1.0f + m[1][1] - m[0][0] - m[2][2]) * 2.0f;
        q.w = (m[0][2] - m[2][0]) / s;
        q.x = (m[0][1] + m[1][0]) / s;
        q.y = 0.25f * s;
        q.z = (m[1][2] + m[2][1]) / s;
      } else {
        s = std::sqrt(1.0f + m[2][2] - m[0][0] - m[1][1]) * 2.0f;
        q.w = (m[1][0] - m[0][1]) / s;
        q.x = (m[0][2] + m[2][0]) / s;
        q.y = (m[1][2] + m[2][1]) / s;
        q.z = 0.25f * s;
      }
    }
  }

  void getRPY(double &roll, double &pitch, double &yaw) const {
    if (std::abs(m[2][0]) >= 1.0) {
      pitch = m[2][0] < 0 ? M_PI / 2.0 : -M_PI / 2.0;
      roll = std::atan2(-m[0][1], -m[0][2]);
      yaw = 0.0;
    } else {
      pitch = std::asin(-m[2][0]);
      roll = std::atan2(m[2][1], m[2][2]);
      yaw = std::atan2(m[1][0], m[0][0]);
    }
  }
  Matrix3x3 transpose() const {
    return Matrix3x3(m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1],
                     m[0][2], m[1][2], m[2][2]);
  }
  Eigen::Matrix3d toEigen() const {
    Eigen::Matrix3d mat;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        mat(i, j) = static_cast<double>(m[i][j]);
    return mat;
  }
};

inline std::ostream &operator<<(std::ostream &os, const Quaternion &q) {
  os << "Quaternion(x=" << q.x << ", y=" << q.y << ", z=" << q.z
     << ", w=" << q.w << ")";
  return os;
}
struct Vector3 {
  double x_, y_, z_;
  Vector3(double x, double y, double z) : x_(x), y_(y), z_(z) {}
  double x() const { return x_; }
  double y() const { return y_; }
  double z() const { return z_; }

  Vector3 operator*(const Matrix3x3 &mat) const {
    return Vector3(mat[0][0] * x_ + mat[0][1] * y_ + mat[0][2] * z_,
                   mat[1][0] * x_ + mat[1][1] * y_ + mat[1][2] * z_,
                   mat[2][0] * x_ + mat[2][1] * y_ + mat[2][2] * z_);
  }
  Vector3 operator-() const { return Vector3(-x_, -y_, -z_); }
};

inline Vector3 operator*(const Matrix3x3 &m, const Vector3 &v) { return v * m; }

} // namespace tf2
template <> struct fmt::formatter<tf2::Quaternion> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const tf2::Quaternion &q, FormatContext &ctx) {
    return fmt::format_to(ctx.out(), "{:.3f}, {:.3f}, {:.3f}, {:.3f}", q.w, q.x,
                          q.y, q.z);
  }
};
struct Position {
  float x, y, z;
  Position() : x(0.0f), y(0.0f), z(0.0f) {}
  Position(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

  Position operator+(const Position &other) const {
    return Position(x + other.x, y + other.y, z + other.z);
  }

  Position operator-(const Position &other) const {
    return Position(x - other.x, y - other.y, z - other.z);
  }

  Position operator*(double scalar) const {
    return Position(x * scalar, y * scalar, z * scalar);
  }
};
template <> struct fmt::formatter<Position> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const Position &p, FormatContext &ctx) {
    return fmt::format_to(ctx.out(), "{:.3f}, {:.3f}, {:.3f}", p.x, p.y, p.z);
  }
};

template <>
struct fmt::formatter<std::vector<tf2::Quaternion>>
    : fmt::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(const std::vector<tf2::Quaternion> &quats, FormatContext &ctx)
      -> decltype(ctx.out()) {
    std::string result = "[";
    for (size_t i = 0; i < quats.size(); ++i) {
      if (i > 0)
        result += ", ";
      result += fmt::format("{{{:.3f}, {:.3f}, {:.3f}, {:.3f}}}", quats[i].w,
                            quats[i].x, quats[i].y, quats[i].z);
    }
    result += "]";
    return fmt::format_to(ctx.out(), "{}", result);
  }
};

struct Transform {
  Position position;
  tf2::Quaternion orientation;
  std::chrono::steady_clock::time_point timestamp;

  Transform() : position(), orientation(), timestamp() {}
  Transform(Position p, tf2::Quaternion q,
            std::chrono::steady_clock::time_point ts)
      : position(p), orientation(q), timestamp(ts) {}
  Transform(Position p, const tf2::Quaternion &q) {
    position = p;
    orientation = q;
    timestamp = std::chrono::steady_clock::now();
  }

  cv::Matx44d toMatrix() const {
    tf2::Matrix3x3 R(orientation);
    cv::Matx44d T = cv::Matx44d::eye();
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        T(i, j) = R[i][j];
    T(0, 3) = position.x;
    T(1, 3) = position.y;
    T(2, 3) = position.z;
    return T;
  }

  static Transform fromMatrix(const cv::Matx44d &T) {
    Position pos(T(0, 3), T(1, 3), T(2, 3));
    tf2::Matrix3x3 R(T(0, 0), T(0, 1), T(0, 2), T(1, 0), T(1, 1), T(1, 2),
                     T(2, 0), T(2, 1), T(2, 2));
    tf2::Quaternion q;
    R.getRotation(q);
    return {pos, q};
  }

  static Transform compose(const Transform &a, const Transform &b) {
    cv::Matx44d result = a.toMatrix() * b.toMatrix();
    return fromMatrix(result);
  }
};

inline Transform createTf(float x, float y, float z, const tf2::Quaternion &q) {
  return Transform(Position(x, y, z), q);
}

inline tf2::Quaternion eulerToQuaternion(double roll, double pitch,
                                         double yaw) {
  tf2::Quaternion q;
  q.setRPY(roll, pitch, yaw);
  return q;
}

class TfTree {
public:
  using Time = std::chrono::steady_clock::time_point;

  void setTransform(const std::string &parent, const std::string &child,
                    const Transform &tf) {
    std::unique_lock lock(mutex_);
    FrameNode &node = nodes_[child];
    node.parent_frame = parent;
    node.transforms[tf.timestamp] = tf; // 支持多个时间点
  }

  bool getTransform(const std::string &source_frame,
                    const std::string &target_frame, const Time &time,
                    Transform &out) const {
    std::shared_lock lock(mutex_);
    if (source_frame == target_frame) {
      out = Transform(); // Identity
      return true;
    }

    try {
      auto path_source = getPathToRoot(source_frame);
      auto path_target = getPathToRoot(target_frame);

      int i = path_source.size() - 1, j = path_target.size() - 1;
      while (i >= 0 && j >= 0 && path_source[i] == path_target[j]) {
        --i;
        --j;
      }

      Transform tf_src_to_common =
          accumulatePath(path_source, i + 1, time, false);
      Transform tf_common_to_target =
          accumulatePath(path_target, j + 1, time, true);

      out = Transform::compose(tf_common_to_target, tf_src_to_common);
      return true;
    } catch (...) {
      return false;
    }
  }

  // 复用已有接口
  bool getTransform(const std::string &source_frame,
                    const std::string &target_frame, Transform &out) const {
    return getTransform(source_frame, target_frame,
                        std::chrono::steady_clock::now(), out);
  }

  Transform
  transform(const Position &pos, const std::string &source_frame,
            const std::string &target_frame,
            const Time &time = std::chrono::steady_clock::now()) const {
    Transform tf;
    if (!getTransform(source_frame, target_frame, time, tf)) {
      throw std::runtime_error("Cannot find transform from " + source_frame +
                               " to " + target_frame);
    }

    cv::Matx44d mat = tf.toMatrix();
    cv::Vec4d p(pos.x, pos.y, pos.z, 1.0);
    cv::Vec4d result = mat * p;
    return Transform(Position(result[0], result[1], result[2]), tf.orientation);
  }

  Transform
  transform(const tf2::Quaternion &ori, const std::string &source_frame,
            const std::string &target_frame,
            const Time &time = std::chrono::steady_clock::now()) const {
    Transform tf;
    if (!getTransform(source_frame, target_frame, time, tf)) {
      throw std::runtime_error("Cannot find transform from " + source_frame +
                               " to " + target_frame);
    }

    tf2::Matrix3x3 R_input(ori);
    cv::Matx33d R;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        R(i, j) = R_input[i][j];

    cv::Matx44d mat_input = cv::Matx44d::eye();
    mat_input.get_minor<3, 3>(0, 0) = R;

    cv::Matx44d mat_result = tf.toMatrix() * mat_input;
    tf2::Matrix3x3 R_result(
        mat_result(0, 0), mat_result(0, 1), mat_result(0, 2), mat_result(1, 0),
        mat_result(1, 1), mat_result(1, 2), mat_result(2, 0), mat_result(2, 1),
        mat_result(2, 2));
    tf2::Quaternion out_q;
    R_result.getRotation(out_q);

    return Transform(tf.position, out_q);
  }

  Transform
  transform(const Transform &input, const std::string &source_frame,
            const std::string &target_frame,
            const Time &time = std::chrono::steady_clock::now()) const {
    Transform tf;
    if (!getTransform(source_frame, target_frame, time, tf)) {
      throw std::runtime_error("Cannot find transform from " + source_frame +
                               " to " + target_frame);
    }

    cv::Matx44d mat_input = input.toMatrix();
    cv::Matx44d mat_result = tf.toMatrix() * mat_input;
    return Transform::fromMatrix(mat_result);
  }

private:
  struct FrameNode {
    std::string parent_frame;
    std::map<Time, Transform> transforms;
  };

  std::unordered_map<std::string, FrameNode> nodes_;
  mutable std::shared_mutex mutex_;

  std::vector<std::string> getPathToRoot(const std::string &frame) const {
    std::vector<std::string> path;
    std::set<std::string> visited;

    std::string current = frame;
    while (nodes_.count(current)) {
      if (visited.count(current))
        throw std::runtime_error("TF loop detected.");
      visited.insert(current);
      path.push_back(current);
      current = nodes_.at(current).parent_frame;
    }
    path.push_back(current); // root
    return path;
  }

  Transform accumulatePath(const std::vector<std::string> &path, int end_idx,
                           const Time &time, bool reverse = false) const {
    Transform result;
    if (!reverse) {
      for (int i = 0; i < end_idx; ++i) {
        const auto &f = path[i];
        const FrameNode &node = nodes_.at(f);
        Transform tf = lookupTransformAtTime(node, time);
        result = Transform::compose(tf, result);
      }
    } else {
      for (int i = end_idx - 1; i >= 0; --i) {
        const auto &f = path[i];
        const FrameNode &node = nodes_.at(f);
        Transform tf = lookupTransformAtTime(node, time);
        result = Transform::compose(invert(tf), result);
      }
    }
    return result;
  }

  Transform interpolate(const Transform &a, const Transform &b,
                        double alpha) const {
    tf2::Quaternion q = a.orientation.slerp(b.orientation, alpha);
    Position p = a.position * (1.0 - alpha) + b.position * alpha;
    return Transform(p, q);
  }

  Transform lookupTransformAtTime(const FrameNode &node,
                                  const Time &time) const {
    const auto &transforms = node.transforms;

    auto it_after = transforms.lower_bound(time);
    if (it_after == transforms.begin()) {
      return it_after->second; // time <= 最早时间
    }

    if (it_after == transforms.end()) {
      return std::prev(it_after)->second; // time > 最新时间
    }

    auto it_before = std::prev(it_after);
    const Time &t0 = it_before->first;
    const Time &t1 = it_after->first;

    double alpha = std::chrono::duration<double>(time - t0).count() /
                   std::chrono::duration<double>(t1 - t0).count();
    return interpolate(it_before->second, it_after->second, alpha);
  }

  Transform invert(const Transform &tf) const {
    cv::Matx44d inv = tf.toMatrix().inv();
    return Transform::fromMatrix(inv);
  }
};

#endif // TF2_HPP
