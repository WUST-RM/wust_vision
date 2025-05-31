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
  double roll;
  double pitch;
  double yaw;
};
namespace tf2 {

class Quaternion {
public:
  double x, y, z, w;

  Quaternion() : x(0.0), y(0.0), z(0.0), w(1.0) {}
  Quaternion(double x_, double y_, double z_, double w_)
      : x(x_), y(y_), z(z_), w(w_) {}

  cv::Matx33d toRotationMatrix() const {
    double xx = x * x, yy = y * y, zz = z * z;
    double xy = x * y, xz = x * z, yz = y * z;
    double xw = x * w, yw = y * w, zw = z * w;

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
  result.roll = static_cast<double>(std::atan2(sinr_cosp, cosr_cosp));

  // pitch (y-axis rotation)
  double sinp = 2.0 * (q.w * q.y - q.z * q.x);
  if (std::abs(sinp) >= 1.0)
    result.pitch = static_cast<double>(std::copysign(M_PI / 2, sinp));
  else
    result.pitch = static_cast<double>(std::asin(sinp));

  // yaw (z-axis rotation)
  double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  result.yaw = static_cast<double>(std::atan2(siny_cosp, cosy_cosp));

  return result;
}
class Matrix3x3 {
public:
  double m[3][3];

  Matrix3x3() {
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        m[i][j] = (i == j) ? 1.0f : 0.0f;
  }
  Matrix3x3(double m00, double m01, double m02, double m10, double m11, double m12,
            double m20, double m21, double m22) {
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
        m[i][j] = static_cast<double>(rot(i, j));
  }
  const double *operator[](int i) const { return m[i]; }
  double *operator[](int i) { return m[i]; }

  void getRotation(Quaternion &q) const {
    double trace = m[0][0] + m[1][1] + m[2][2];
    if (trace > 0.0) {
      double s = std::sqrt(trace + 1.0f) * 2.0f;
      q.w = 0.25f * s;
      q.x = (m[2][1] - m[1][2]) / s;
      q.y = (m[0][2] - m[2][0]) / s;
      q.z = (m[1][0] - m[0][1]) / s;
    } else {
      int i = (m[0][0] > m[1][1]) ? ((m[0][0] > m[2][2]) ? 0 : 2)
                                  : ((m[1][1] > m[2][2]) ? 1 : 2);
      double s;
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
  double x, y, z;
  Position() : x(0.0f), y(0.0f), z(0.0f) {}
  Position(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

  Position operator+(const Position &other) const {
    return Position(x + other.x, y + other.y, z + other.z);
  }

  Position operator-(const Position &other) const {
    return Position(x - other.x, y - other.y, z - other.z);
  }

  Position operator*(double scalar) const {
    return Position(x * scalar, y * scalar, z * scalar);
  }
  Eigen::Vector3d toEigen() const { return {x, y, z}; }
  static Position fromEigen(const Eigen::Vector3d& v) { return {v.x(), v.y(), v.z()}; }
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

//   static Transform compose(const Transform &a, const Transform &b) {
//   cv::Matx44d result = a.toMatrix() * b.toMatrix();
//   Transform t = fromMatrix(result);
//   t.timestamp = a.timestamp;  // 或者其他策略
//   return t;
// }
// Eigen::Isometry3d toEigenIso() const {
//   Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
//   iso.translate(position.toEigen());
//   iso.rotate(Eigen::Quaterniond(
//       orientation.w, orientation.x, 
//       orientation.y, orientation.z
//   ));
//   return iso;
// }

static Transform fromEigen(const Eigen::Isometry3d& iso) {
  Eigen::Vector3d pos = iso.translation();
  Eigen::Quaterniond q(iso.rotation());
  return {
      Position::fromEigen(pos),
      tf2::Quaternion(q.x(), q.y(), q.z(), q.w()),
      std::chrono::steady_clock::now()
  };
}

static Transform compose(const Transform& a, const Transform& b) {
  Eigen::Isometry3d a_iso = a.toEigenIso();
  Eigen::Isometry3d b_iso = b.toEigenIso();
  return fromEigen(a_iso * b_iso);
}
Eigen::Isometry3d toEigenIso() const {
  Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
  iso.translate(Eigen::Vector3d(position.x, position.y, position.z));
  iso.rotate(Eigen::Quaterniond(
      orientation.w, orientation.x, 
      orientation.y, orientation.z
  ));
  return iso;
}

static Transform fromEigen(const Eigen::Isometry3d& iso, std::chrono::steady_clock::time_point stamp) {
  Eigen::Vector3d pos = iso.translation();
  Eigen::Quaterniond q(iso.rotation());
  return {
      {pos.x(), pos.y(), pos.z()},
      tf2::Quaternion(q.x(), q.y(), q.z(), q.w()),
      stamp
  };
}

};

inline Transform createTf(double x, double y, double z, const tf2::Quaternion &q) {
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
      using Duration = std::chrono::steady_clock::duration;
      static constexpr Duration STATIC_THRESHOLD = std::chrono::hours(24); // 24小时视为静态
      static constexpr Duration RECENT_THRESHOLD = std::chrono::nanoseconds(100); // 0.0001ms内视为最新
      //static constexpr Duration RECENT_THRESHOLD = std::chrono::milliseconds(1); // 1ms内视为最新
      
      // 设置变换关系，支持静态标记
      void setTransform(const std::string &parent, const std::string &child,
                        const Transform &tf, bool is_static = false) {
          std::unique_lock lock(mutex_);
          FrameNode &node = nodes_[child];
          node.parent_frame = parent;
          
          if (is_static) {
              // 静态变换：清除所有历史，只保留当前变换
              node.transforms.clear();
              node.transforms[tf.timestamp] = tf;
              node.is_static = true;
          } else {
              // 动态变换：添加新变换，维护最近1000个变换
              node.is_static = false;
              node.transforms[tf.timestamp] = tf;
              
              // 动态变换缓冲区管理（保留最近1000个）
              if (node.transforms.size() > 3000) {
                  node.transforms.erase(node.transforms.begin());
              }
          }
      }
  
      // 核心查询方法
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
      Transform transform(const Transform &input, 
        const std::string &source_frame,
        const std::string &target_frame,
        const Time time) const {
        // 1. 获取坐标系间变换
        Transform frame_tf;
        if (!getTransform(source_frame, target_frame, time, frame_tf)) {
        throw std::runtime_error("Cannot find transform from " + 
                              source_frame + " to " + target_frame);    
        }

        // 2. 转换为 Eigen 等距变换
        Eigen::Isometry3d input_iso = input.toEigenIso();
        Eigen::Isometry3d frame_tf_iso = frame_tf.toEigenIso();

        // 3. 执行变换: 目标坐标系 = frame_tf * source
        Eigen::Isometry3d result_iso = frame_tf_iso * input_iso;

        // 4. 转换回 Transform 结构
        return Transform::fromEigen(result_iso, input.timestamp);
        }
  
  private:
      struct FrameNode {
          std::string parent_frame;
          std::map<Time, Transform> transforms;
          bool is_static = false;
          mutable Time last_query_time;  // 上次查询时间
          mutable Transform last_transform; // 上次查询结果缓存
      };
  
      std::unordered_map<std::string, FrameNode> nodes_;
      mutable std::shared_mutex mutex_;
  
      // 获取到根节点的路径（带环检测）
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
  
      // 沿路径累加变换
      Transform accumulatePath(const std::vector<std::string> &path, int end_idx,
                               const Time &time, bool reverse) const {
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
  
      // 时间插值查询（核心优化）
      Transform lookupTransformAtTime(const FrameNode &node,
                                      const Time &time) const {
          // 静态变换直接返回（忽略时间戳）
          if (node.is_static) {
              return node.transforms.begin()->second;
          }
          
          // 检查时间查询缓存（避免重复计算）
          if (std::chrono::abs(time - node.last_query_time) < RECENT_THRESHOLD) {
              return node.last_transform;
          }
          
          const auto &transforms = node.transforms;
          
          // 空变换处理
          if (transforms.empty()) {
              throw std::runtime_error("No transforms available");
          }
          
          // 单变换直接返回
          if (transforms.size() == 1) {
              return transforms.begin()->second;
          }
          
          // 查找时间边界
          auto it_after = transforms.lower_bound(time);
          
          // 边界情况处理
          if (it_after == transforms.begin()) {
              return cacheResult(node, time, it_after->second);
          }
          if (it_after == transforms.end()) {
              return cacheResult(node, time, std::prev(it_after)->second);
          }
          
          // 获取前后变换
          auto it_before = std::prev(it_after);
          const Time &t0 = it_before->first;
          const Time &t1 = it_after->first;
          
          // 时间差过小直接返回最近变换
          Duration delta = t1 - t0;
          if (delta < RECENT_THRESHOLD) {
              return cacheResult(node, time, it_after->second);
          }
          
          // 计算插值比例
          double alpha = std::chrono::duration<double>(time - t0).count() /
                       std::chrono::duration<double>(delta).count();
          
          // 执行插值
          return cacheResult(node, time, 
                            interpolate(it_before->second, it_after->second, alpha));
      }
      
      // 插值结果缓存
      Transform cacheResult(const FrameNode &node, const Time &time, 
                           const Transform &result) const {
          // 非const访问需要mutable
          const_cast<FrameNode&>(node).last_query_time = time;
          const_cast<FrameNode&>(node).last_transform = result;
          return result;
      }
  
      // 高效插值实现（使用Eigen）
      Transform interpolate(const Transform &a, const Transform &b, double alpha) const {
          // 位置线性插值
          Eigen::Vector3d pos = 
              (1.0 - alpha) * a.position.toEigen() + 
              alpha * b.position.toEigen();
          
          // 旋转球面插值
          Eigen::Quaterniond qa(a.orientation.w, a.orientation.x, 
                               a.orientation.y, a.orientation.z);
          Eigen::Quaterniond qb(b.orientation.w, b.orientation.x, 
                               b.orientation.y, b.orientation.z);
          Eigen::Quaterniond q = qa.slerp(alpha, qb).normalized();
          
          // 时间戳插值
          auto interpolated_time = a.timestamp + 
              std::chrono::duration_cast<Duration>(
                  std::chrono::duration<double>((b.timestamp - a.timestamp) * alpha));
          
          return Transform(Position::fromEigen(pos), 
                           tf2::Quaternion(q.x(), q.y(), q.z(), q.w()),
                           interpolated_time);
      }
  
      // 变换求逆（使用Eigen）
      Transform invert(const Transform &tf) const {
          Eigen::Isometry3d iso = tf.toEigenIso();
          return Transform::fromEigen(iso.inverse());
      }
  };

#endif // TF2_HPP