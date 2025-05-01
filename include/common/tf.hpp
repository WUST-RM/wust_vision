#ifndef TF2_HPP
#define TF2_HPP

#include <cmath>
#include <iostream>

namespace tf2 {

// 自定义 Quaternion 类
class Quaternion
{
public:
  double x, y, z, w;

  // 默认构造函数产生单位四元数
  Quaternion() : x(0.0), y(0.0), z(0.0), w(1.0) {}
};

// 自定义 3x3 矩阵类
class Matrix3x3
{
public:
  double m[3][3];

  // 构造函数，按行赋值
  Matrix3x3(
    double m00, double m01, double m02,
    double m10, double m11, double m12,
    double m20, double m21, double m22)
  {
    m[0][0] = m00; m[0][1] = m01; m[0][2] = m02;
    m[1][0] = m10; m[1][1] = m11; m[1][2] = m12;
    m[2][0] = m20; m[2][1] = m21; m[2][2] = m22;
  }

  // 根据旋转矩阵计算四元数
  void getRotation(Quaternion &q) const
  {
    double trace = m[0][0] + m[1][1] + m[2][2];
    if (trace > 0.0) {
      double s = std::sqrt(trace + 1.0) * 2.0; // s = 4 * qw
      q.w = 0.25 * s;
      q.x = (m[2][1] - m[1][2]) / s;
      q.y = (m[0][2] - m[2][0]) / s;
      q.z = (m[1][0] - m[0][1]) / s;
    } else if ((m[0][0] > m[1][1]) && (m[0][0] > m[2][2])) {
      double s = std::sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0;
      q.w = (m[2][1] - m[1][2]) / s;
      q.x = 0.25 * s;
      q.y = (m[0][1] + m[1][0]) / s;
      q.z = (m[0][2] + m[2][0]) / s;
    } else if (m[1][1] > m[2][2]) {
      double s = std::sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0;
      q.w = (m[0][2] - m[2][0]) / s;
      q.x = (m[0][1] + m[1][0]) / s;
      q.y = 0.25 * s;
      q.z = (m[1][2] + m[2][1]) / s;
    } else {
      double s = std::sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0;
      q.w = (m[1][0] - m[0][1]) / s;
      q.x = (m[0][2] + m[2][0]) / s;
      q.y = (m[1][2] + m[2][1]) / s;
      q.z = 0.25 * s;
    }
  }
  
};
inline std::ostream& operator<<(std::ostream& os, const Quaternion& q)
    {
    os << "Quaternion(x=" << q.x
        << ", y=" << q.y
        << ", z=" << q.z
        << ", w=" << q.w << ")";
    return os;
    }


}  // namespace tf2

#endif  // TF2_HPP
