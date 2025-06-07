#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <vector>

/**
 * @brief 基于 yaw-pitch 状态的简单 MPC 控制器
 * 状态向量: [yaw, pitch]ᵀ
 */
class SimpleMPC {
public:
  SimpleMPC(double dt, int horizon) : m_dt(dt), m_horizon(horizon) {
    initSystemMatrices();
    buildHessian();
  }

  /**
   * @brief 计算 MPC 控制量
   * @param cur_state 当前状态 [yaw, pitch]
   * @param ref_traj  期望轨迹，长度必须等于 horizon
   * @return 返回当前时刻的控制输入 [Δyaw, Δpitch]
   */
  Eigen::Vector2d computeControl(const Eigen::Vector2d &cur_state,
                                 const std::vector<Eigen::Vector2d> &ref_traj) {
    if (ref_traj.size() != static_cast<size_t>(m_horizon)) {
      std::cerr << "[MPC] Reference trajectory size mismatch!\n";
      return Eigen::Vector2d::Zero();
    }

    // 构造线性项 f = ∑ Bᵀ Q (A x - r)
    m_f.setZero();
    for (int i = 0; i < m_horizon; ++i) {
      Eigen::Vector2d pred_state = m_A * cur_state;     // 状态预测
      Eigen::Vector2d error = pred_state - ref_traj[i]; // 与参考的误差
      m_f.segment<2>(2 * i) = m_B.transpose() * m_Q * error; // 线性项累加
    }

    // 求解最优控制向量 u = -H⁻¹ f（简化为解析解）
    Eigen::VectorXd u_opt = -m_H.ldlt().solve(m_f);

    // 只使用第一个控制步
    return u_opt.segment<2>(0);
  }

  // 设置角度误差与控制权重（yaw 和 pitch 分开）
  void setWeight(double q_yaw, double q_pitch, double r_yaw, double r_pitch) {
    m_Q = Eigen::Matrix2d::Zero();
    m_Q(0, 0) = q_yaw;
    m_Q(1, 1) = q_pitch;

    m_R = Eigen::Matrix2d::Zero();
    m_R(0, 0) = r_yaw;
    m_R(1, 1) = r_pitch;

    buildHessian(); // 更新 H
  }

private:
  double m_dt;
  int m_horizon;

  // 状态空间模型
  Eigen::Matrix2d m_A; // 状态转移矩阵
  Eigen::Matrix2d m_B; // 控制矩阵

  // 权重矩阵
  Eigen::Matrix2d m_Q; // 状态误差权重
  Eigen::Matrix2d m_R; // 控制权重

  // MPC 优化矩阵
  Eigen::MatrixXd m_H; // Hessian（二次项）
  Eigen::VectorXd m_f; // 线性项

private:
  void initSystemMatrices() {
    m_A = Eigen::Matrix2d::Identity();        // x(k+1) = x(k) + B*u
    m_B = Eigen::Matrix2d::Identity() * m_dt; // 简单积分器模型
  }

  void buildHessian() {
    const int N = m_horizon;
    m_H = Eigen::MatrixXd::Zero(2 * N, 2 * N);
    m_f = Eigen::VectorXd::Zero(2 * N);

    // 构造 H = ∑ (Bᵀ Q B + R)
    Eigen::Matrix2d H_block = m_B.transpose() * m_Q * m_B + m_R;
    for (int i = 0; i < N; ++i) {
      m_H.block<2, 2>(2 * i, 2 * i) = H_block;
    }
  }
};
