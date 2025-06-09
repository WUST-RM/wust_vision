#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <utility>

#include "impl/joint.hpp" // 保留你的 get_transform、is_link 等定义

namespace fast_tf {

namespace internal {

// 将 Transform、Translation、Rotation 转换为 Eigen 结构体
template <internal::is_transform TransformT>
inline std::tuple<Eigen::Translation3d, Eigen::Quaterniond>
extract_translation_rotation(const TransformT &transform) {
  return {static_cast<Eigen::Translation3d>(transform.translation()),
          static_cast<Eigen::Quaterniond>(transform.linear())};
}

template <internal::is_translation TranslationT>
inline std::tuple<Eigen::Translation3d, Eigen::Quaterniond>
extract_translation_rotation(const TranslationT &translation) {
  return {static_cast<Eigen::Translation3d>(translation),
          Eigen::Quaterniond::Identity()};
}

template <internal::is_rotation RotationT>
inline std::tuple<Eigen::Translation3d, Eigen::Quaterniond>
extract_translation_rotation(const RotationT &rotation) {
  return {Eigen::Translation3d::Identity(),
          static_cast<Eigen::Quaterniond>(rotation)};
}

} // namespace internal

// 保存变换结构体（含时间戳）
struct TransformInfo {
  Eigen::Translation3d translation;
  Eigen::Quaterniond rotation;
  double timestamp;
};

// 替代 ROS2 Broadcaster 的 TfBroadcaster
class TfBroadcaster {
public:
  using FramePair = std::pair<std::string, std::string>;

  static TfBroadcaster &instance() {
    static TfBroadcaster inst;
    return inst;
  }

  static double now() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(
               steady_clock::now().time_since_epoch())
        .count();
  }

  void broadcast(const std::string &parent, const std::string &child,
                 const Eigen::Translation3d &translation,
                 const Eigen::Quaterniond &rotation, double timestamp = now()) {
    FramePair key{parent, child};
    tf_map_[key] = TransformInfo{translation, rotation, timestamp};
  }

  void print_all() const {
    for (const auto &[key, info] : tf_map_) {
      std::cout << key.first << " -> " << key.second << " @ " << info.timestamp
                << "\n";
      std::cout << "  Translation: [" << info.translation.x() << ", "
                << info.translation.y() << ", " << info.translation.z()
                << "]\n";
      std::cout << "  Rotation (quat): [" << info.rotation.w() << ", "
                << info.rotation.x() << ", " << info.rotation.y() << ", "
                << info.rotation.z() << "]\n";
    }
  }

  const std::map<FramePair, TransformInfo> &get_all_transforms() const {
    return tf_map_;
  }

private:
  std::map<FramePair, TransformInfo> tf_map_;
};

// 单个广播（From -> To）
template <internal::is_link From, internal::is_link To,
          typename... JointCollectionTs>
requires(internal::has_joint<From, To> &&requires(
    const JointCollectionTs &...collections) {
  get_transform<From, To>(collections...);
}) inline void broadcast(const JointCollectionTs &...collections) {
  auto transform = get_transform<From, To>(collections...);
  auto [translation, rotation] =
      internal::extract_translation_rotation(transform);
  TfBroadcaster::instance().broadcast(From::name, To::name, translation,
                                      rotation);
}

// 广播所有变换
template <typename JointCollectionT>
inline void broadcast_all(const JointCollectionT &collection) {
  collection.for_each([&collection]<typename From, typename To>() {
    broadcast<From, To>(collection);
  });
}

// 广播被修改的变换（例如需要判断 updated 标志的实现）
template <typename JointCollectionT>
inline void broadcast_all_modified(const JointCollectionT &collection) {
  collection.for_each_modified([&collection]<typename From, typename To>() {
    broadcast<From, To>(collection);
  });
}

} // namespace fast_tf
