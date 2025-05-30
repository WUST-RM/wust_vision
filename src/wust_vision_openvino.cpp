#include "wust_vision_openvino.hpp"
#include "common/gobal.hpp"
#include "common/logger.hpp"

#include "common/calculation.hpp"
#include "common/tf.hpp"
#include "common/tools.hpp"
#include "control/armor_solver.hpp"
#include "type/type.hpp"
#include <csignal>
#include <functional>
#include <iostream>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <ostream>
#include <string>
#include <unistd.h>

WustVision::WustVision() {
  detector_ = nullptr;
  init();
}
WustVision::~WustVision() {
  WUST_INFO(vision_logger) << "Shutting down WustVision...";

  is_inited_ = false;
  stopTimer();

  detector_.reset();

  measure_tool_.reset();

  if (thread_pool_) {
    thread_pool_->waitUntilEmpty();
    thread_pool_.reset();
  }

  WUST_INFO(vision_logger) << "WustVision shutdown complete.";
}
void WustVision::init() {
  YAML::Node config =
      YAML::LoadFile("/home/nuc/wust_vision/config/config_openvino.yaml");
  std::string log_level_ =
      config["logger"]["log_level"].as<std::string>("INFO");
  std::string log_path_ =
      config["logger"]["log_path"].as<std::string>("wust_log");
  bool use_logcli = config["logger"]["use_logcli"].as<bool>();
  bool use_logfile = config["logger"]["use_logfile"].as<bool>();
  bool use_simplelog = config["logger"]["use_simplelog"].as<bool>();
  initLogger(log_level_, log_path_, use_logcli, use_logfile, use_simplelog);
  debug_mode_ = config["debug"]["debug_mode"].as<bool>();
  debug_w = config["debug"]["debug_w"].as<int>(640);
  debug_h = config["debug"]["debug_h"].as<int>(480);
  debug_show_dt_ = config["debug"]["debug_show_dt"].as<double>(0.05);
  use_calculation_ = config["use_calculation"].as<bool>();
  auto classify_model_path = config["classify_model_path"].as<std::string>();
  auto classify_label_path = config["classify_label_path"].as<std::string>();
  const std::string model_path =
      config["model"]["model_path"].as<std::string>();
  auto device_type = config["model"]["device_type"].as<std::string>();

  float conf_threshold = config["model"]["conf_threshold"].as<float>();
  int top_k = config["model"]["top_k"].as<int>();
  float nms_threshold = config["model"]["nms_threshold"].as<float>();
  gimbal2camera_x_ = config["tf"]["gimbal2camera_x"].as<double>(0.0);
  gimbal2camera_y_ = config["tf"]["gimbal2camera_y"].as<double>(0.0);
  gimbal2camera_z_ = config["tf"]["gimbal2camera_z"].as<double>(0.0);
  gimbal2camera_roll_ = config["tf"]["gimbal2camera_roll"].as<double>(0.0);
  gimbal2camera_pitch_ = config["tf"]["gimbal2camera_pitch"].as<double>(0.0);
  gimbal2camera_yaw_ = config["tf"]["gimbal2camera_yaw"].as<double>(0.0);

  float expand_ratio_w = config["light"]["expand_ratio_w"].as<float>();
  float expand_ratio_h = config["light"]["expand_ratio_h"].as<float>();
  int binary_thres = config["light"]["binary_thres"].as<int>();

  LightParams l_params = {
      .min_ratio = config["light"]["min_ratio"].as<double>(),
      .max_ratio = config["light"]["max_ratio"].as<double>(),
      .max_angle = config["light"]["max_angle"].as<double>()};

  if (model_path.empty()) {
    WUST_ERROR(vision_logger) << "Model path is empty.";
    return;
  }

  WUST_INFO(vision_logger) << "Model path: " << model_path.c_str();

  const std::string camera_info_path =
      config["camera"]["camera_info_path"].as<std::string>();
  measure_tool_ = std::make_unique<MonoMeasureTool>(camera_info_path);
  armor_pose_estimator_ =
      std::make_unique<ArmorPoseEstimator>(camera_info_path);

  initTF();
  bool use_serial = config["use_serial"].as<bool>();
  if (use_serial) {
    initSerial();
  }

  initTracker(config["tracker"]);
  detect_color_ = config["detect_color"].as<int>(0);
  max_infer_running_ = config["max_infer_running"].as<int>(4);
  detector_ = std::make_unique<OpenVino>(
      model_path, classify_model_path, classify_label_path, device_type,
      l_params, conf_threshold, top_k, nms_threshold, expand_ratio_w,
      expand_ratio_h, binary_thres);
  detector_->detect_color_ = detect_color_;

  detector_->setCallback(std::bind(&WustVision::DetectCallback, this,
                                   std::placeholders::_1, std::placeholders::_2,
                                   std::placeholders::_3));
  thread_pool_ =
      std::make_unique<ThreadPool>(std::thread::hardware_concurrency(), 100);
  solver_ = std::make_unique<Solver>(config);

  bool trigger_mode = config["camera"]["trigger_mode"].as<bool>(false);
  bool invert_image = config["camera"]["invert_image"].as<bool>(false);
  int exposure_time_us = config["camera"]["exposure_time_us"].as<int>(3500);
  float gain = config["camera"]["gain"].as<float>(7.0f);

  hikcamera::ImageCapturer::CameraProfile profile;
  profile.trigger_mode = trigger_mode;
  profile.invert_image = invert_image;
  profile.exposure_time = std::chrono::microseconds(exposure_time_us);
  profile.gain = gain;

  capturer_ = std::make_unique<hikcamera::ImageCapturer>(
      profile, nullptr, hikcamera::SyncMode::NONE);

  control_rate = config["control_rate"].as<int>(100);
  startTimer();
  capture_running_ = true;
  capture_thread_ =
      std::make_unique<std::thread>(&WustVision::captureLoop, this);

  is_inited_ = true;
}

void WustVision::captureLoop() {
  while (capture_running_ && is_inited_) {
    // auto start = std::chrono::high_resolution_clock::now();
    using namespace std::chrono_literals;

    auto frame = capturer_->read();
    
      auto now =  std::chrono::steady_clock::now();

    if (!frame.empty()) {

      thread_pool_->enqueue(
          [frame = std::move(frame), this, now]() {
            processImage(frame, now);
          });
    }
  }
}
void WustVision::stop() {
  is_inited_ = false;
  capture_running_ = false;
  if (capture_thread_ && capture_thread_->joinable()) {
    capture_thread_->join();
  }
  stopTimer();
  std::this_thread::sleep_for(std::chrono::seconds(1));

  detector_.reset();
  measure_tool_.reset();
  if (thread_pool_) {
    thread_pool_->waitUntilEmpty();
    thread_pool_.reset();
  }
  if (video_writer_.isOpened()) {
    video_writer_.release();
  }
  serial_.stopThread();

  if (thread_pool_) {
    thread_pool_->waitUntilEmpty();
  }

  WUST_INFO(vision_logger) << "WustVision shutdown complete.";
}
void WustVision::stopTimer() {
  timer_running_ = false;
  if (timer_thread_.joinable()) {
    timer_thread_.join();
  }
}
void WustVision::startTimer() {
  if (timer_running_)
    return;
  timer_running_ = true;
  int ms_interval = 1000 / control_rate;
  timer_thread_ = std::thread([this, &ms_interval]() {
    const auto interval = std::chrono::microseconds(ms_interval);
    auto next_time = std::chrono::steady_clock::now() + interval;

    while (timer_running_) {
      std::this_thread::sleep_until(next_time);
      if (!timer_running_)
        break;
      this->timerCallback();
      next_time += interval;
    }
  });
}
void WustVision::initTF() {
  // odom 是世界坐标系的根节点
  tf_tree_.setTransform("", "odom",
                        createTf(0, 0, 0, tf2::Quaternion(0, 0, 0, 1)));

  // camera 相对于 odom，设置 odom -> camera 的变换
  tf_tree_.setTransform("odom", "gimbal_odom",
                        createTf(0, 0, 0, tf2::Quaternion(0, 0, 0, 1)));
  tf_tree_.setTransform("gimbal_odom", "gimbal_link",
                        createTf(0, 0, 0, tf2::Quaternion(0, 0, 0, 1)));
  gimbal2camera_roll = gimbal2camera_roll_ * M_PI / 180;
  gimbal2camera_pitch = gimbal2camera_pitch_ * M_PI / 180;
  gimbal2camera_yaw = gimbal2camera_yaw_ * M_PI / 180;
  tf2::Quaternion origimbal2camera = eulerToQuaternion(
      gimbal2camera_roll, gimbal2camera_pitch, gimbal2camera_yaw);
  tf_tree_.setTransform("gimbal_link", "camera",
                        createTf(gimbal2camera_x_, gimbal2camera_y_,
                                 gimbal2camera_z_, origimbal2camera));

  // camera_optical_frame 相对于 camera，设置 camera -> camera_optical_frame
  // 的旋转变换
  double yaw = M_PI / 2;
  double roll = -M_PI / 2;
  double pitch = 0.0;

  tf2::Quaternion orientation;
  orientation.setRPY(roll, pitch, yaw);

  tf_tree_.setTransform("camera", "camera_optical_frame",
                        createTf(0, 0, 0, orientation));
}
void WustVision::initSerial() {
  SerialPortConfig cfg{/*baud*/ 115200, /*csize*/ 8,
                       boost::asio::serial_port_base::parity::none,
                       boost::asio::serial_port_base::stop_bits::one,
                       boost::asio::serial_port_base::flow_control::none};

  serial_.init("/dev/ttyACM0", cfg);
  serial_.startThread();
}

void WustVision::initTracker(const YAML::Node &config) {
  // 目标参考坐标系
  target_frame_ = config["target_frame"].as<std::string>("odom");

  // Tracker 基础参数
  double max_match_distance = config["max_match_distance"].as<double>(0.2);
  double max_match_yaw_diff = config["max_match_yaw_diff"].as<double>(1.0);
  double max_match_z_diff = config["max_match_z_diff"].as<double>(0.1);
  tracker_ = std::make_unique<Tracker>(max_match_distance, max_match_yaw_diff,
                                       max_match_z_diff);
  tracker_->buffer_size_ = config["obs_vyaw_buffer_thres"].as<int>(5);
  tracker_->obs_yaw_stationary_thresh =
      config["obs_yaw_stationary_thresh"].as<float>(1.0);
  tracker_->pred_yaw_stationary_thresh =
      config["pred_yaw_stationary_thresh"].as<float>(0.5);
  tracker_->min_valid_velocity =
      config["min_valid_velocity_thresh"].as<float>(0.01);
  tracker_->max_inconsistent_count_ =
      config["max_inconsistent_count"].as<int>(3);
  tracker_->rotation_inconsistent_cooldown_limit_ =
      config["rotation_inconsistent_cooldown_limit"].as<int>(5);
  tracker_->jump_thresh = config["jump_thresh"].as<double>(0.4);

  // 跟踪判定参数
  tracker_->tracking_thres = config["tracking_thres"].as<int>(5);
  lost_time_thres_ = config["lost_time_thres"].as<double>(0.3);

  // EKF 噪声参数
  s2qx_ = config["ekf"]["s2qx"].as<double>(20.0);
  s2qy_ = config["ekf"]["s2qy"].as<double>(20.0);
  s2qz_ = config["ekf"]["s2qz"].as<double>(20.0);
  s2qyaw_ = config["ekf"]["s2qyaw"].as<double>(100.0);
  s2qr_ = config["ekf"]["s2qr"].as<double>(800.0);
  s2qd_zc_ = config["ekf"]["s2qd_zc"].as<double>(800.0);

  r_x_ = config["ekf"]["r_x"].as<double>(0.05);
  r_y_ = config["ekf"]["r_y"].as<double>(0.05);
  r_z_ = config["ekf"]["r_z"].as<double>(0.05);
  r_yaw_ = config["ekf"]["r_yaw"].as<double>(0.02);

  // EKF 状态预测函数
  auto f = Predict(0.005); // dt 固定为 5ms

  // EKF 观测函数
  auto h = Measure();

  // EKF 过程噪声协方差 Q
  auto u_q = [this]() {
    Eigen::Matrix<double, X_N, X_N> q;
    double t = dt_, x = s2qx_, y = s2qy_, z = s2qz_, yaw = s2qyaw_, r = s2qr_,
           d_zc = s2qd_zc_;
    double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x,
           q_vx_vx = pow(t, 2) * x;
    double q_y_y = pow(t, 4) / 4 * y, q_y_vy = pow(t, 3) / 2 * y,
           q_vy_vy = pow(t, 2) * y;
    double q_z_z = pow(t, 4) / 4 * x, q_z_vz = pow(t, 3) / 2 * x,
           q_vz_vz = pow(t, 2) * z;
    double q_yaw_yaw = pow(t, 4) / 4 * yaw, q_yaw_vyaw = pow(t, 3) / 2 * x,
           q_vyaw_vyaw = pow(t, 2) * yaw;
    double q_r = pow(t, 4) / 4 * r;
    double q_d_zc = pow(t, 4) / 4 * d_zc;
    // clang-format off
        //    xc      v_xc    yc      v_yc    zc      v_zc    yaw         v_yaw       r       d_za
        q <<  q_x_x,  q_x_vx, 0,      0,      0,      0,      0,          0,          0,      0,
              q_x_vx, q_vx_vx,0,      0,      0,      0,      0,          0,          0,      0,
              0,      0,      q_y_y,  q_y_vy, 0,      0,      0,          0,          0,      0,
              0,      0,      q_y_vy, q_vy_vy,0,      0,      0,          0,          0,      0,
              0,      0,      0,      0,      q_z_z,  q_z_vz, 0,          0,          0,      0,
              0,      0,      0,      0,      q_z_vz, q_vz_vz,0,          0,          0,      0,
              0,      0,      0,      0,      0,      0,      q_yaw_yaw,  q_yaw_vyaw, 0,      0,
              0,      0,      0,      0,      0,      0,      q_yaw_vyaw, q_vyaw_vyaw,0,      0,
              0,      0,      0,      0,      0,      0,      0,          0,          q_r,    0,
              0,      0,      0,      0,      0,      0,      0,          0,          0,      q_d_zc;

    // clang-format on
    return q;
  };

  // EKF 观测噪声协方差 R（基于测量值调整）
  auto u_r = [this](const Eigen::Matrix<double, Z_N, 1> &z) {
    Eigen::Matrix<double, Z_N, Z_N> r;
    // clang-format off
        r << r_x_ * std::abs(z[0]), 0, 0, 0,
             0, r_y_ * std::abs(z[1]), 0, 0,
             0, 0, r_z_ * std::abs(z[2]), 0,
             0, 0, 0, r_yaw_;
    // clang-format on
    return r;
  };

  // 初始协方差
  Eigen::DiagonalMatrix<double, X_N> p0;
  p0.setIdentity();

  // 初始化 EKF 滤波器
  tracker_->ekf = std::make_unique<RobotStateEKF>(f, h, u_q, u_r, p0);
}
void WustVision::armorsCallback(Armors armors_, const cv::Mat &src_img) {
  transformArmorData(armors_);
  if (armors_.timestamp <= last_time_) {
    // WUST_WARN(vision_logger) << "Received out-of-order armor data,
    // discarded.";
    return;
  }

  if (debug_mode_) {
    std::lock_guard<std::mutex> target_lock(img_mutex_);
    imgframe_.img = src_img.clone();
    imgframe_.timestamp = armors_.timestamp;
    std::lock_guard<std::mutex> armor_gobal_lock(armors_gobal_mutex_);
    armors_gobal = armors_;
  }
  if (use_calculation_) {
    command_callback(armors_);
    return;
  }
  Target target_;
  auto time = armors_.timestamp;
  target_.timestamp = time;
  target_.frame_id = target_frame_;
  target_.type = tracker_->type;

  // Update tracker
  if (tracker_->tracker_state == Tracker::LOST) {
    tracker_->init(armors_);
    target_.tracking = false;
  } else {
    dt_ = std::chrono::duration<double>(time - last_time_).count();
    tracker_->lost_thres = std::abs(static_cast<int>(lost_time_thres_ / dt_));
    if (tracker_->tracked_id == ArmorNumber::OUTPOST) {
      tracker_->ekf->setPredictFunc(
          Predict{dt_, MotionModel::CONSTANT_ROTATION});
    } else {
      tracker_->ekf->setPredictFunc(
          Predict{dt_, MotionModel::CONSTANT_VEL_ROT});
    }
    tracker_->update(armors_);

    if (tracker_->tracker_state == Tracker::DETECTING) {
      target_.tracking = false;
    } else if (tracker_->tracker_state == Tracker::TRACKING ||
               tracker_->tracker_state == Tracker::TEMP_LOST) {
      target_.tracking = true;
      // Fill target
      const auto &state = tracker_->target_state;
      target_.id = tracker_->tracked_id;
      target_.armors_num = static_cast<int>(tracker_->tracked_armors_num);

      target_.position_.x = state(0);
      target_.velocity_.x = state(1);
      target_.position_.y = state(2);
      target_.velocity_.y = state(3);
      target_.position_.z = state(4);
      target_.velocity_.z = state(5);
      target_.yaw = state(6);
      target_.v_yaw = state(7);
      target_.radius_1 = state(8);
      target_.radius_2 = tracker_->another_r;
      target_.d_zc = state(9);
      target_.d_za = tracker_->d_za;
    }
  }

  target_.yaw_diff = tracker_->yaw_diff_;
  target_.position_diff = tracker_->position_diff_;
  {
    std::lock_guard<std::mutex> target_lock(armor_target_mutex_);
    armor_target = target_;
  }

  last_time_ = time;
}

Armors WustVision::visualizeTargetProjection(Target armor_target_) {

  Armors armor_data;
  armor_data.frame_id = "odom";
  armor_data.timestamp = armor_target_.timestamp;

  if (armor_target_.tracking) {
    double yaw = armor_target_.yaw, r1 = armor_target_.radius_1,
           r2 = armor_target_.radius_2;
    float xc = armor_target_.position_.x, yc = armor_target_.position_.y,
          zc = armor_target_.position_.z;
    double d_za = armor_target_.d_za, d_zc = armor_target_.d_zc;
    xc = xc + armor_target_.velocity_.x * debug_show_dt_;
    yc = yc + armor_target_.velocity_.y * debug_show_dt_;
    zc = zc + armor_target_.velocity_.z * debug_show_dt_;
    yaw = yaw + armor_target_.v_yaw * debug_show_dt_;

    bool is_current_pair = true;

    armor_data.armors.clear();

    size_t a_n = armor_target_.armors_num;

    armor_data.armors.reserve(a_n);

    for (size_t i = 0; i < a_n; ++i) {
      double tmp_yaw = yaw + i * (2 * M_PI / a_n);
      double cos_yaw = std::cos(tmp_yaw);
      double sin_yaw = std::sin(tmp_yaw);

      Position pos;
      if (a_n == 4) {
        double r = is_current_pair ? r1 : r2;
        pos.z = zc + d_zc + (is_current_pair ? 0 : d_za);
        pos.x = xc - r * cos_yaw;
        pos.y = yc - r * sin_yaw;
        is_current_pair = !is_current_pair;
      } else {
        pos.z = zc;
        pos.x = xc - r1 * cos_yaw;
        pos.y = yc - r1 * sin_yaw;
      }

      tf2::Quaternion ori;
      ori.setRPY(M_PI / 2,
                 armor_target_.id == ArmorNumber::OUTPOST ? -0.2618 : 0.2618,
                 tmp_yaw);

      armor_data.armors.emplace_back(Armor{.type = armor_target_.type,
                                           .pos = pos,
                                           .ori = ori,
                                           //.target_pos = {xc, yc, zc},
                                           .distance_to_image_center = 0.0f});
    }
  }
  return armor_data;
}
void WustVision::DetectCallback(const std::vector<ArmorObject> &objs,
                                 std::chrono::steady_clock::time_point timestamp,
                                const cv::Mat &src_img) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  detect_finish_count_++;
  if (objs.size() >= 6) {
    WUST_WARN(vision_logger) << "Detected " << objs.size() << " objects"
                             << "too much";
    infer_running_count_--;
    return;
  }
  if (measure_tool_ == nullptr) {
    WUST_WARN(vision_logger) << "NO camera info";
    return;
  }
  Armors armors;
  armors.timestamp = timestamp;
  armors.frame_id = "camera_optical_frame";
  try {
    auto target_time = armors.timestamp;
    Transform tf;
    if (!tf_tree_.getTransform(armors.frame_id, target_frame_, target_time,
                               tf)) {
      throw std::runtime_error("Transform not found.");
    }

    tf2::Quaternion tf_quat = tf.orientation;
    // std::cout<<tf.orientation.x<<" "<<tf.orientation.y<<"
    // "<<tf.orientation.z<<" "<<tf.orientation.w<<std::endl;
    Eigen::Quaterniond eigen_quat(tf_quat.w, tf_quat.x, tf_quat.y, tf_quat.z);
    imu_to_camera_ = eigen_quat.toRotationMatrix(); // Eigen::Matrix3d
    imu_to_camera_ =
        Sophus::SO3d::fitToSO3(eigen_quat.toRotationMatrix()).matrix();
    //  std::cout<<imu_to_camera_<<std::endl;

  } catch (const std::exception &e) {

    return;
  }
  armors.armors =
      armor_pose_estimator_->extractArmorPoses(objs, imu_to_camera_);

  measure_tool_->processDetectedArmors(objs, detect_color_, armors);

  infer_running_count_--;
  armorsCallback(armors, src_img);

  // thread_pool_->enqueue([this, armors = std::move(armors), src_img]() {
  //   this->armorsCallback(armors, src_img);
  // });

  // drawresult(src_img,objs,timestamp_nanosec);
}
void WustVision::transformArmorData(Armors &armors) {
  for (auto &armor : armors.armors) {
    try {
      Transform tf(armor.pos, armor.ori, armors.timestamp);
      auto pose_in_target_frame = tf_tree_.transform(
          tf, armors.frame_id, target_frame_, armors.timestamp);
      // auto pose_in_target_frame = tf_tree_.transform(tf,
      // target_frame_,armors.frame_id,  armors.timestamp);
      armor.target_pos = pose_in_target_frame.position;
      armor.target_ori = pose_in_target_frame.orientation;

      armor.yaw = getRPYFromQuaternion(armor.target_ori).yaw;
      double yaw = armor.yaw * 180 / M_PI;
      auto now = std::chrono::steady_clock::now();
      std::cout << "now (ns since epoch): " << now.time_since_epoch().count() << " ns" << std::endl;
      std::cout << "timestamp (ns): " << pose_in_target_frame.timestamp.time_since_epoch().count() << " ns" << std::endl;
      // std::cout<<"YAW:"<<yaw<<std::endl;

      //  WUST_DEBUG(vision_logger)<<"Z:"<<armor.target_pos.z;
    } catch (const std::exception &e) {
      WUST_ERROR(vision_logger)
          << "Can't find transform from " << armors.frame_id << " to "
          << target_frame_ << ": " << e.what();
      return;
    }
  }
}

void WustVision::timerCallback() {
  static GimbalCmd aa;

  if (!is_inited_)
    return;

  Target target;
  {
    std::lock_guard<std::mutex> lock(armor_target_mutex_);
    target = armor_target;
  }
  bool appear ;
  if(tracker_->tracker_state == Tracker::LOST)
  {
    appear = false;
  }else
  {
    appear = true;
  }
  auto now = std::chrono::steady_clock::now();
  auto latency_nano = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          now - target.timestamp)
                          .count();
  latency_ms = static_cast<double>(latency_nano) / 1e6;
  GimbalCmd gimbal_cmd;
  
  if (target.id != ArmorNumber::UNKNOWN) {
    if (target.tracking) {
      try {
        auto now = std::chrono::steady_clock::now();
        gimbal_cmd = solver_->solve(target, now);
        aa=gimbal_cmd;
        if (gimbal_cmd.fire_advice) {
          fire_count_++;
        }
        serial_.transformGimbalCmd(gimbal_cmd,appear);
      } catch (...) {
        WUST_ERROR(vision_logger) << "solver error";
        serial_.transformGimbalCmd(aa,appear);
      }
    } else {
      serial_.transformGimbalCmd(aa,appear);
    }
  } else {
    serial_.transformGimbalCmd(aa,appear);
  }
  
  if (debug_mode_) {
    Armors armor_data = visualizeTargetProjection(target);

    for (auto &armor : armor_data.armors) {
      try {
        Transform tf(armor.pos, armor.ori);
        auto pose_in_target_frame =
            tf_tree_.transform(tf, armor_data.frame_id, "camera_optical_frame",target.timestamp);
        // auto pose_in_target_frame = tf_tree_.transform(tf,
        // "camera_optical_frame", armor_data.frame_id, armor_data.timestamp);
        armor.target_pos = pose_in_target_frame.position;
        armor.target_ori = pose_in_target_frame.orientation;
      } catch (const std::exception &e) {
        WUST_ERROR(vision_logger)
            << "Can't find transform from " << armor_data.frame_id << " to "
            << target_frame_ << ": " << e.what();
        continue;
      }
    }
    Target_info target_info;
    target_info.select_id = gimbal_cmd.select_id;
    
    if (!measure_tool_->reprojectArmorsCorners(armor_data, target_info))
      return;
    dumpTargetToFile(target, "/tmp/target_status.txt");
    Tracker::State state = tracker_->tracker_state;
    cv::Mat src;
    {
      std::lock_guard<std::mutex> lock(img_mutex_);
      src = imgframe_.img.clone();
    }

    

    Armors armors;
    {
      std::lock_guard<std::mutex> lock(armors_gobal_mutex_);
      armors = armors_gobal;
    }

    draw_debug_overlay(imgframe_, &armors, &target_info, &target, state,
                       gimbal_cmd);
  }
}

void WustVision::processImage(const cv::Mat &frame,std::chrono::steady_clock::time_point timestamp) {

  img_recv_count_++;
  if (infer_running_count_.load() >= max_infer_running_) {
    return;
  }

  infer_running_count_++;
  printStats();
  detector_->pushInput(frame, timestamp);
}

void WustVision::printStats() {
  using namespace std::chrono;

  auto now = steady_clock::now();

  if (last_stat_time_steady_.time_since_epoch().count() == 0) {
    last_stat_time_steady_ = now;
    return;
  }

  auto elapsed = duration_cast<duration<double>>(now - last_stat_time_steady_);
  if (elapsed.count() >= 1.0) {
    WUST_INFO(vision_logger)
        << "Received: " << img_recv_count_
        << ", Detected: " << detect_finish_count_
        << ", FPS: " << detect_finish_count_ / elapsed.count()
        << " Latency: " << latency_ms << "ms"
        << "  Fire: " << fire_count_;

    img_recv_count_ = 0;
    detect_finish_count_ = 0;
    fire_count_ = 0;
    last_stat_time_steady_ = now;
  }
}

WustVision *global_vision = nullptr;
std::mutex mtx;
std::condition_variable c;
bool exit_flag = false;

void signalHandler(int signum) {
  WUST_INFO("main") << "Interrupt signal (" << signum << ") received.";
  if (global_vision) {
    global_vision->stop();
  }
  {
    std::lock_guard<std::mutex> lk(mtx);
    exit_flag = true;
  }
  c.notify_one();
}

int main() {
  WustVision vision;
  global_vision = &vision;

  std::signal(SIGINT, signalHandler);

  {
    std::unique_lock<std::mutex> lk(mtx);
    c.wait(lk, [] { return exit_flag; });
  }

  return 0;
}
