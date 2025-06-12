#include "wust_vision_openvino.hpp"
#include "common/calculation.hpp"
#include "common/gobal.hpp"
#include "common/logger.hpp"
#include "common/matplottools.hpp"
#include "common/tf.hpp"
#include "common/tools.hpp"
#include "common/toolsgobal.hpp"
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
WustVision::WustVision() { init(); }
WustVision::~WustVision() {}
void WustVision::stop() {
  is_inited_ = false;
  // capture_running_ = false;
  // if (capture_thread_ && capture_thread_->joinable()) {
  //   capture_thread_->join();
  // }
  if (!only_nav_enable) {
    camera_->stopCamera();

    stopTimer();
    std::this_thread::sleep_for(std::chrono::seconds(1));

    detector_.reset();
    measure_tool_.reset();
    camera_.reset();
    if (thread_pool_) {
      thread_pool_->waitUntilEmpty();
      thread_pool_.reset();
    }
    if (robot_cmd_plot_thread_.joinable()) {
      robot_cmd_plot_thread_.join();
    }
    if (thread_pool_) {
      thread_pool_->waitUntilEmpty();
    }
  }
  serial_.stopThread();

  WUST_INFO(vision_logger) << "WustVision shutdown complete.";
}
void WustVision::init() {
  config = YAML::LoadFile("/home/hy/wust_vision/config/config_openvino.yaml");
  std::string log_level_ =
      config["logger"]["log_level"].as<std::string>("INFO");
  std::string log_path_ =
      config["logger"]["log_path"].as<std::string>("wust_log");
  bool use_logcli = config["logger"]["use_logcli"].as<bool>();
  bool use_logfile = config["logger"]["use_logfile"].as<bool>();
  bool use_simplelog = config["logger"]["use_simplelog"].as<bool>();
  initLogger(log_level_, log_path_, use_logcli, use_logfile, use_simplelog);
  control_rate = config["control"]["control_rate"].as<int>();
  only_nav_enable = config["only_nav_enable"].as<bool>();
  if (!only_nav_enable) {
    attack_mode = config["init_attack_mode"].as<int>();
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
    odom2gimbal_pitch = config["tf"]["odom2gimbal_pitch"].as<double>();
    odom2gimbal_roll = config["tf"]["odom2gimbal_roll"].as<double>();
    odom2gimbal_yaw = config["tf"]["odom2gimbal_yaw"].as<double>();
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
    initTracker(config["tracker"]);
    detect_color_ = config["detect_color"].as<int>(0);
    max_infer_running_ = config["max_infer_running"].as<int>(4);
    detector_ = std::make_unique<OpenVino>(
        model_path, classify_model_path, classify_label_path, device_type,
        l_params, conf_threshold, top_k, nms_threshold, expand_ratio_w,
        expand_ratio_h, binary_thres);
    detector_->detect_color_ = detect_color_;

    detector_->setCallback(
        std::bind(&WustVision::DetectCallback, this, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3));
    initRune(camera_info_path);

    thread_pool_ =
        std::make_unique<ThreadPool>(std::thread::hardware_concurrency(), 100);
    solver_ = std::make_unique<Solver>(config);
    std::string camera_serial = config["camera"]["serial"].as<std::string>("");
    camera_ = std::make_unique<HikCamera>();
    if (!camera_->initializeCamera(camera_serial)) {
      WUST_ERROR(vision_logger) << "Camera initialization failed.";
      return;
    }

    camera_->setParameters(config["camera"]["acquisition_frame_rate"].as<int>(),
                           config["camera"]["exposure_time"].as<int>(),
                           config["camera"]["gain"].as<double>(),
                           config["camera"]["adc_bit_depth"].as<std::string>(),
                           config["camera"]["pixel_format"].as<std::string>());
    camera_->setFrameCallback([this](const ImageFrame &frame) {
      static bool first_is_inited = false;

      if (is_inited_) {
        thread_pool_->enqueue(
            [frame = std::move(frame), this]() { processImage(frame); });
      } else {
        return;
      }
    });

    camera_->startCamera();
    // bool trigger_mode = config["camera"]["trigger_mode"].as<bool>(false);
    // bool invert_image = config["camera"]["invert_image"].as<bool>(false);
    // int exposure_time_us =
    // config["camera"]["exposure_time_us"].as<int>(3500); float gain =
    // config["camera"]["gain"].as<float>(7.0f);

    // hikcamera::ImageCapturer::CameraProfile profile;
    // profile.trigger_mode = trigger_mode;
    // profile.invert_image = invert_image;
    // profile.exposure_time = std::chrono::microseconds(exposure_time_us);
    // profile.gain = gain;

    // capturer_ = std::make_unique<hikcamera::ImageCapturer>(
    //     profile, nullptr, hikcamera::SyncMode::NONE);
    // capture_running_ = true;
    // capture_thread_ =
    //     std::make_unique<std::thread>(&WustVision::captureLoop, this);
    startTimer();
    robot_cmd_plot_thread_ = std::thread(&robotCmdLoggerThread);
  } else {
    WUST_INFO(vision_logger) << "only nav mode";
  }

  initSerial();
  is_inited_ = true;
}
void WustVision::initRune(const std::string &camera_info_path) {
  rune_detector_ = initRuneDetector();
  auto rune_solver_params = RuneSolver::RuneSolverParams{
      .compensator_type =
          config["rune_solver"]["compensator_type"].as<std::string>(),
      .gravity = config["rune_solver"]["gravity"].as<double>(9.8),
      .bullet_speed = config["rune_solver"]["bullet_speed"].as<double>(25.0),
      .angle_offset_thres =
          config["rune_solver"]["angle_offset_thres"].as<double>(0.78),
      .lost_time_thres =
          config["rune_solver"]["lost_time_thres"].as<double>(0.5),
      .auto_type_determined =
          config["rune_solver"]["auto_type_determined"].as<bool>(true),
  };

  rune_solver_ = std::make_unique<RuneSolver>(rune_solver_params);
  rune_solver_->predict_offset_ =
      config["rune_solver"]["predict_offset"].as<double>(0.0);
  YAML::Node camera_config = YAML::LoadFile(camera_info_path);

  std::array<double, 9> camera_k =
      camera_config["camera_matrix"]["data"].as<std::array<double, 9>>();
  std::vector<double> camera_d =
      camera_config["distortion_coefficients"]["data"]
          .as<std::vector<double>>();
  rune_solver_->pnp_solver = std::make_unique<PnPSolver>(camera_k, camera_d);
  rune_solver_->pnp_solver->setObjectPoints("rune", RUNE_OBJECT_POINTS);
  // EKF for filtering the position of R tag
  // state: x, y, z, yaw
  // measurement: x, y, z, yaw
  // f - Process function
  auto f = rune_motion_model::Predict();
  // h - Observation function
  auto h = rune_motion_model::Measure();
  // update_Q - process noise covariance matrix
  std::vector<double> q_vec =
      config["rune_solver"]["ekf"]["q"].as<std::vector<double>>();

  auto u_q = [q_vec]() {
    Eigen::Matrix<double, rune_motion_model::X_N, rune_motion_model::X_N> q =
        Eigen::MatrixXd::Zero(4, 4);
    q.diagonal() << q_vec[0], q_vec[1], q_vec[2], q_vec[3];
    return q;
  };
  // update_R - measurement noise covariance matrix
  std::vector<double> r_vec =
      config["rune_solver"]["ekf"]["r"].as<std::vector<double>>();
  auto u_r = [r_vec](
                 const Eigen::Matrix<double, rune_motion_model::Z_N, 1> &z) {
    Eigen::Matrix<double, rune_motion_model::Z_N, rune_motion_model::Z_N> r =
        Eigen::MatrixXd::Zero(4, 4);
    r.diagonal() << r_vec[0], r_vec[1], r_vec[2], r_vec[3];
    return r;
  };
  // P - error estimate covariance matrix
  Eigen::MatrixXd p0 = Eigen::MatrixXd::Identity(4, 4);
  rune_solver_->ekf =
      std::make_unique<rune_motion_model::RuneCenterEKF>(f, h, u_q, u_r, p0);
}
std::unique_ptr<RuneDetector> WustVision::initRuneDetector() {
  rune_binary_thresh_ = config["rune_detector"]["min_lightness"].as<int>(100);
  detect_r_tag_ = config["rune_detector"]["detect_r_tag"].as<bool>(false);
  std::string model_path = config["rune_detector"]["model"].as<std::string>();

  std::string device_type =
      config["rune_detector"]["device_type"].as<std::string>("CPU");

  WUST_INFO("rune_detector")
      << "model : " << model_path << " device_type : " << device_type;

  float conf_threshold =
      config["rune_detector"]["confidence_threshold"].as<float>(0.50);
  int top_k = config["rune_detector"]["top_k"].as<int>(128);
  float nms_threshold = config["rune_detector"]["nms_threshold"].as<float>(0.3);

  // Create detector
  auto rune_detector = std::make_unique<RuneDetector>(
      model_path, device_type, conf_threshold, top_k, nms_threshold);
  // Set detect callback
  rune_detector->setCallback(
      std::bind(&WustVision::inferResultCallback, this, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3));
  // init detector
  rune_detector->init();
  return rune_detector;
}

void WustVision::captureLoop() {
  while (capture_running_ && is_inited_) {
    // auto start = std::chrono::high_resolution_clock::now();
    using namespace std::chrono_literals;

    auto frame = capturer_->read();

    auto now = std::chrono::steady_clock::now();

    if (!frame.empty()) {

      thread_pool_->enqueue([frame = std::move(frame), this, now]() {
        processImage(frame, now);
      });
    }
  }
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
                        createTf(0, 0, 0, tf2::Quaternion(0, 0, 0, 1)), true);

  // camera 相对于 odom，设置 odom -> camera 的变换
  tf_tree_.setTransform("odom", "gimbal_odom",
                        createTf(0, 0, 0, tf2::Quaternion(0, 0, 0, 1)), true);
  double odom2gimbal_roll_ = odom2gimbal_roll * M_PI / 180;
  double odom2gimbal_pitch_ = odom2gimbal_pitch * M_PI / 180;
  double odom2gimbal_yaw_ = odom2gimbal_yaw * M_PI / 180;
  tf2::Quaternion oriodom2gimbal;
  oriodom2gimbal.setRPY(odom2gimbal_roll_, odom2gimbal_pitch_,
                        odom2gimbal_yaw_);

  tf_tree_.setTransform("gimbal_odom", "gimbal_link",
                        createTf(0, 0, 0, oriodom2gimbal), false);
  gimbal2camera_roll = gimbal2camera_roll_ * M_PI / 180;
  gimbal2camera_pitch = gimbal2camera_pitch_ * M_PI / 180;
  gimbal2camera_yaw = gimbal2camera_yaw_ * M_PI / 180;
  tf2::Quaternion origimbal2camera;
  origimbal2camera.setRPY(gimbal2camera_roll, gimbal2camera_pitch,
                          gimbal2camera_yaw);
  tf_tree_.setTransform("gimbal_link", "camera",
                        createTf(gimbal2camera_x_, gimbal2camera_y_,
                                 gimbal2camera_z_, origimbal2camera),
                        true);

  // camera_optical_frame 相对于 camera，设置 camera -> camera_optical_frame
  // 的旋转变换
  double yaw = M_PI / 2;
  double roll = -M_PI / 2;
  double pitch = 0.0;

  tf2::Quaternion orientation;
  orientation.setRPY(roll, pitch, yaw);

  tf_tree_.setTransform("camera", "camera_optical_frame",
                        createTf(0, 0, 0, orientation), true);
}
void WustVision::initSerial() {
  SerialPortConfig cfg{/*baud*/ 115200, /*csize*/ 8,
                       boost::asio::serial_port_base::parity::none,
                       boost::asio::serial_port_base::stop_bits::one,
                       boost::asio::serial_port_base::flow_control::none};

  std::string device_name = config["control"]["device_name"].as<std::string>();
  serial_.init(device_name, cfg);
  serial_.alpha_yaw = config["control"]["alpha_yaw"].as<double>();
  serial_.alpha_pitch = config["control"]["alpha_pitch"].as<double>();
  serial_.max_yaw_change = config["control"]["max_yaw_change"].as<double>();
  serial_.max_pitch_change = config["control"]["max_pitch_change"].as<double>();
  bool if_use_nav = config["control"]["use_nav"].as<bool>(false);
  use_serial = config["control"]["use_serial"].as<bool>();
  serial_.startThread(use_serial, if_use_nav);
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
  auto f = armor_motion_model::Predict(0.005); // dt 固定为 5ms

  // EKF 观测函数
  auto h = armor_motion_model::Measure();

  // EKF 过程噪声协方差 Q
  auto u_q = [this]() {
    Eigen::Matrix<double, armor_motion_model::X_N, armor_motion_model::X_N> q;
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
  auto u_r = [this](
                 const Eigen::Matrix<double, armor_motion_model::Z_N, 1> &z) {
    Eigen::Matrix<double, armor_motion_model::Z_N, armor_motion_model::Z_N> r;
    // clang-format off
        r << r_x_ * std::abs(z[0]), 0, 0, 0,
             0, r_y_ * std::abs(z[1]), 0, 0,
             0, 0, r_z_ * std::abs(z[2]), 0,
             0, 0, 0, r_yaw_;
    // clang-format on
    return r;
  };

  // 初始协方差
  Eigen::DiagonalMatrix<double, armor_motion_model::X_N> p0;
  p0.setIdentity();

  // 初始化 EKF 滤波器
  tracker_->ekf =
      std::make_unique<armor_motion_model::RobotStateEKF>(f, h, u_q, u_r, p0);
}
void WustVision::runeTargetCallback(const Rune rune_target) {
  // rune_solver_->pnp_solver is nullptr when camera_info is not received
  if (rune_solver_->pnp_solver == nullptr) {
    return;
  }

  // Keep the last detected target
  if (!rune_target.is_lost) {
    last_rune_target_ = rune_target;
  }
  double observed_angle = 0;
  if (rune_solver_->tracker_state == RuneSolver::LOST) {
    observed_angle = rune_solver_->init(rune_target);
  } else {
    observed_angle = rune_solver_->update(rune_target);
  }
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
    // return;
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
      tracker_->ekf->setPredictFunc(armor_motion_model::Predict{
          dt_, armor_motion_model::MotionModel::CONSTANT_ROTATION});
    } else {
      tracker_->ekf->setPredictFunc(armor_motion_model::Predict{
          dt_, armor_motion_model::MotionModel::CONSTANT_VEL_ROT});
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
    // std::cout<<imu_to_camera_<<std::endl;

  } catch (const std::exception &e) {

    return;
  }
  armors.armors =
      armor_pose_estimator_->extractArmorPoses(objs, imu_to_camera_);

  // measure_tool_->processDetectedArmors(objs, detect_color_, armors);

  infer_running_count_--;
  armorsCallback(armors, src_img);
}
void WustVision::inferResultCallback(
    std::vector<RuneObject> &objs,
    std::chrono::steady_clock::time_point timestamp,
    const cv::Mat &src_img) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  detect_finish_count_++;
  // Used to draw debug info
  cv::Mat debug_img;
  if (debug_mode_) {
    debug_img = src_img.clone();
  }
  Rune rune_target;
  rune_target.frame_id = "camera_optical_frame";
  rune_target.timestamp = timestamp;
  rune_target.is_big_rune = false;

  // Erase all object that not match the color
  objs.erase(std::remove_if(objs.begin(), objs.end(),
                            [c = static_cast<EnemyColor>(detect_color_)](
                                const auto &obj) { return obj.color != c; }),
             objs.end());

  if (!objs.empty()) {
    // Sort by probability
    std::sort(objs.begin(), objs.end(),
              [](const RuneObject &a, const RuneObject &b) {
                return a.prob > b.prob;
              });

    cv::Point2f r_tag;
    cv::Mat binary_roi = cv::Mat::zeros(1, 1, CV_8UC3);
    if (detect_r_tag_) {
      // Detect R tag using traditional method
      std::tie(r_tag, binary_roi) = rune_detector_->detectRTag(
          src_img, rune_binary_thresh_, objs.at(0).pts.r_center);
    } else {
      // Use the average center of all objects as the center of the R tag
      r_tag = std::accumulate(
          objs.begin(), objs.end(), cv::Point2f(0, 0),
          [n = static_cast<float>(objs.size())](cv::Point2f p, auto &o) {
            return p + o.pts.r_center / n;
          });
    }
    // Assign the center of the R tag to all objects
    std::for_each(objs.begin(), objs.end(),
                  [r = r_tag](RuneObject &obj) { obj.pts.r_center = r; });

    // Draw binary roi
    if (debug_mode_ && !debug_img.empty()) {
      cv::Rect roi = cv::Rect(debug_img.cols - binary_roi.cols, 0,
                              binary_roi.cols, binary_roi.rows);
      binary_roi.copyTo(debug_img(roi));
      cv::rectangle(debug_img, roi, cv::Scalar(150, 150, 150), 2);
    }

    // The final target is the inactivated rune with the highest probability
    auto result_it = std::find_if(
        objs.begin(), objs.end(),
        [c = static_cast<EnemyColor>(detect_color_)](const auto &obj) -> bool {
          return obj.type == RuneType::INACTIVATED && obj.color == c;
        });

    if (result_it != objs.end()) {
      rune_target.is_lost = false;
      rune_target.pts[0].x = result_it->pts.r_center.x;
      rune_target.pts[0].y = result_it->pts.r_center.y;
      rune_target.pts[1].x = result_it->pts.bottom_left.x;
      rune_target.pts[1].y = result_it->pts.bottom_left.y;
      rune_target.pts[2].x = result_it->pts.top_left.x;
      rune_target.pts[2].y = result_it->pts.top_left.y;
      rune_target.pts[3].x = result_it->pts.top_right.x;
      rune_target.pts[3].y = result_it->pts.top_right.y;
      rune_target.pts[4].x = result_it->pts.bottom_right.x;
      rune_target.pts[4].y = result_it->pts.bottom_right.y;
    } else {
      // All runes are activated
      rune_target.is_lost = true;
    }
  } else {
    // All runes are not the target color
    rune_target.is_lost = true;
  }
  infer_running_count_--;
  runeTargetCallback(rune_target);
  {
    std::lock_guard<std::mutex> target_lock(rune_gobal_mutex_);
    rune_gobal = rune_target;
  }
  if (debug_mode_) {
    drawRune(debug_img, objs, timestamp);
  }
}
void WustVision::transformArmorData(Armors &armors) {
  for (auto &armor : armors.armors) {

    try {
      Transform tf(armor.pos, armor.ori, armors.timestamp);
      auto pose_in_target_frame = tf_tree_.transform(
          tf, armors.frame_id, target_frame_, armors.timestamp);

      armor.target_pos = pose_in_target_frame.position;
      armor.target_ori = pose_in_target_frame.orientation;

      armor.yaw = getRPYFromQuaternion(armor.target_ori).yaw;
      double yaw = armor.yaw * 180 / M_PI;

    } catch (const std::exception &e) {
      WUST_ERROR(vision_logger)
          << "Can't find transform from " << armors.frame_id << " to "
          << target_frame_ << ": " << e.what();
      return;
    }
  }
}

void WustVision::timerCallback() {

  if (!is_inited_)
    return;

  Target target;
  {
    std::lock_guard<std::mutex> lock(armor_target_mutex_);
    target = armor_target;
  }
  Rune rune;
  {
    std::lock_guard<std::mutex> lock(rune_gobal_mutex_);
    rune = rune_gobal;
  }
  bool appear;
  if (tracker_->tracker_state == Tracker::LOST) {
    appear = false;
  } else {
    appear = true;
  }
  auto now = std::chrono::steady_clock::now();
  AttackMode mode = toAttackMode(attack_mode);
  
  if(mode == AttackMode::ARMOR)
  {
  auto latency_nano = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          now - target.timestamp)
                          .count();
  latency_ms = static_cast<double>(latency_nano) / 1e6;
  }else {
    auto latency_nano = std::chrono::duration_cast<std::chrono::nanoseconds>(
      now - rune.timestamp)
      .count();
  latency_ms = static_cast<double>(latency_nano) / 1e6;
  }
  GimbalCmd gimbal_cmd;

  if (target.id != ArmorNumber::UNKNOWN) {
    if (target.tracking) {
      try {
        
        switch (mode) {
        case AttackMode::ARMOR: {
          gimbal_cmd = solver_->solve(target, now);
        } break;
        case AttackMode::SMALL_RUNE: {
          gimbal_cmd = rune_solver_->solve();
        } break;
        case AttackMode::BIG_RUNE: {
          gimbal_cmd = rune_solver_->solve();
        }
        case AttackMode::UNKNOWN:
          break;
        }
        last_cmd_ = gimbal_cmd;
        if (gimbal_cmd.fire_advice) {
          fire_count_++;
        }
        serial_.transformGimbalCmd(gimbal_cmd, appear);
      } catch (...) {
        WUST_ERROR(vision_logger) << "solver error";
        serial_.transformGimbalCmd(last_cmd_, appear);
      }
    } else {
      serial_.transformGimbalCmd(last_cmd_, appear);
    }
  } else {

    serial_.transformGimbalCmd(last_cmd_, appear);
  }

  if (debug_mode_) {
    Armors armor_data = visualizeTargetProjection(target);

    for (auto &armor : armor_data.armors) {
      try {
        Transform tf(armor.pos, armor.ori);
        auto pose_in_target_frame = tf_tree_.transform(
            tf, armor_data.frame_id, "camera_optical_frame", target.timestamp);

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
    write_target_log_to_json(target);
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

    draw_debug_overlaywrite(imgframe_, &armors, &target_info, &target, state,
                            gimbal_cmd);

    auto now = std::chrono::steady_clock::now();
    double t = std::chrono::duration<double>(now - start_time_).count();
    {
      std::lock_guard<std::mutex> lock(yaw_log_mutex_);

      target_yaw_log_.emplace_back(t, target.yaw);
      if (target_yaw_log_.size() > 1000) {
        target_yaw_log_.erase(target_yaw_log_.begin(),
                              target_yaw_log_.begin() + target_yaw_log_.size() -
                                  1000);
      }
    }
    {
      std::lock_guard<std::mutex> lock(robot_cmd_mutex_);
      time_log_.push_back(t);
      cmd_yaw_log_.push_back(last_cmd_.yaw);
      cmd_pitch_log_.push_back(last_cmd_.pitch);
      if (!armors.armors.empty()) {
        auto min_armor_it = std::min_element(
            armors.armors.begin(), armors.armors.end(),
            [](const Armor &a, const Armor &b) {
              return a.distance_to_image_center < b.distance_to_image_center;
            });
        const Armor &min_armor = *min_armor_it;
        last_distance =
            std::sqrt(min_armor.target_pos.x * min_armor.target_pos.x +
                      min_armor.target_pos.y * min_armor.target_pos.y +
                      min_armor.target_pos.z * min_armor.target_pos.z);
        armor_dis_log_.push_back(last_distance);
      } else {
        armor_dis_log_.push_back(last_distance);
      }

      if (time_log_.size() > 100) {
        time_log_.erase(time_log_.begin());
        cmd_yaw_log_.erase(cmd_yaw_log_.begin());
        cmd_pitch_log_.erase(cmd_pitch_log_.begin());
        armor_dis_log_.erase(armor_dis_log_.begin());
      }
    }
  }
}
void WustVision::processImage(const ImageFrame &frame) {

  img_recv_count_++;
  if (infer_running_count_.load() >= max_infer_running_) {

    return;
  }

  cv::Mat img = convertToMat(frame);
  infer_running_count_++;
  printStats();
  AttackMode mode = toAttackMode(attack_mode);
  switch (mode) {
  case AttackMode::ARMOR: {
    detector_->pushInput(img, frame.timestamp);
  } break;
  case AttackMode::SMALL_RUNE: {
    rune_detector_->pushInput(img, frame.timestamp);
  } break;
  case AttackMode::BIG_RUNE: {
    rune_detector_->pushInput(img, frame.timestamp);
  }
  case AttackMode::UNKNOWN:
    break;
  }
}
void WustVision::processImage(const cv::Mat &frame,
                              std::chrono::steady_clock::time_point timestamp) {

  img_recv_count_++;
  if (infer_running_count_.load() >= max_infer_running_) {
    return;
  }

  infer_running_count_++;
  printStats();
  AttackMode mode = toAttackMode(attack_mode);
  switch (mode) {
  case AttackMode::ARMOR: {
    detector_->pushInput(frame, timestamp);
  } break;
  case AttackMode::SMALL_RUNE: {
    rune_detector_->pushInput(frame, timestamp);
  } break;
  case AttackMode::BIG_RUNE: {
    rune_detector_->pushInput(frame, timestamp);
  }
  case AttackMode::UNKNOWN:
    break;
  }
  

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

void signalHandler(int signum) {
  WUST_INFO("main") << "Interrupt signal (" << signum << ") received.";
  exit_flag.store(true, std::memory_order_release);
}

int main() {
  WustVision vision;
  global_vision = &vision;

  std::signal(SIGINT, signalHandler);

  std::thread wait_thread([] {
    while (!exit_flag.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    c.notify_one();
  });

  {
    std::unique_lock<std::mutex> lk(mtx);
    c.wait(lk, [] { return exit_flag.load(std::memory_order_acquire); });
  }

  wait_thread.join();
  vision.stop();
  return 0;
}