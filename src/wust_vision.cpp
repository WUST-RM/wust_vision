#include "wust_vision.hpp"
#include "common/logger.hpp"
#include "common/tf.hpp"
#include "type/type.hpp"
#include <csignal>
#include <iostream>
#include <vector>
#include "string"

WustVision::WustVision()
{
    init();
}
WustVision::~WustVision() {
    WUST_INFO(vision_logger) << "Shutting down WustVision...";


    is_inited_ = false;

    camera_.stopCamera();


    
    detector_.reset();
  
    measure_tool_.reset();


    thread_pool_.reset();
    if (thread_pool_) {
        thread_pool_->waitUntilEmpty(); 
    }


    WUST_INFO(vision_logger) << "WustVision shutdown complete.";
}
void WustVision::stop() {
    is_inited_ = false;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    camera_.stopCamera();  
    detector_.reset();
  
  
  
    measure_tool_.reset();
  
    thread_pool_.reset();
  
    camera_.getImageQueue().shutdown();  
  
      if (thread_pool_) {
          thread_pool_->waitUntilEmpty(); 
      }
  
    WUST_INFO(vision_logger) << "WustVision shutdown complete.";
  }

void  WustVision::init()
{
    
    const std::string model_path = "/home/hy/wust_vision/model/opt-1208-001.onnx";
    AdaptedTRTModule::Params params;
    params.input_w = 416;
    params.input_h = 416;
    params.num_classes = 8;
    params.num_colors = 4;
    params.conf_threshold = 0.25;
    params.nms_threshold = 0.3;
    params.top_k = 128;

    const std::string camera_info_path = "/home/hy/wust_vision/config/camera_info.yaml";
    measure_tool_=std::make_unique<MonoMeasureTool>(camera_info_path);
    initTF();
    initTracker();

    detect_color_=0;

    if (model_path.empty()) {
        WUST_ERROR(vision_logger)<< "Model path is empty.";
        return;
    }

  // Create AdaptedTRTModule
  detector_ = std::make_unique<AdaptedTRTModule>(model_path, params);
  detector_->setCallback(std::bind(
    &WustVision::DetectCallback, this, std::placeholders::_1,
    std::placeholders::_2, std::placeholders::_3));
  
  thread_pool_ = std::make_unique<ThreadPool>(std::thread::hardware_concurrency(), 100);
  
  if (!camera_.initializeCamera()) {
    WUST_ERROR(vision_logger) << "Camera initialization failed." ;
    return ;
    }
    camera_.setParameters(165,4000,7.0,"Bits_8","BayerRG8");
    camera_.startCamera();
  is_inited_ = true;

}
void WustVision::initTF()
{
    // odom 是世界坐标系的根节点
    tf_tree_.setTransform("", "odom", createTf(0, 0, 0, tf2::Quaternion(0, 0, 0, 1)));

    // camera 相对于 odom，设置 odom -> camera 的变换
    tf_tree_.setTransform("odom", "camera", createTf(0, 0, 0, tf2::Quaternion(0, 0, 0, 1)));

    // camera_optical_frame 相对于 camera，设置 camera -> camera_optical_frame 的旋转变换
    double yaw = -M_PI / 2;
    double roll = -M_PI / 2;
    double pitch = 0.0;

    tf2::Quaternion orientation;
    orientation.setRPY(roll, pitch, yaw);
    tf_tree_.setTransform("camera", "camera_optical_frame", createTf(0, 0, 0, orientation));
}

void WustVision::initTracker()
{ target_frame_="odom";
  double max_match_distance = 0.2;
  double max_match_yaw_diff = 1.0;
  tracker_ = std::make_unique<Tracker>(max_match_distance, max_match_yaw_diff);
  tracker_->tracking_thres =  5;
  lost_time_thres_ = 0.3;
  // EKF
  // xa = x_armor, xc = x_robot_center
  // state: xc, v_xc, yc, v_yc, zc, v_zc, yaw, v_yaw, r, d_zc
  // measurement: p, y, d, yaw
  // f - Process function
  auto f = Predict(0.005);
  // h - Observation function
  auto h = Measure();
  // update_Q - process noise covariance matrix
  s2qx_ = 20.0;
  s2qy_ = 20.0;
  s2qz_ = 20.0;
  s2qyaw_ = 100.0;
  s2qr_ = 800.0;
  s2qd_zc_ = 800.0;

  auto u_q = [this]() {
    Eigen::Matrix<double, X_N, X_N> q;
    double t = dt_, x = s2qx_, y = s2qy_, z = s2qz_, yaw = s2qyaw_, r = s2qr_, d_zc=s2qd_zc_;
    double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx = pow(t, 2) * x;
    double q_y_y = pow(t, 4) / 4 * y, q_y_vy = pow(t, 3) / 2 * y, q_vy_vy = pow(t, 2) * y;
    double q_z_z = pow(t, 4) / 4 * x, q_z_vz = pow(t, 3) / 2 * x, q_vz_vz = pow(t, 2) * z;
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
  // update_R - measurement noise covariance matrix
  r_x_ = 0.05;
  r_y_ = 0.05;
  r_z_ = 0.05;
  r_yaw_ = 0.02;
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
  // P - error estimate covariance matrix
  Eigen::DiagonalMatrix<double, X_N> p0;
  p0.setIdentity();
  tracker_->ekf = std::make_unique<RobotStateEKF>(f, h, u_q, u_r, p0);
}
void WustVision::armorsCallback(const Armors& armors_msg) {
    // Lazy initialize solver owing to weak_from_this() can't be called in constructor

  
  
    

    
    
  
    // Init message
    
    Target target_msg;
    auto time = std::chrono::steady_clock::now();
    target_msg.timestamp = time;
    target_msg.frame_id = target_frame_;
  
  
    // Update tracker
    if (tracker_->tracker_state == Tracker::LOST) {
      tracker_->init(armors_msg);
      target_msg.tracking = false;
    } else {
      dt_ = std::chrono::duration<double>(time - last_time_).count();
      tracker_->lost_thres = std::abs(static_cast<int>(lost_time_thres_ / dt_));
      if (tracker_->tracked_id == ArmorNumber::OUTPOST) {
        tracker_->ekf->setPredictFunc(Predict{dt_, MotionModel::CONSTANT_ROTATION});
      } else {
        tracker_->ekf->setPredictFunc(Predict{dt_, MotionModel::CONSTANT_VEL_ROT});
      }
      tracker_->update(armors_msg);

      if (tracker_->tracker_state == Tracker::DETECTING) {
        target_msg.tracking = false;
      } else if (tracker_->tracker_state == Tracker::TRACKING ||
                 tracker_->tracker_state == Tracker::TEMP_LOST) {
        target_msg.tracking = true;
        // Fill target message
        const auto &state = tracker_->target_state;
        target_msg.id = tracker_->tracked_id;
        target_msg.armors_num = static_cast<int>(tracker_->tracked_armors_num);
        target_msg.position_.x = state(0);
        target_msg.velocity_.x = state(1);
        target_msg.position_.y = state(2);
        target_msg.velocity_.y = state(3);
        target_msg.position_.z = state(4);
        target_msg.velocity_.z = state(5);
        target_msg.yaw = state(6);
        target_msg.v_yaw = state(7);
        target_msg.radius_1 = state(8);
        target_msg.radius_2 = tracker_->another_r;
        target_msg.d_zc = state(9);
        target_msg.d_za = tracker_->d_za;
      }
    }
    //WUST_DEBUG(vision_logger)<<"x:"<<target_msg.position_.x<<"y:"<<target_msg.position_.y<<"z:"<<target_msg.position_.z;
   
  
    last_time_ = time;
  }
void WustVision::DetectCallback(
    const std::vector<ArmorObject>& objs, int64_t timestamp_nanosec, const cv::Mat& src_img)
{   std::lock_guard<std::mutex> lock(callback_mutex_);
    detect_finish_count_++;
    if(objs.size()>=6){
    WUST_WARN(vision_logger)<<"Detected "<<objs.size()<<" objects"<<"too much";
    infer_running_count_--;
    return;}
    if (measure_tool_ == nullptr) {
    WUST_WARN(vision_logger)<<"NO camera info";
    return;
  } 
    Armors armors;
    for (auto & obj : objs) {
    if (detect_color_ == 0 && obj.color != ArmorColor::RED) {
        continue;
    } else if (detect_color_ == 1 && obj.color != ArmorColor::BLUE) {
        continue;
    }

    cv::Point3f target_position;
    cv::Mat target_rvec;
    std::string armor_type;

    if (!measure_tool_->calcArmorTarget(obj, target_position, target_rvec, armor_type)) {
        //WUST_WARN(vision_logger) << "Calculate target position failed";
        continue;
    }

    
    if (!cv::checkRange(cv::Mat(target_position))) {
    //WUST_WARN(vision_logger) << "Invalid target position (NaN)";
    continue;
    }


   
    if (target_rvec.empty() || target_rvec.total() != 3 || target_rvec.rows * target_rvec.cols != 3 || !cv::checkRange(target_rvec)) {
        //WUST_WARN(vision_logger) << "Invalid rotation vector (empty or NaN): " << target_rvec;
        continue;
    }

    try {
        cv::Mat rot_mat;
        cv::Rodrigues(target_rvec, rot_mat);

        tf2::Matrix3x3 tf_rot_mat(
            rot_mat.at<double>(0, 0), rot_mat.at<double>(0, 1), rot_mat.at<double>(0, 2),
            rot_mat.at<double>(1, 0), rot_mat.at<double>(1, 1), rot_mat.at<double>(1, 2),
            rot_mat.at<double>(2, 0), rot_mat.at<double>(2, 1), rot_mat.at<double>(2, 2));
        tf2::Quaternion tf_quaternion;
        tf_rot_mat.getRotation(tf_quaternion);

        if (!std::isfinite(tf_quaternion.x) || !std::isfinite(tf_quaternion.y) ||
        !std::isfinite(tf_quaternion.z) || !std::isfinite(tf_quaternion.w)) {
        WUST_WARN(vision_logger) << "Quaternion contains NaN or Inf";
        continue;
        }


        //WUST_INFO(vision_logger) << "Position: " << target_position << " Quaternion: " << tf_quaternion;
        Armor armor;
        armor.pos={target_position.x,target_position.y,target_position.z};
        armor.ori={tf_quaternion.x,tf_quaternion.y,tf_quaternion.z,tf_quaternion.w};

        armor.number=obj.number;
        armor.type=armor_type;
        armor.distance_to_image_center=measure_tool_->calcDistanceToCenter(obj);
        armors.armors.emplace_back(armor);
        //WUST_INFO(vision_logger)<<"yaw"<<armor.rpy_.yaw;

    } catch (const cv::Exception& e) {
        WUST_ERROR(vision_logger) << "cv::Rodrigues failed: " << e.what();
        continue;
    }
}
    armors.timestamp=std::chrono::steady_clock::now();
    armors.frame_id="camera_optical_frame";
    for (auto& armor : armors.armors) {
      try {
          auto pose_intargetframe =tf_tree_.transform(armor.pos, armors.frame_id, target_frame_);
          armor.pos = pose_intargetframe.position;
          armor.ori = pose_intargetframe.orientation;
      } catch (const std::exception& e) {
          WUST_ERROR(vision_logger) << "Can't find transform from " << armors.frame_id << " to " << target_frame_ << ": " << e.what();
          return;
      }
  }
  
    


  
    

    infer_running_count_--;

    thread_pool_->enqueue([this, armors = std::move(armors)]() {
      this->armorsCallback(armors);
  });

    drawresult(src_img, objs,timestamp_nanosec);
    
    
}
void WustVision::processImage(const ImageFrame& frame) {
  
    

    img_recv_count_++;
        if (infer_running_count_.load() >= 4) {
       WUST_WARN(vision_logger)<<"Infer running too much ("<<infer_running_count_.load()<<"), dropping frame";
       return;    
        }


    cv::Mat img = convertToMat(frame);
    infer_running_count_++;
    printStats();
    auto timestamp_nanosec = std::chrono::duration_cast<std::chrono::nanoseconds>(
        frame.timestamp.time_since_epoch())
        .count();
    detector_->pushInput(img, timestamp_nanosec);
    
   

   
}
void WustVision::printStats()
{
  using namespace std::chrono;
  
  auto now = steady_clock::now();
  
  if (last_stat_time_steady_.time_since_epoch().count() == 0) {
    last_stat_time_steady_ = now;
    return;
  }

  auto elapsed = duration_cast<duration<double>>(now - last_stat_time_steady_);
  if (elapsed.count() >= 1.0) {
   WUST_INFO(vision_logger)<< "Received: " << img_recv_count_ << ", Detected: " << detect_finish_count_ << ", FPS: " << detect_finish_count_ / elapsed.count();

    img_recv_count_ = 0;
    detect_finish_count_ = 0;
    last_stat_time_steady_ = now;
  }
}
void WustVision::imageConsumer(ThreadSafeQueue<ImageFrame>& queue, ThreadPool& pool) {
    while (is_inited_) {
        ImageFrame frame;
  
  
        while (queue.size() > 1) {
            queue.try_pop(frame);
        }
  
   
        if (!queue.wait_and_pop(frame) && queue.is_shutdown()) {
            WUST_INFO(vision_logger) << "Queue is shutdown, exiting consumer thread.";
            break;
        }
  
       
        pool.enqueue([frame = std::move(frame), this]() {
            processImage(frame);
        });
    }
  }
  

WustVision* global_vision = nullptr;
void signalHandler(int signum) {
  WUST_INFO("main") << "Interrupt signal (" << signum << ") received.";
  if (global_vision) {
      global_vision->stop();  
  }
}
int main() 
{ 
    WustVision vision;
    global_vision = &vision;

    std::signal(SIGINT, signalHandler);
    std::thread consumer_thread([&vision]() {
        vision.imageConsumer(vision.camera_.getImageQueue(), *vision.thread_pool_);
    });

    consumer_thread.join();
    
}



