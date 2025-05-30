#include "common/gobal.hpp"
#include "common/tf.hpp"
#include "control/armor_solver.hpp"
#include "detect/armor_pose_estimator.hpp"
#include "detect/trt.hpp"
#include "driver/image_capturer.hpp"
#include "driver/serial.hpp"
#include "tracker/tracker.hpp"
#include "type/type.hpp"
#include "yaml-cpp/yaml.h"
#include <opencv2/core/mat.hpp>
class WustVision {
public:
  WustVision();
  ~WustVision();
  void init();

  void processImage(const cv::Mat &frame, int64_t timestamp_nanosec);
  void captureLoop();
  void printStats();
  void DetectCallback(const std::vector<ArmorObject> &objs,
                      int64_t timestamp_nanosec, const cv::Mat &src_img);
  void stop();
  void armorsCallback(Armors armors_, const cv::Mat &src_img);
  void initTF();
  void initSerial();
  void initTracker(const YAML::Node &config);
  void timerCallback();
  void startTimer();
  void stopTimer();
  void transformArmorData(Armors &armors);
  Armors visualizeTargetProjection(Target armor_target_);

  std::unique_ptr<AdaptedTRTModule> detector_;
  std::unique_ptr<ThreadPool> thread_pool_;
  bool is_inited_ = false;
  size_t img_recv_count_ = 0;
  size_t detect_finish_count_ = 0;
  size_t fire_count_ = 0;
  std::chrono::steady_clock::time_point last_stat_time_steady_;
  std::atomic<int> infer_running_count_{0};
  int max_infer_running_;
  std::mutex callback_mutex_;

  int detect_color_;
  std::string vision_logger = "tensorrt_vision";
  std::atomic<bool> run_loop_{false};
  double latency_ms;

  std::atomic<bool> timer_running_{false};
  std::thread timer_thread_;
  std::unique_ptr<Tracker> tracker_;
  Target armor_target;
  std::mutex armor_target_mutex_;
  Armors armors_gobal;
  std::mutex armors_gobal_mutex_;
  double s2qx_, s2qy_, s2qz_, s2qyaw_, s2qr_, s2qd_zc_;
  double r_x_, r_y_, r_z_, r_yaw_;
  double lost_time_thres_;
  double gimbal2camera_x_, gimbal2camera_y_, gimbal2camera_z_,
      gimbal2camera_yaw_, gimbal2camera_roll_, gimbal2camera_pitch_;
  std::string target_frame_;
  std::chrono::steady_clock::time_point last_time_;
  double dt_;
  double debug_show_dt_;
  imgframe imgframe_;
  std::mutex img_mutex_;
  Serial serial_;
  std::unique_ptr<Solver> solver_;
  bool use_calculation_ = false;

  std::unique_ptr<ArmorPoseEstimator> armor_pose_estimator_;
  Eigen::Matrix3d imu_to_camera_;

  std::unique_ptr<hikcamera::ImageCapturer> capturer_;
  std::unique_ptr<std::thread> capture_thread_;
  std::atomic<bool> capture_running_;
};