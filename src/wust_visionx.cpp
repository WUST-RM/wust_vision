#include "wust_visionx.hpp"
#include "common/logger.hpp"

#include <opencv2/highgui.hpp>
#include <functional>
#include <string>
#include "common/tf.hpp"
#include <csignal>
WustVision::WustVision()
{
  detector_=nullptr;  
  init();
}
WustVision::~WustVision() {
  WUST_INFO("vision_main") << "Shutting down WustVision...";

  
  is_inited_ = false;


  camera_.stopCamera();


  
  detector_.reset();



  measure_tool_.reset();

  thread_pool_.reset();

  WUST_INFO("vision_main") << "WustVision shutdown complete.";
}
void  WustVision::init()
{ 
  auto classify_model_path = "/home/hy/wust_vision/model/mlp.onnx";
  auto classify_label_path = "/home/hy/wust_vision/model/label.txt";
  const std::string model_path =  "/home/hy/wust_vision/model/opt-1208-001.onnx";
  auto device_type = "GPU";
  float conf_threshold = 0.25;
  int top_k = 128;
  float nms_threshold = 0.3;

  float expand_ratio_w = 2.0;
  float expand_ratio_h = 1.5;
  int binary_thres = 85;

  LightParams l_params = {
    .min_ratio =  0.08,
    .max_ratio =  0.4,
    .max_angle =  40.0,
    };

  if (model_path.empty()) {
    WUST_ERROR("vision_main")<< "Model path is empty." ;
    return;
  }

  WUST_INFO("vision_main") <<"Model path: "<<model_path.c_str();
    
  
  const std::string camera_info_path = "/home/hy/wust_vision/config/camera_info.yaml";
  measure_tool_=std::make_unique<MonoMeasureTool>(camera_info_path);

  detect_color_=0;
  detector_ = std::make_unique<OpenVino>(
    model_path, classify_model_path, classify_label_path, device_type,l_params, conf_threshold, top_k,
    nms_threshold, expand_ratio_w, expand_ratio_h, binary_thres);
  detector_->detect_color_ = detect_color_;

  detector_->setCallback(std::bind(
      &WustVision::DetectCallback, this, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3));
  thread_pool_ = std::make_unique<ThreadPool>(std::thread::hardware_concurrency(), 100);
  if (!camera_.initializeCamera()) {
    WUST_ERROR("vision_main") << "Camera initialization failed." ;
    return ;
    }
    camera_.setParameters(165,3500,7.0,"Bits_8","BayerRG8");
    camera_.startCamera();
  is_inited_ = true;

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

  WUST_INFO("vision_main") << "WustVision shutdown complete.";
}
void WustVision::DetectCallback(
  const std::vector<ArmorObject>& objs, int64_t timestamp_nanosec, const cv::Mat& src_img)
{   
  std::lock_guard<std::mutex> lock(callback_mutex_);
  detect_finish_count_++;
  if(objs.size()>=6){
  WUST_WARN("vision_main")<<"Detected "<<objs.size()<<" objects"<<"too much";
  infer_running_count_--;
  return;}
  if (measure_tool_ == nullptr) {
  WUST_WARN("vision_main")<<"NO camera info";
  return;
} 
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
      //WUST_WARN("vision_main") << "Calculate target position failed";
      continue;
  }

  
  if (!cv::checkRange(cv::Mat(target_position))) {
  //WUST_WARN("vision_main") << "Invalid target position (NaN)";
  continue;
  }


 
  if (target_rvec.empty() || target_rvec.total() != 3 || target_rvec.rows * target_rvec.cols != 3 || !cv::checkRange(target_rvec)) {
      //WUST_WARN("vision_main") << "Invalid rotation vector (empty or NaN): " << target_rvec;
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
      WUST_WARN("vision_main") << "Quaternion contains NaN or Inf";
      continue;
      }


      //WUST_INFO("vision_main") << "Position: " << target_position << " Quaternion: " << tf_quaternion;

  } catch (const cv::Exception& e) {
      WUST_ERROR("vision_main") << "cv::Rodrigues failed: " << e.what();
      continue;
  }
}




  

  infer_running_count_--;

  drawresult(src_img, objs,timestamp_nanosec);
  
  
}
void WustVision::processImage(const ImageFrame& frame) {
  
    

  img_recv_count_++;
      if (infer_running_count_.load() >= 4) {
     WUST_WARN("vision_main")<<"Infer running too much ("<<infer_running_count_.load()<<"), dropping frame";
     return;    
      }

  // 图像转换与处理
  cv::Mat img = convertToMat(frame);
  infer_running_count_++;
  printStats();
  auto timestamp_nanosec = std::chrono::duration_cast<std::chrono::nanoseconds>(
      frame.timestamp.time_since_epoch())
      .count();
  
  detector_->pushInput(img, timestamp_nanosec);
  
 

 
}

void WustVision::imageConsumer(ThreadSafeQueue<ImageFrame>& queue, ThreadPool& pool) {
  while (is_inited_) {
      ImageFrame frame;


      while (queue.size() > 1) {
          queue.try_pop(frame);
      }

 
      if (!queue.wait_and_pop(frame) && queue.is_shutdown()) {
          WUST_INFO("vision_main") << "Queue is shutdown, exiting consumer thread.";
          break;
      }

     
      pool.enqueue([frame = std::move(frame), this]() {
          processImage(frame);
      });
  }
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
   WUST_INFO("test")<< "Received: " << img_recv_count_ << ", Detected: " << detect_finish_count_ << ", FPS: " << detect_finish_count_ / elapsed.count();

    img_recv_count_ = 0;
    detect_finish_count_ = 0;
    last_stat_time_steady_ = now;
  }
}


WustVision* global_vision = nullptr;
void signalHandler(int signum) {
  WUST_INFO("main") << "Interrupt signal (" << signum << ") received.";
  if (global_vision) {
      global_vision->stop();  
  }
}
int main() {
    WustVision vision;
    global_vision = &vision;

    std::signal(SIGINT, signalHandler);

    std::thread consumer_thread([&vision]() {
        vision.imageConsumer(vision.camera_.getImageQueue(), *vision.thread_pool_);
    });

    consumer_thread.join();
    return 0;
}
