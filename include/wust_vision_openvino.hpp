#include "driver/hik.hpp"
#include "type/image.hpp"
#include "detect/openvino.hpp"
#include "type/type.hpp"
#include "yaml-cpp/yaml.h"
#include "common/gobal.hpp"
#include "driver/serial.hpp"
#include <opencv2/core/mat.hpp>
#include "control/armor_solver.hpp"
class WustVision {
public:
    WustVision();
    ~WustVision();
   
   
    void init();
   
    void processImage(const ImageFrame& frame);
    void imageConsumer(ThreadSafeQueue<ImageFrame>& queue, ThreadPool& pool);
   
    void printStats();
    void DetectCallback(
        const std::vector<ArmorObject>& objs, int64_t timestamp_nanosec, const cv::Mat& src_img);
    void stop();
    void armorsCallback(const Armors& armors_,const cv::Mat& src_img);
    void initTF();
    void initSerial();
    void initTracker(const YAML::Node& config);
    void timerCallback();
    void startTimer();
    void stopTimer();
    Armors visualizeTargetProjection(Target armor_target_);
   


    HikCamera camera_;
    ThreadSafeQueue<ImageFrame> image_queue_;
    std::thread image_thread_;
    std::unique_ptr<ThreadPool> thread_pool_;
    std::unique_ptr<OpenVino> detector_;
    bool is_inited_ = false;
    size_t img_recv_count_ = 0;
    size_t detect_finish_count_ = 0;
    std::chrono::steady_clock::time_point last_stat_time_steady_;
    std::atomic<int> infer_running_count_{0};
    int max_infer_running_ ; 
    std::mutex callback_mutex_;
    
    
    std::string vision_logger="openvino_vision";
    std::atomic<bool> run_loop_{false};
    bool debug_mode_ = false;
    bool show_armor_ = false;
    bool show_target_ = false;
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
    
    
    std::string target_frame_;
    std::chrono::steady_clock::time_point last_time_;
    double dt_;
    imgframe imgframe_;
    std::mutex img_mutex_;
    cv::VideoWriter video_writer_;  
    bool is_recording_;
    Serial serial_;
    std::unique_ptr<Solver> solver_;

};