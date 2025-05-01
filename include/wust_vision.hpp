#include "driver/hik.hpp"
#include "type/image.hpp"
#include "detect/trt.hpp"


class  WustVision{
public:
    WustVision();
    ~WustVision();
    void init();
    void imageConsumer(ThreadSafeQueue<ImageFrame>& queue, ThreadPool& pool);
    void processImage(const ImageFrame& frame);
    void printStats();
    void DetectCallback(
        const std::vector<ArmorObject> & objs, int64_t timestamp_nanosec, const cv::Mat & src_img);
    void stop();

    HikCamera camera_;
    ThreadSafeQueue<ImageFrame> image_queue_;
    std::unique_ptr<AdaptedTRTModule> detector_;
    std::unique_ptr<ThreadPool> thread_pool_;
    bool is_inited_ = false;
    size_t img_recv_count_ = 0;
    size_t detect_finish_count_ = 0;
    std::chrono::steady_clock::time_point last_stat_time_steady_;
    std::atomic<int> infer_running_count_{0};
    int max_infer_running_ ; 
    std::mutex callback_mutex_;
    std::unique_ptr<MonoMeasureTool> measure_tool_;
    int detect_color_;

};