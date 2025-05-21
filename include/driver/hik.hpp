// driver/hik.hpp
#ifndef HIK_HPP
#define HIK_HPP

#include <iostream>
#include <opencv2/videoio.hpp>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include "type/image.hpp"         // ImageFrame, ThreadSafeQueue 定义
#include "MvCameraControl.h"      // 海康 SDK
#include "common/ThreadPool.h"    // 如果你还在用线程池

class HikCamera {
public:
    HikCamera();
    ~HikCamera();

    // 初始化：传入 video_path 则进入视频回放模式，否则打开物理相机
    bool initializeCamera(const std::string& video_path);

    // 仅硬件模式生效
    void setParameters(double acquisition_frame_rate,
                       double exposure_time,
                       double gain,
                       const std::string& adc_bit_depth,
                       const std::string& pixel_format);

    // 启动／停止采集
    void startCamera();
    void stopCamera();

    // 重启（硬件模式）
    bool restartCamera();

    // 外部获取图像队列
    ThreadSafeQueue<ImageFrame>& getImageQueue();

private:
    // ——— SDK 异步回调入口 ———
    static void __stdcall onImageCallback(
        unsigned char* pData,
        MV_FRAME_OUT_INFO* pFrameInfo,
        void* pUser
    );

    // 真正处理每帧的逻辑：时间戳、转换、入队、（可选监测重启）
    void handleFrame(unsigned char* pData, MV_FRAME_OUT_INFO* pFrameInfo);

    // （保留）视频回放循环
    void videoCaptureLoop();

    // 成员变量
    void*                       camera_handle_{ nullptr };   // MV_CC_DEVICE_HANDLE
    MV_IMAGE_BASIC_INFO         img_info_{};
    MV_CC_PIXEL_CONVERT_PARAM   convert_param_{};
    ThreadSafeQueue<ImageFrame> image_queue_;
    std::thread                 capture_thread_;

    // 日志、状态、参数回溯
    std::string                 hik_logger = "hik_camera";
    double                      last_frame_rate_{ 30.0 };
    double                      last_exposure_time_{ 0.0 };
    double                      last_gain_{ 0.0 };
    std::string                 last_adc_bit_depth_;
    std::string                 last_pixel_format_;
    bool                        in_low_frame_rate_state_{ false };
    std::chrono::steady_clock::time_point low_frame_rate_start_time_;

    // 控制变量
    std::atomic<bool>           stop_signal_{ false };
    bool                        is_video_mode_{ false };
    cv::VideoCapture            video_cap_;
    int                         video_fps_{ 30 };

};

#endif // HIK_HPP
