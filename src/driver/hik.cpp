// driver/hik.cpp
#include "driver/hik.hpp"
#include "common/logger.hpp"
#include <stdexcept>
#include <chrono>

HikCamera::HikCamera() = default;

HikCamera::~HikCamera() {
    stopCamera();
    if (capture_thread_.joinable())
        capture_thread_.join();

    if (camera_handle_) {
        MV_CC_StopGrabbing(camera_handle_);
        MV_CC_CloseDevice(camera_handle_);
        MV_CC_DestroyHandle(&camera_handle_);
    }
    if (video_cap_.isOpened())
        video_cap_.release();

    WUST_INFO(hik_logger) << "Camera destroyed!";
}

bool HikCamera::initializeCamera(const std::string& video_path) {
    if (stop_signal_) return false;

    if (!video_path.empty()) {
        // —— 视频回放模式 —— //
        is_video_mode_ = true;
        video_cap_.open(video_path);
        if (!video_cap_.isOpened()) {
            WUST_ERROR(hik_logger) << "Failed to open video file: " << video_path;
            return false;
        }
        // 读取视频分辨率和帧率
        img_info_.nWidthValue  = static_cast<int>(video_cap_.get(cv::CAP_PROP_FRAME_WIDTH));
        img_info_.nHeightValue = static_cast<int>(video_cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
        video_fps_ = video_cap_.get(cv::CAP_PROP_FPS);
        if (video_fps_ <= 0) video_fps_ = 30.0;
        WUST_INFO(hik_logger) << "Video mode initialized: "
                              << img_info_.nWidthValue << "×" << img_info_.nHeightValue
                              << " @" << video_fps_ << "fps";
        return true;
    }

    // —— 硬件模式 —— //
    MV_CC_DEVICE_INFO_LIST dev_list;
    while (!stop_signal_) {
        int ret = MV_CC_EnumDevices(MV_USB_DEVICE, &dev_list);
        if (ret != MV_OK) {
            WUST_WARN(hik_logger) << "EnumDevices failed, retrying...";
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        if (dev_list.nDeviceNum == 0) {
            WUST_WARN(hik_logger) << "No camera found, retrying...";
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        break;
    }
    if (stop_signal_) return false;

    // 创建并打开设备
    int ret = MV_CC_CreateHandle(&camera_handle_, dev_list.pDeviceInfo[0]);
    if (ret != MV_OK) {
        WUST_ERROR(hik_logger) << "CreateHandle failed: " << ret;
        return false;
    }
    ret = MV_CC_OpenDevice(camera_handle_);
    if (ret != MV_OK) {
        WUST_ERROR(hik_logger) << "OpenDevice failed: " << ret;
        return false;
    }

    // 获取图像信息
    ret = MV_CC_GetImageInfo(camera_handle_, &img_info_);
    if (ret != MV_OK) {
        WUST_ERROR(hik_logger) << "GetImageInfo failed: " << ret;
        return false;
    }

    // 预设像素转换参数
    convert_param_.nWidth         = img_info_.nWidthValue;
    convert_param_.nHeight        = img_info_.nHeightValue;
    convert_param_.enDstPixelType = PixelType_Gvsp_RGB8_Packed;

    return true;
}

void HikCamera::setParameters(double frame_rate,
                              double exposure_time,
                              double gain,
                              const std::string& adc_bit_depth,
                              const std::string& pixel_format)
{
    if (is_video_mode_) {
        WUST_WARN(hik_logger) << "Video mode ignores parameter settings";
        return;
    }

    // 开启并设置采集帧率
    MV_CC_SetBoolValue(camera_handle_, "AcquisitionFrameRateEnable", true);
    MV_CC_SetFloatValue(camera_handle_, "AcquisitionFrameRate", frame_rate);
    WUST_INFO(hik_logger) << "AcquisitionFrameRate: " << frame_rate;

    // 设置曝光
    MV_CC_SetFloatValue(camera_handle_, "ExposureTime", exposure_time);
    WUST_INFO(hik_logger) << "ExposureTime: " << exposure_time;

    // 设置增益
    MV_CC_SetFloatValue(camera_handle_, "Gain", gain);
    WUST_INFO(hik_logger) << "Gain: " << gain;

    // 设置 ADC 位深
    int ret = MV_CC_SetEnumValueByString(camera_handle_, "ADCBitDepth", adc_bit_depth.c_str());
    if (ret == MV_OK)
        WUST_INFO(hik_logger) << "ADCBitDepth: " << adc_bit_depth;
    else
        WUST_ERROR(hik_logger) << "Set ADCBitDepth failed: " << ret;

    // 设置像素格式
    ret = MV_CC_SetEnumValueByString(camera_handle_, "PixelFormat", pixel_format.c_str());
    if (ret == MV_OK)
        WUST_INFO(hik_logger) << "PixelFormat: " << pixel_format;
    else
        WUST_ERROR(hik_logger) << "Set PixelFormat failed: " << ret;

    // 记录参数，重启时用
    last_frame_rate_     = static_cast<float>(frame_rate);
    last_exposure_time_  = static_cast<float>(exposure_time);
    last_gain_           = static_cast<float>(gain);
    last_adc_bit_depth_  = adc_bit_depth;
    last_pixel_format_   = pixel_format;
}

void HikCamera::startCamera() {
    stop_signal_ = false;

    if (is_video_mode_) {
        // 视频回放线程
        capture_thread_ = std::thread(&HikCamera::videoCaptureLoop, this);
        WUST_INFO(hik_logger) << "Video capture thread started";
        return;
    }

    // 硬件模式下注册回调并启动异步抓图
    int ret = MV_CC_RegisterImageCallBack(
        camera_handle_,
        &HikCamera::onImageCallback,
        this
    );
    if (ret != MV_OK) {
        WUST_ERROR(hik_logger) << "RegisterImageCallBackEx failed: " << ret;
        return;
    }

    ret = MV_CC_StartGrabbing(camera_handle_);
    if (ret != MV_OK) {
        WUST_ERROR(hik_logger) << "StartGrabbing failed: " << ret;
        return;
    }

    WUST_INFO(hik_logger) << "Asynchronous hardware capture started";
}

void HikCamera::stopCamera() {
    stop_signal_ = true;
    if (!is_video_mode_) {
        MV_CC_StopGrabbing(camera_handle_);
        // 可选：反注册回调
        MV_CC_RegisterImageCallBackEx(camera_handle_, nullptr, nullptr);
    }
}

void __stdcall HikCamera::onImageCallback(
    unsigned char* pData,
    MV_FRAME_OUT_INFO* pFrameInfo,
    void* pUser
) {
    auto self = reinterpret_cast<HikCamera*>(pUser);
    if (self && !self->stop_signal_) {
        try {
            self->handleFrame(pData, pFrameInfo);
        } catch (const std::exception& e) {
            WUST_ERROR("hik_camera") << "Exception in onImageCallback: " << e.what();
        } catch (...) {
            WUST_ERROR("hik_camera") << "Unknown exception in onImageCallback";
        }
    }
}

void HikCamera::handleFrame(unsigned char* pData, MV_FRAME_OUT_INFO* pFrameInfo) {
    // 1. 时间戳校正：到达时间减半曝光
    auto arrival = std::chrono::steady_clock::now();
    MVCC_FLOATVALUE exp_val{0};
    float exp_us = 0.f;
    if (MV_CC_GetFloatValue(camera_handle_, "ExposureTime", &exp_val) == MV_OK) {
        exp_us = exp_val.fCurValue;
    }
    auto timestamp = arrival - std::chrono::microseconds(int64_t(exp_us/2.f));

    // 2. 构建 ImageFrame
    ImageFrame frame;
    frame.width     = pFrameInfo->nWidth;
    frame.height    = pFrameInfo->nHeight;
    frame.step      = frame.width * 3;
    frame.timestamp = timestamp;
    frame.data.resize(frame.width * frame.height * 3);

    // 3. 像素格式转换
    convert_param_.pDstBuffer      = frame.data.data();
    convert_param_.nDstBufferSize  = int(frame.data.size());
    convert_param_.pSrcData        = pData;
    convert_param_.nSrcDataLen     = pFrameInfo->nFrameLen;
    convert_param_.enSrcPixelType  = pFrameInfo->enPixelType;
    MV_CC_ConvertPixelType(camera_handle_, &convert_param_);

    // 4. 推入线程安全队列
    image_queue_.push(std::move(frame));

    // 5. （可选）帧率监测、低帧率重启逻辑可在这里实现
}

void HikCamera::videoCaptureLoop() {
    WUST_INFO(hik_logger) << "Starting video capture loop!";
    cv::Mat mat;
    auto frame_interval = std::chrono::milliseconds(int(1000.0 / video_fps_));

    while (!stop_signal_) {
        auto t0 = std::chrono::steady_clock::now();

        if (!video_cap_.read(mat)) {
            // 播放结束，循环播放
            video_cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        // BGR -> RGB
        cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);

        ImageFrame f;
        f.timestamp = std::chrono::steady_clock::now();
        f.width     = mat.cols;
        f.height    = mat.rows;
        f.step      = mat.step;
        f.data.assign(mat.datastart, mat.dataend);
        image_queue_.push(std::move(f));

        auto dt = std::chrono::steady_clock::now() - t0;
        if (dt < frame_interval) {
            std::this_thread::sleep_for(frame_interval - dt);
        }
    }
}

bool HikCamera::restartCamera() {
    // 停止并销毁
    MV_CC_StopGrabbing(camera_handle_);
    MV_CC_CloseDevice(camera_handle_);
    MV_CC_DestroyHandle(&camera_handle_);
    camera_handle_ = nullptr;
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // 重新初始化
    if (!initializeCamera("") && !stop_signal_) {
        WUST_ERROR(hik_logger) << "Re-initialize camera failed!";
        return false;
    }
    // 重新设置参数
    setParameters(last_frame_rate_,
                  last_exposure_time_,
                  last_gain_,
                  last_adc_bit_depth_,
                  last_pixel_format_);

    // 重新启动抓图
    int ret = MV_CC_StartGrabbing(camera_handle_);
    if (ret != MV_OK) {
        WUST_ERROR(hik_logger) << "StartGrabbing after restart failed: " << ret;
        return false;
    }
    WUST_INFO(hik_logger) << "Camera restarted successfully!";
    return true;
}

ThreadSafeQueue<ImageFrame>& HikCamera::getImageQueue() {
    return image_queue_;
}
