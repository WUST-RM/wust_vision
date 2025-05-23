#include "driver/hik.hpp"
#include "common/logger.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <stdexcept>
#include <chrono>

HikCamera::HikCamera() : camera_handle_(nullptr), fail_count_(0), is_video_mode_(false) { }

HikCamera::~HikCamera() {
    stopCamera();
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    if (camera_handle_) {
        MV_CC_StopGrabbing(camera_handle_);
        MV_CC_CloseDevice(camera_handle_);
        MV_CC_DestroyHandle(&camera_handle_);
    }
    if (video_cap_.isOpened()) {
        video_cap_.release();
    }
    WUST_INFO(hik_logger) << "Camera destroyed!";
}

// 初始化相机：枚举设备、创建句柄、打开设备、获取图像信息等
bool HikCamera::initializeCamera(const std::string& video_path) {
    if (!video_path.empty()) {
        // 视频模式初始化
        is_video_mode_ = true;
        video_cap_.open(video_path);
        if (!video_cap_.isOpened()) {
            WUST_ERROR(hik_logger) << "Failed to open video file: " << video_path;
            return false;
        }
         // 获取视频基本信息
         img_info_.nWidthValue = static_cast<int>(video_cap_.get(cv::CAP_PROP_FRAME_WIDTH));
         img_info_.nHeightValue = static_cast<int>(video_cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
         video_fps_ = video_cap_.get(cv::CAP_PROP_FPS);
         if (video_fps_ <= 0) video_fps_ = 30;
         
         WUST_INFO(hik_logger) << "Video mode initialized: "
                             << img_info_.nWidthValue << "x" << img_info_.nHeightValue
                             << " @" << video_fps_ << "fps";
         return true;
    }
    MV_CC_DEVICE_INFO_LIST device_list;
    while (true) {
        int n_ret = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);
        if (n_ret != MV_OK) {
            WUST_WARN(hik_logger) << "Failed to enumerate devices, retrying...";
            std::this_thread::sleep_for(std::chrono::seconds(1));
        } else if (device_list.nDeviceNum == 0) {
            WUST_WARN(hik_logger) << "No camera found, retrying...";
            std::this_thread::sleep_for(std::chrono::seconds(1));
        } else {
            WUST_INFO(hik_logger) << "Found " << device_list.nDeviceNum << " cameras!";
            break;
        }
    }

    int n_ret = MV_CC_CreateHandle(&camera_handle_, device_list.pDeviceInfo[0]);
    if (n_ret != MV_OK) {
        WUST_ERROR(hik_logger) << "Failed to create camera handle!";
        return false;
    }

    n_ret = MV_CC_OpenDevice(camera_handle_);
    if (n_ret != MV_OK) {
        WUST_ERROR(hik_logger) << "Failed to open camera device!";
        return false;
    }

    n_ret = MV_CC_GetImageInfo(camera_handle_, &img_info_);
    if (n_ret != MV_OK) {
        WUST_ERROR(hik_logger) << "Failed to get camera image info!";
        return false;
    }

    // 为转换后的图像数据预留内存
    convert_param_.nWidth = img_info_.nWidthValue;
    convert_param_.nHeight = img_info_.nHeightValue;
    convert_param_.enDstPixelType = PixelType_Gvsp_RGB8_Packed;
  

    return true;
}

// 设置相机参数：帧率、曝光、增益、ADC位深及像素格式（这里硬编码参数，可按需修改）
void HikCamera::setParameters(double acquisition_frame_rate,
    double exposure_time,
    double gain,
    const std::string& adc_bit_depth,
    const std::string& pixel_format)
{   
    if (is_video_mode_) {
        WUST_WARN(hik_logger) << "Video mode ignores parameter settings";
        return;
    }
    MVCC_FLOATVALUE f_value;

    // 设置采集帧率
    MV_CC_SetBoolValue(camera_handle_, "AcquisitionFrameRateEnable", true);
    MV_CC_SetFloatValue(camera_handle_, "AcquisitionFrameRate", acquisition_frame_rate);
    WUST_INFO(hik_logger) << "Acquisition frame rate: " << acquisition_frame_rate;

    // 设置曝光时间（单位：微秒）
    MV_CC_SetFloatValue(camera_handle_, "ExposureTime", exposure_time);
    WUST_INFO(hik_logger) << "Exposure time: " << exposure_time;

    // 设置增益
    MV_CC_GetFloatValue(camera_handle_, "Gain", &f_value);
    MV_CC_SetFloatValue(camera_handle_, "Gain", gain);
    WUST_INFO(hik_logger) << "Gain: " << gain;

    // 设置 ADC 位深
    int status = MV_CC_SetEnumValueByString(camera_handle_, "ADCBitDepth", adc_bit_depth.c_str());
    if (status == MV_OK) {
        WUST_INFO(hik_logger) << "ADC Bit Depth set to " << adc_bit_depth;
    } else {
        WUST_ERROR(hik_logger) << "Failed to set ADC Bit Depth, status = " << status;
    }

    // 设置像素格式
    status = MV_CC_SetEnumValueByString(camera_handle_, "PixelFormat", pixel_format.c_str());
    if (status == MV_OK) {
        WUST_INFO(hik_logger) << "Pixel Format set to " << pixel_format;
    } else {
        WUST_ERROR(hik_logger) << "Failed to set Pixel Format, status = " << status;
    }

    last_frame_rate_ = acquisition_frame_rate;
    last_exposure_time_ = exposure_time;
    last_gain_ = gain;
    last_adc_bit_depth_ = adc_bit_depth;
    last_pixel_format_ = pixel_format;
}


// 启动图像采集，采集线程不断获取图像帧并推入队列
void HikCamera::startCamera() {
    if (is_video_mode_) {
        WUST_INFO(hik_logger) << "Starting video capture loop";
        capture_thread_ = std::thread(&HikCamera::videoCaptureLoop, this);
    } else {
        int n_ret = MV_CC_StartGrabbing(camera_handle_);
        if (n_ret != MV_OK) {
            WUST_ERROR(hik_logger) << "Failed to start camera grabbing!";
        }
        MVCC_INTVALUE stParam = {0};
        if (MV_CC_GetIntValue(camera_handle_, "Width", &stParam) == MV_OK) {
            expected_width_ = stParam.nCurValue;
        }
        if (MV_CC_GetIntValue(camera_handle_, "Height", &stParam) == MV_OK) {
            expected_height_ = stParam.nCurValue;
        }
        capture_thread_ = std::thread(&HikCamera::hikCaptureLoop, this);
    }
}
void HikCamera::videoCaptureLoop() {
    WUST_INFO(hik_logger) << "Starting video capture loop!";
    cv::Mat frame;
    auto frame_interval = std::chrono::milliseconds(
        static_cast<int>(1000 / (video_fps_ > 0 ? video_fps_ : 30))
    );

    while (!stop_signal_) {
        auto start_time = std::chrono::steady_clock::now();
        
        if (!video_cap_.read(frame)) {
            // 循环播放
            video_cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        // 转换OpenCV Mat到ImageFrame
        ImageFrame img_frame;
        img_frame.timestamp = std::chrono::steady_clock::now();
        img_frame.width = frame.cols;
        img_frame.height = frame.rows;
        img_frame.step = frame.step;

        // 转换BGR到RGB格式
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        img_frame.data.assign(frame.data, frame.data + frame.total() * frame.elemSize());

        if (on_frame_callback_) {
       // on_frame_callback_(frame);
        }

        // 控制帧率
        auto process_time = std::chrono::steady_clock::now() - start_time;
        if (process_time < frame_interval) {
            std::this_thread::sleep_for(frame_interval - process_time);
        }
    }
}

bool HikCamera::restartCamera() {
    if (is_video_mode_) {
        WUST_INFO(hik_logger) << "Restarting video playback...";
        video_cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
        return true;
    }
    WUST_WARN(hik_logger) << "Restarting camera from scratch...";

    MV_CC_StopGrabbing(camera_handle_);
    MV_CC_CloseDevice(camera_handle_);
    MV_CC_DestroyHandle(&camera_handle_);
    camera_handle_ = nullptr;

    std::this_thread::sleep_for(std::chrono::seconds(1));

    if (!initializeCamera("")) {
        WUST_ERROR(hik_logger) << "Failed to re-initialize camera.";
        return false;
    }

    setParameters(last_frame_rate_, last_exposure_time_, last_gain_, last_adc_bit_depth_, last_pixel_format_);

    int n_ret = MV_CC_StartGrabbing(camera_handle_);
    if (n_ret != MV_OK) {
        WUST_ERROR(hik_logger) << "Failed to start grabbing after restart.";
        return false;
    }

    WUST_INFO(hik_logger) << "Camera restarted successfully!";
    return true;
}



void HikCamera::hikCaptureLoop() {
    MV_FRAME_OUT out_frame;
    WUST_INFO(hik_logger) << "Starting image capture loop!";

    auto fail_start_time = std::chrono::steady_clock::now();
    bool in_fail_state = false;

    in_low_frame_rate_state_ = false;
    auto last_frame_rate_check = std::chrono::steady_clock::now();
    int frame_counter = 0;

    try {
        while (!stop_signal_) {
            int n_ret = MV_CC_GetImageBuffer(camera_handle_, &out_frame, 1);
            if (n_ret == MV_OK) {
                in_fail_state = false;
                ++frame_counter;
                

                ImageFrame frame;
                frame.width = out_frame.stFrameInfo.nWidth;
                frame.height = out_frame.stFrameInfo.nHeight;
                frame.step = frame.width * 3;
               
                frame.data.resize(frame.width * frame.height * 3);

                convert_param_.pDstBuffer = frame.data.data();
                convert_param_.nDstBufferSize = static_cast<int>(frame.data.size());
                convert_param_.pSrcData = out_frame.pBufAddr;
                convert_param_.nSrcDataLen = out_frame.stFrameInfo.nFrameLen;
                convert_param_.enSrcPixelType = out_frame.stFrameInfo.enPixelType;

                MV_CC_ConvertPixelType(camera_handle_, &convert_param_);
                auto current_time = std::chrono::steady_clock::now();
                frame.timestamp = current_time;
         
                if (on_frame_callback_) {
                    on_frame_callback_(frame);
                }

                MV_CC_FreeImageBuffer(camera_handle_, &out_frame);
                continue;
                if (std::chrono::duration_cast<std::chrono::seconds>(
                        current_time - last_frame_rate_check).count() >= 1) {
                    float actual_fps = static_cast<float>(frame_counter);
                    frame_counter = 0;
                    last_frame_rate_check = current_time;

                    if (actual_fps < last_frame_rate_ * 0.5f) {
                        if (!in_low_frame_rate_state_) {
                            low_frame_rate_start_time_ = current_time;
                            in_low_frame_rate_state_ = true;
                            WUST_WARN(hik_logger) << "Low FPS detected: " << actual_fps;
                        } else if (std::chrono::duration_cast<std::chrono::seconds>(
                                       current_time - low_frame_rate_start_time_).count() >= 5) {
                            WUST_ERROR(hik_logger) << "Low FPS persisted for 5s. Restarting camera...";
                            if (restartCamera()) {
                                in_low_frame_rate_state_ = false;
                                WUST_INFO(hik_logger) << "Camera restarted successfully";
                            } else {
                                WUST_ERROR(hik_logger) << "Restart failed, exiting capture loop.";
                                break;
                            }
                        }
                    } else if (in_low_frame_rate_state_) {
                        in_low_frame_rate_state_ = false;
                        WUST_INFO(hik_logger) << "FPS recovered to normal: " << actual_fps;
                    }
                }

            } else {
                if (!in_fail_state) {
                    fail_start_time = std::chrono::steady_clock::now();
                    in_fail_state = true;
                }

                if (std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::steady_clock::now() - fail_start_time).count() > 5) {
                    if (!restartCamera()) {
                        WUST_ERROR(hik_logger) << "Failed to restart camera after hardware failure.";
                        break;
                    }
                    fail_start_time = std::chrono::steady_clock::now();
                    in_fail_state = false;
                }
            }
        }
    } catch (const std::exception& e) {
        WUST_ERROR(hik_logger) << "Exception in capture loop: " << e.what();
        stop_signal_ = true;
    } catch (...) {
        WUST_ERROR(hik_logger) << "Unknown exception in capture loop!";
        stop_signal_ = true;
    }
    WUST_INFO(hik_logger) << "Exiting image capture loop.";
}



void HikCamera::stopCamera() {
    stop_signal_ = true;
    
}