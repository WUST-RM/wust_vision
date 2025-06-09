#include "driver/image_capturer.hpp"

#include <cstring>

#include <chrono>
#include <ratio>
#include <stdexcept>
#include <thread>
#include <tuple>

#include "common/logger.hpp"
#include <MvCameraControl.h>
#include <unistd.h>
namespace hikcamera {

class ImageCapturer::Impl {
public:
  Impl(const CameraProfile &profile, char *user_defined_name,
       const SyncMode &sync_mode)
      : user_defined_name_(user_defined_name), profile_(profile),
        sync_mode_(sync_mode) {

    const int max_retries = 30;
    int retry_count = 0;
    bool success = false;

    while (retry_count < max_retries) {
      auto device_info = search_camera(user_defined_name_);
      if (device_info && init_camera(*device_info, profile_, sync_mode_)) {
        success = true;
        break;
      }

      ++retry_count;
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    if (!success) {
      throw std::runtime_error{
          "Failed to init camera after 10 retries, see log for details."};
    }

    try {
      using namespace std::chrono_literals;
      read(1000ms);
    } catch (...) {
      this->~Impl();
      throw;
    }
  }

  ~Impl() {
    uninit_camera();
    delete[] converted_data_buffer_;
  }

  std::tuple<int, int> get_image_size() const {
    return {convert_parameter_.nWidth, convert_parameter_.nHeight};
  }

  cv::Mat read(std::chrono::duration<unsigned int, std::milli> timeout) {
    MV_FRAME_OUT stImageInfo;

    unsigned int ret =
        MV_CC_GetImageBuffer(camera_handle_, &stImageInfo, timeout.count());
    if (ret != MV_OK) {
      WUST_WARN(hik_logger) << "Image getting timeout. nRet [" << ret << "]"
                            << "try to restart camera";

      auto device_info = search_camera(user_defined_name_);
      init_camera(*device_info, profile_, sync_mode_);
      sleep(1);
      return cv::Mat();
    }
    // Only consider the situation where the size and format of each frame of
    // image passed in by the camera remain unchanged.
    if (converted_data_buffer_ == nullptr) {
      if (!is_rgb_pixel_type(stImageInfo.stFrameInfo.enPixelType))
        throw std::runtime_error{"RGB camera needed!"};

      converted_data_size_ =
          stImageInfo.stFrameInfo.nWidth * stImageInfo.stFrameInfo.nHeight * 3;
      converted_data_buffer_ = new unsigned char[converted_data_size_];

      convert_parameter_.nWidth = stImageInfo.stFrameInfo.nWidth; // image width
      convert_parameter_.nHeight =
          stImageInfo.stFrameInfo.nHeight; // image height
      convert_parameter_.nSrcDataLen =
          stImageInfo.stFrameInfo.nFrameLen; // input data size
      convert_parameter_.enSrcPixelType =
          stImageInfo.stFrameInfo.enPixelType; // input pixel format
      convert_parameter_.enDstPixelType =
          PixelType_Gvsp_RGB8_Packed; // 使用 RGB 格式代替 BGR
      // convert_parameter_.enDstPixelType = PixelType_Gvsp_BGR8_Packed; //
      // output pixel format
      convert_parameter_.pDstBuffer =
          converted_data_buffer_; // output data buffer
      convert_parameter_.nDstBufferSize =
          converted_data_size_; // output buffer size
    }

    convert_parameter_.pSrcData = stImageInfo.pBufAddr; // input data buffer

    ret = MV_CC_ConvertPixelType(camera_handle_, &convert_parameter_);

    if (ret != MV_OK) {
      WUST_ERROR(hik_logger)
          << "Failed to convert pixel type. nRet [" << ret << "]";
      // RCLCPP_ERROR(logger_, "Failed to convert pixel type. nRet [%u]", ret);
      // throw std::runtime_error{"Failed to convert pixel type"};
    }

    // Using this constructor, cv::Mat will not automatically release the
    // buffer, which needs to be released manually.
    cv::Mat img{stImageInfo.stFrameInfo.nHeight, stImageInfo.stFrameInfo.nWidth,
                CV_8UC3, convert_parameter_.pDstBuffer};

    // Double buffer swapness
    MV_CC_FreeImageBuffer(camera_handle_, &stImageInfo);

    return img;
  }
  bool software_trigger_on() {
    auto ret = MV_CC_SetCommandValue(camera_handle_, "TriggerSoftware");
    if (MV_OK != ret) {
      WUST_ERROR(hik_logger)
          << "Failed To send software trigger nRet [" << ret << "]";
      // RCLCPP_ERROR(logger_, "Failed To send software trigger nRet [%u]",
      // ret);
      return false;
    }
    return true;
  }
  bool set_frame_rate_inner_trigger_mode(float frame_rate) {

    auto ret =
        MV_CC_SetBoolValue(camera_handle_, "AcquisitionFrameRateEnable", true);
    if (MV_OK != ret) {
      WUST_ERROR(hik_logger)
          << "Failed To set frame rate control enable nRet [" << ret << "]";
      // RCLCPP_ERROR(logger_, "Failed To set frame rate control enable nRet
      // [%u]", ret);
      return false;
    }
    ret =
        MV_CC_SetFloatValue(camera_handle_, "AcquisitionFrameRate", frame_rate);
    if (MV_OK != ret) {
      WUST_ERROR(hik_logger) << "Failed To set frame rate nRet [" << ret << "]";
      // RCLCPP_ERROR(logger_, "Failed To set frame rate nRet [%u]", ret);
      return false;
    }
    return true;
  }

private:
#define SDK_RET_ASSERT(ret, message)                                           \
  do {                                                                         \
    if ((ret) != MV_OK) {                                                      \
      WUST_ERROR(hik_logger) << message << " nRet [" << ret << "]";            \
      return false;                                                            \
    }                                                                          \
  } while (false)

  bool is_same_device_name(MV_CC_DEVICE_INFO *pstMVDevInfo,
                           const char *targetName) {
    if (nullptr == pstMVDevInfo) {
      WUST_ERROR(hik_logger) << "The Pointer of pstMVDevInfo is NULL!";
      // RCLCPP_ERROR(logger_, "The Pointer of pstMVDevInfo is NULL!");
      return false;
    }
    const unsigned char *deviceName;
    if (pstMVDevInfo->nTLayerType == MV_GIGE_DEVICE) {
      deviceName = pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName;
    } else if (pstMVDevInfo->nTLayerType == MV_USB_DEVICE) {
      deviceName = pstMVDevInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName;
    } else {
      return false;
    }

    return strcmp(reinterpret_cast<const char *>(deviceName), targetName) == 0;
  }

  bool print_device_info(MV_CC_DEVICE_INFO *pstMVDevInfo) {
    if (nullptr == pstMVDevInfo) {
      WUST_ERROR(hik_logger) << "The Pointer of pstMVDevInfo is NULL!";
      // RCLCPP_ERROR(logger_, "The Pointer of pstMVDevInfo is NULL!");
      return false;
    }
    if (pstMVDevInfo->nTLayerType == MV_GIGE_DEVICE) {
      unsigned nIp1 =
          ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >>
           24);
      unsigned nIp2 =
          ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >>
           16);
      unsigned nIp3 =
          ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8);
      unsigned nIp4 =
          (pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff);
      WUST_INFO(hik_logger)
          << "DeviceIp:" << nIp1 << "." << nIp2 << "." << nIp3 << "." << nIp4;
      // RCLCPP_INFO(logger_, "DeviceIp: %d.%d.%d.%d", nIp1, nIp2, nIp3, nIp4);
      WUST_INFO(hik_logger)
          << "UserDefinedName:"
          << pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName;
      // RCLCPP_INFO(
      //     logger_, "UserDefinedName: %s",
      //     pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName);
    } else if (pstMVDevInfo->nTLayerType == MV_USB_DEVICE) {
      WUST_INFO(hik_logger)
          << "UserDefinedName:"
          << pstMVDevInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName;
      // RCLCPP_INFO(
      //     logger_, "UserDefinedName: %s",
      //     pstMVDevInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName);
      WUST_INFO(hik_logger)
          << "Serial Number:"
          << pstMVDevInfo->SpecialInfo.stUsb3VInfo.chSerialNumber;
      // RCLCPP_INFO(
      //     logger_, "Serial Number: %s",
      //     pstMVDevInfo->SpecialInfo.stUsb3VInfo.chSerialNumber);
      WUST_INFO(hik_logger)
          << "Device Number:"
          << pstMVDevInfo->SpecialInfo.stUsb3VInfo.nDeviceNumber;
      // RCLCPP_INFO(
      //     logger_, "Device Number: %u",
      //     pstMVDevInfo->SpecialInfo.stUsb3VInfo.nDeviceNumber);
    } else {
      WUST_ERROR(hik_logger) << "Neither a GigE camera nor a USB camera.";
      // RCLCPP_ERROR(logger_, "Neither a GigE camera nor a USB camera.");
      return false;
    }

    return true;
  }

  MV_CC_DEVICE_INFO *search_camera(const char *user_defined_name) {
    MV_CC_DEVICE_INFO_LIST device_list;

    memset(&device_list, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
    unsigned int ret =
        MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &device_list);
    if (ret != MV_OK) {
      WUST_ERROR(hik_logger) << "Failed to enum Devices. nRet [%u]";
      // RCLCPP_ERROR(logger_, "Failed to enum Devices. nRet [%u]", ret);
      return nullptr;
    }
    if (device_list.nDeviceNum == 0) {
      WUST_ERROR(hik_logger) << "Find No Devices.";
      // RCLCPP_ERROR(logger_, "Find No Devices.");
      return nullptr;
    }

    if (user_defined_name == nullptr) {
      if (device_list.nDeviceNum > 1) {
        WUST_ERROR(hik_logger)
            << device_list.nDeviceNum
            << "devices was found, please pass in the device name";
        // RCLCPP_ERROR(
        //     logger_, "Must pass in the device name because %u devices were
        //     found.", device_list.nDeviceNum);
        return nullptr;
      }
      return device_list.pDeviceInfo[0];
    } else {
      for (unsigned int i = 0; i < device_list.nDeviceNum; i++) {
        if (is_same_device_name(device_list.pDeviceInfo[i], user_defined_name))
          return device_list.pDeviceInfo[i];
      }
      WUST_ERROR(hik_logger)
          << device_list.nDeviceNum
          << "devices was found, but no device matches the name passed in:"
          << user_defined_name;
      // RCLCPP_ERROR(
      //     logger_, "%u devices was found, but no device matches the name
      //     passed in: %s", device_list.nDeviceNum, user_defined_name);
      return nullptr;
    }
  }

  bool init_camera(MV_CC_DEVICE_INFO &device_info, const CameraProfile &profile,
                   const SyncMode &sync_mode) {

    profile_ = profile;
    sync_mode_ = sync_mode;
    auto pDeviceInfo = &device_info;

    unsigned int ret;

    ret = MV_CC_CreateHandleWithoutLog(&camera_handle_, pDeviceInfo);
    SDK_RET_ASSERT(ret, "Failed to create handle.");
    FinalAction destroy_handle{
        [this]() { MV_CC_DestroyHandle(camera_handle_); }};

    ret = MV_CC_OpenDevice(camera_handle_);
    SDK_RET_ASSERT(ret, "Failed to open device.");
    FinalAction close_device{[this]() { MV_CC_CloseDevice(camera_handle_); }};

    if (pDeviceInfo->nTLayerType == MV_GIGE_DEVICE) {
      int nPacketSize = MV_CC_GetOptimalPacketSize(camera_handle_);
      if (nPacketSize <= 0) {
        WUST_ERROR(hik_logger) << "Get invalid packet Size: " << nPacketSize;
        // RCLCPP_ERROR(logger_, "Get invalid packet Size: %d", nPacketSize);
        return false;
      }

      ret = MV_CC_SetIntValue(camera_handle_, "GevSCPSPacketSize", nPacketSize);
      SDK_RET_ASSERT(ret, "Failed to set packet Size.");
    }

    ret = MV_CC_SetEnumValue(camera_handle_, "TriggerMode",
                             profile.trigger_mode ? MV_TRIGGER_MODE_ON
                                                  : MV_TRIGGER_MODE_OFF);
    SDK_RET_ASSERT(ret, "Failed to set trigger Mode.");

    ret = MV_CC_SetBoolValue(camera_handle_, "ReverseX", profile.invert_image);
    SDK_RET_ASSERT(ret, "Failed to set reverse x.");
    ret = MV_CC_SetBoolValue(camera_handle_, "ReverseY", profile.invert_image);
    SDK_RET_ASSERT(ret, "Failed to set reverse y.");

    ret = MV_CC_SetEnumValue(camera_handle_, "ExposureAuto",
                             MV_EXPOSURE_AUTO_MODE_OFF);
    SDK_RET_ASSERT(ret, "Failed to set auto exposure.");

    ret = MV_CC_SetFloatValue(camera_handle_, "ExposureTime",
                              profile.exposure_time.count());
    SDK_RET_ASSERT(ret, "Failed to set exposure time.");

    ret = MV_CC_SetFloatValue(camera_handle_, "Gain", profile.gain);
    SDK_RET_ASSERT(ret, "Failed to set gain.");

    ret =
        MV_CC_SetBoolValue(camera_handle_, "AcquisitionFrameRateEnable", false);
    SDK_RET_ASSERT(ret, "Failed to set acquisition frame rate enable.");

    ret = MV_CC_SetBayerCvtQuality(camera_handle_, 2);
    SDK_RET_ASSERT(ret, "Failed to set bayer cvt quality.");

    ret = MV_CC_StartGrabbing(camera_handle_);
    SDK_RET_ASSERT(ret, "Failed to start grabbing.");

    if (sync_mode == SyncMode::SOFTWARE) {
      ret =
          MV_CC_SetEnumValue(camera_handle_, "TriggerMode", MV_TRIGGER_MODE_ON);
      SDK_RET_ASSERT(ret, "Failed to start TriggerMode. : soft trigger");

      ret = MV_CC_SetEnumValue(camera_handle_, "TriggerSource",
                               MV_TRIGGER_SOURCE_SOFTWARE);
      SDK_RET_ASSERT(ret, "Failed to set trigger source. : soft trigger");
    }

    destroy_handle.disable();
    close_device.disable();
    return true;
  }

  static bool is_rgb_pixel_type(MvGvspPixelType enType) {
    switch (enType) {
    case PixelType_Gvsp_BGR8_Packed:
    case PixelType_Gvsp_YUV422_Packed:
    case PixelType_Gvsp_YUV422_YUYV_Packed:
    case PixelType_Gvsp_BayerGR8:
    case PixelType_Gvsp_BayerRG8:
    case PixelType_Gvsp_BayerGB8:
    case PixelType_Gvsp_BayerBG8:
    case PixelType_Gvsp_BayerGB10:
    case PixelType_Gvsp_BayerGB10_Packed:
    case PixelType_Gvsp_BayerBG10:
    case PixelType_Gvsp_BayerBG10_Packed:
    case PixelType_Gvsp_BayerRG10:
    case PixelType_Gvsp_BayerRG10_Packed:
    case PixelType_Gvsp_BayerGR10:
    case PixelType_Gvsp_BayerGR10_Packed:
    case PixelType_Gvsp_BayerGB12:
    case PixelType_Gvsp_BayerGB12_Packed:
    case PixelType_Gvsp_BayerBG12:
    case PixelType_Gvsp_BayerBG12_Packed:
    case PixelType_Gvsp_BayerRG12:
    case PixelType_Gvsp_BayerRG12_Packed:
    case PixelType_Gvsp_BayerGR12:
    case PixelType_Gvsp_BayerGR12_Packed:
      return true;
    default:
      return false;
    }
  }

  void uninit_camera() {
    unsigned int nRet;
    nRet = MV_CC_StopGrabbing(camera_handle_);
    if (MV_OK != nRet)
      WUST_ERROR(hik_logger)
          << "Failed to stop grabbing. nRet [" << nRet << "]";
    // RCLCPP_ERROR(logger_, "Failed to stop grabbing. nRet [%u]", nRet);

    nRet = MV_CC_CloseDevice(camera_handle_);
    if (MV_OK != nRet)
      WUST_ERROR(hik_logger) << "Failed to close device. nRet [" << nRet << "]";
    // RCLCPP_ERROR(logger_, "Failed to close device. nRet [%u]", nRet);

    nRet = MV_CC_DestroyHandle(camera_handle_);
    if (MV_OK != nRet)
      WUST_ERROR(hik_logger)
          << "Failed to destroy handle. nRet [" << nRet << "]";
    // RCLCPP_ERROR(logger_, "Failed to destroy handle. nRet [%u]", nRet);
  }

  template <typename Func> struct FinalAction {
    explicit FinalAction(Func func) : clean_{func}, enabled_(true) {}

    FinalAction(const FinalAction &) = delete;
    FinalAction &operator=(const FinalAction &) = delete;
    FinalAction(FinalAction &&) = delete;
    FinalAction &operator=(FinalAction &&) = delete;

    ~FinalAction() {
      if (enabled_)
        clean_();
    }

    void disable() { enabled_ = false; };

  private:
    Func clean_;
    bool enabled_;
  };

  void *camera_handle_ = nullptr;

  MV_CC_DEVICE_INFO *device_info_ = nullptr; // 初始化时传入并保存
  char *user_defined_name_;

  CameraProfile profile_;
  SyncMode sync_mode_;

  unsigned int converted_data_size_ = 0;
  unsigned char *converted_data_buffer_ = nullptr;
  MV_CC_PIXEL_CONVERT_PARAM convert_parameter_;
  std::string hik_logger = "hik";

  // rclcpp::Logger logger_ = rclcpp::get_logger("hikcamera");
};

ImageCapturer::ImageCapturer(const CameraProfile &profile,
                             char *user_defined_name,
                             const SyncMode &sync_mode) {
  impl_ = new ImageCapturer::Impl{profile, user_defined_name, sync_mode};
}

ImageCapturer::~ImageCapturer() { delete impl_; }

bool ImageCapturer::software_trigger_on() {
  return impl_->software_trigger_on();
}

bool ImageCapturer::set_frame_rate_inner_trigger_mode(float frame_rate) {
  return impl_->set_frame_rate_inner_trigger_mode(frame_rate);
}

cv::Mat
ImageCapturer::read(std::chrono::duration<unsigned int, std::milli> timeout) {
  return impl_->read(timeout);
}

std::tuple<int, int> ImageCapturer::get_width_height() const {
  return impl_->get_image_size();
}
} // namespace hikcamera