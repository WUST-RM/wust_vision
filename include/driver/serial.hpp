#include "common/gobal.hpp"
#include "driver/packet_typedef.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <thread>
#include "driver/serial_type.hpp"
static constexpr int USB_PROTECT_SLEEP_MS = 1000;
static constexpr int USB_NOT_OK_SLEEP_MS =1000;
class Serial
{
public:
    Serial();
    Serial(const std::string &device_name,
        const SerialPortConfig &config);
    ~Serial();
    void init(std::string device_name,SerialPortConfig config);
    void serialPortProtect();
    void startThread();
    void stopThread();
    void receiveData();
    void imu_cbk(ReceiveImuData & imu_data);
    bool usbOk() const { return is_usb_ok_; }
    void sendData();
    void transformGimbalCmd(GimbalCmd & gimbal_cmd);


  std::string              device_name_;
  SerialPortConfig         config_;
  std::atomic<bool>        is_usb_ok_;
  std::atomic<bool>        running_;
  std::thread              protect_thread_;
  std::thread              receive_thread_;
  std::thread              send_thread_;
  SerialDriver             driver_;
  std::string              serial_logger="serial";
  SendRobotCmdData send_robot_cmd_data_;
};