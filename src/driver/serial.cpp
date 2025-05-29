#include "driver/serial.hpp"
#include "common/gobal.hpp"
#include "common/logger.hpp"
#include "common/tools.hpp"
#include "driver/crc8_crc16.hpp"
#include "driver/packet_typedef.hpp"
#include "type/type.hpp"
#include <cmath>
#include <iostream>
Serial::Serial()
    : device_name_(""), config_(SerialPortConfig()), is_usb_ok_(false),
      running_(false), driver_() {}
Serial::Serial(const std::string &device_name, const SerialPortConfig &config)
    : device_name_(device_name), config_(config), is_usb_ok_(false),
      running_(false), driver_() {}

Serial::~Serial() { stopThread(); }
void Serial::init(std::string device_name, SerialPortConfig config) {
  device_name_ = device_name;
  config_ = config;
}
void Serial::startThread() {
  running_ = true;
  protect_thread_ = std::thread(&Serial::serialPortProtect, this);
  receive_thread_ = std::thread(&Serial::receiveData, this);
  send_thread_ = std::thread(&Serial::sendData, this);
}

void Serial::stopThread() {
  running_ = false;
  if (protect_thread_.joinable()) {
    protect_thread_.join();
  }
  if (receive_thread_.joinable()) {
    receive_thread_.join();
  }
  if(send_thread_.joinable())
  {
    send_thread_.join();
  }
  if (driver_.is_open()) {
    driver_.close();
  }
}

void Serial::serialPortProtect() {
  WUST_INFO(serial_logger) << "Start serialPortProtect!";

  // 1. 初始化串口
  driver_.init_port(device_name_, config_);

  // 2. 尝试第一次打开
  try {
    if (!driver_.is_open()) {
      driver_.open();
      WUST_INFO(serial_logger) << "Serial port opened: " << device_name_;
    }
    is_usb_ok_ = true;
  } catch (const std::exception &ex) {
    WUST_ERROR(serial_logger) << "Open serial port failed: " << ex.what();
    is_usb_ok_ = false;
  }

  // 3. 循环监测，断开重连
  while (running_) {
    // 每次循环都打印一次状态
    //   WUST_INFO(serial_logger) << "[INFO] Serial port monitor (usb_ok=" <<
    //   is_usb_ok_
    //             << "): " << device_name_ ;

    if (!is_usb_ok_) {
      try {
        if (driver_.is_open()) {
          driver_.close();
          WUST_WARN(serial_logger) << "Serial port closed for reconnect";
        }
        driver_.open();
        if (driver_.is_open()) {
          WUST_INFO(serial_logger) << "Serial port re-opened successfully";
          is_usb_ok_ = true;
        }
      } catch (const std::exception &ex) {
        is_usb_ok_ = false;
        WUST_ERROR(serial_logger)
            << "Re-open serial port failed: " << ex.what() << "\n";
      }
    }

    std::this_thread::sleep_for(
        std::chrono::milliseconds(USB_PROTECT_SLEEP_MS));
  }

  WUST_INFO(serial_logger) << "serialPortProtect stopped";
}
void Serial::receiveData() {
  WUST_INFO(serial_logger) << "receiveData started";

  std::vector<uint8_t> sof_buf(1);
  std::vector<uint8_t> header_buf;
  std::vector<uint8_t> data_buf;

  int sof_count = 0;
  int retry_count = 0;

  while (running_) {
    // 如果串口未就绪，等待并重试
    if (!is_usb_ok_) {
      WUST_WARN(serial_logger)
          << "eceive: usb is not ok! Retry count: " << retry_count++;
      std::this_thread::sleep_for(
          std::chrono::milliseconds(USB_NOT_OK_SLEEP_MS));
      continue;
    }

    try {
      // 读一个字节，检查 SOF
      sof_buf.resize(39);
      driver_.receive(sof_buf);
      // if (sof_buf[0] != SOF_RECEIVE) {
      //   ++sof_count;
      //   // WUST_INFO(serial_logger) << "Finding SOF, count=" << sof_count ;
      //   continue;
      // }
      //sof_count = 0;

      // 读剩余 header（3 字节）
      // header_buf.resize(3);
      // driver_.receive(header_buf);
      // // 把 SOF 插到最前面
      // header_buf.insert(header_buf.begin(), sof_buf[0]);

      // // 反序列化 HeaderFrame
      // HeaderFrame hf = fromVector<HeaderFrame>(header_buf);

      // // CRC8 校验
      // if (!crc8::verify_CRC8_check_sum(reinterpret_cast<uint8_t *>(&hf),
      //                                  sizeof(hf))) {
      //   // WUST_ERROR(serial_logger) << "Header CRC8 failed\n";
      //   continue;
      // }

      // 读 data + CRC16
      //data_buf.resize(hf.len + 2);
      // int received = driver_.receive(data_buf);
      // int total = received;
      // int remain = (hf.len + 2) - received;
      // 如果没读完，就继续读
      // while (remain > 0) {
      //   std::vector<uint8_t> tmp(remain);
      //   int n = driver_.receive(tmp);
      //   data_buf.insert(data_buf.begin() + total, tmp.begin(), tmp.begin() + n);
      //   total += n;
      //   remain -= n;
      // }

      // // 把 header_buf 拼回 data_buf 前面，得到完整包
      // data_buf.insert(data_buf.begin(), header_buf.begin(), header_buf.end());

      //（可选）CRC16 校验
      // if (!crc16::verify_CRC16_check_sum(data_buf)) {
      //   std::cerr << "[ERROR] Data CRC16 failed\n";
      //   continue;
      // }

      // 根据 ID 解析并回调
      // switch (sof_buf[0]) {
      // case ID_AIM_INFO: {
      auto aim = fromVector<ReceiveAimINFO>(sof_buf);
      aim_cbk(aim);

      //   break;
      // }
      // case ID_IMU: {
      //   auto imu = fromVector<ReceiveImuData>(data_buf);
      //   imu_cbk(imu);
      //   break;
      // }
      // default:
      // WUST_WARN(serial_logger) << "Unknown packet ID=" << (int)sof_buf[0];
      // }

    } catch (const std::exception &ex) {
      WUST_ERROR(serial_logger) << "receiveData exception: " << ex.what();
      is_usb_ok_ = false; // 触发重连逻辑
    }
  }

  WUST_INFO(serial_logger) << "receiveData stopped";
}
void Serial::aim_cbk(ReceiveAimINFO &aim_data) {
  static uint32_t last_time = 0;
  static int valid_count = 0;
  static bool out_of_order_detected = false;
  static int last_reset_count = -1;
  
  //std::cout<<"roll:"<<aim_data.roll<<"pitch:"<<aim_data.pitch<<"yaw:"<<aim_data.yaw<<std::endl;

  // if (!out_of_order_detected) {
  //   if (aim_data.time_stamp <= last_time) {
  //     WUST_WARN(serial_logger)
  //         << "Received out-of-order imu data, entering recovery mode.";
  //     out_of_order_detected = true;
  //     valid_count = 0;
  //     last_time = aim_data.time_stamp;
  //   } else {
  //     last_time = aim_data.time_stamp;
  //   }
  // }
  
  // if (out_of_order_detected) {
  //   if (aim_data.time_stamp > last_time) {
  //     valid_count++;
  //     if (valid_count >= 100) {
  //       WUST_INFO(serial_logger) << "IMU timestamp recovered after 100 valid "
  //                                   "frames, exiting recovery mode.";
  //       out_of_order_detected = false;
  //       valid_count = 0;
  //       last_reset_count = -1;
  //     }
  //   } else {
  //     valid_count = 0;
  //     last_reset_count = -1;
  //   }
  //   return;
  // }
  double roll= aim_data.roll*M_PI/180.0;
  double pitch= aim_data.pitch*M_PI/180.0;
  double yaw= aim_data.yaw*M_PI/180.0;
 // WUST_INFO(serial_logger)<<"roll:"<<roll<<" pitch:"<<pitch<<" yaw:"<<yaw;
  
  if (aim_data.manual_reset_count != last_reset_count) {
    WUST_INFO(serial_logger)
        << "Manual reset count changed: " << last_reset_count << " -> "
        << aim_data.manual_reset_count;
    if_manual_reset = true;
    last_reset_count = aim_data.manual_reset_count;
  } else {
    if_manual_reset = false;
  }

  tf2::Quaternion q;
  
  
  
  q.setRPY(roll, pitch, yaw);

  Transform gimbal_tf(Position(0, 0, 0), q);
  tf_tree_.setTransform("gimbal_odom", "gimbal_link", gimbal_tf);

  detect_color_ = aim_data.detect_color;
  controller_delay = aim_data.controller_delay;
  velocity = aim_data.bullet_speed;

  if (debug_mode_) {
    dumpAimToFile(aim_data, "/tmp/aim_status.txt");
  }
}

void Serial::imu_cbk(ReceiveImuData &imu_data) {
  static uint32_t last_time = 0;
  static int valid_count = 0;
  static bool out_of_order_detected = false;

  if (!out_of_order_detected) {

    if (imu_data.time_stamp <= last_time) {
      WUST_WARN(serial_logger)
          << "Received out-of-order imu data, entering recovery mode.";
      out_of_order_detected = true;
      valid_count = 0;
      last_time = imu_data.time_stamp;
    } else {
      last_time = imu_data.time_stamp;
    }
  }

  if (out_of_order_detected) {

    if (imu_data.time_stamp > last_time) {
      valid_count++;
      // last_time = imu_data.time_stamp;
      if (valid_count >= 100) {
        WUST_INFO(serial_logger) << "IMU timestamp recovered after 100 valid "
                                    "frames, exiting recovery mode.";
        out_of_order_detected = false;
        valid_count = 0;
      }
    } else {
      valid_count = 0;
    }
    return;
  }

  imu_data.data.roll *= M_PI / 180.0;
  imu_data.data.pitch *= M_PI / 180.0;
  imu_data.data.yaw *= M_PI / 180.0;
  imu_data.data.roll_vel *= M_PI / 180.0;
  imu_data.data.pitch_vel *= M_PI / 180.0;
  imu_data.data.yaw_vel *= M_PI / 180.0;

  tf2::Quaternion q;
  q.setRPY(imu_data.data.roll, imu_data.data.pitch, imu_data.data.yaw);

  Transform gimbal_tf(Position(0, 0, 0), q);

  tf_tree_.setTransform("gimbal_odom", "gimbal_link", gimbal_tf);
  if (debug_mode_) {
    dumpImuToFile(imu_data, "/tmp/imu_status.txt");
  }
}

void Serial::sendData() {
  WUST_INFO(serial_logger) << "Start sendData!";

  //send_robot_cmd_data_.frame_header.sof = SOF_SEND;
  send_robot_cmd_data_.cmd_ID = ID_ROBOT_CMD;
  //send_robot_cmd_data_.frame_header.len = sizeof(SendRobotCmdData) - 6;
  // send_robot_cmd_data_.data.speed_vector.vx = 0;
  // send_robot_cmd_data_.data.speed_vector.vy = 0;
  // send_robot_cmd_data_.data.speed_vector.wz = 0;
  // 添加帧头crc8校验
  // crc8::append_CRC8_check_sum(
  //     reinterpret_cast<uint8_t *>(&send_robot_cmd_data_), sizeof(HeaderFrame));

  int retry_count = 0;

  while (running_) {
    if (!is_usb_ok_) {
      WUST_WARN(serial_logger)
          << "send: usb is not ok! Retry count:" << retry_count++;
      std::this_thread::sleep_for(
          std::chrono::milliseconds(USB_NOT_OK_SLEEP_MS));
      continue;
    }

    try {
      // 整包数据校验
      // 添加数据段crc16校验
      // crc16::append_CRC16_check_sum(
      //     reinterpret_cast<uint8_t *>(&send_robot_cmd_data_),
      //     sizeof(SendRobotCmdData));

      // 发送数据
      //std::cout  << "send_robot_cmd_data_" << std::endl;
      std::vector<uint8_t> send_data = toVector(send_robot_cmd_data_);
      driver_.send(send_data);
    } catch (const std::exception &ex) {
      WUST_ERROR(serial_logger) << "Error sending data: " << ex.what();
      is_usb_ok_ = false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1000/control_rate));
  }
}
void Serial::transformGimbalCmd(GimbalCmd &gimbal_cmd) {
  // send_robot_cmd_data_.data.gimbal.yaw=gimbal_cmd.yaw;
  // send_robot_cmd_data_.data.gimbal.pitch=gimbal_cmd.pitch;
  // send_robot_cmd_data_.data.gimbal.distance=gimbal_cmd.distance;
  // send_robot_cmd_data_.data.shoot.pitch_diff=gimbal_cmd.pitch_diff;
  // send_robot_cmd_data_.data.shoot.yaw_diff=gimbal_cmd.yaw_diff;
  // send_robot_cmd_data_.data.shoot.fire=gimbal_cmd.fire_advice;
  // send_robot_cmd_data_.data.debug.detect_color=detect_color_;
  send_robot_cmd_data_.yaw = gimbal_cmd.yaw;
  send_robot_cmd_data_.pitch = gimbal_cmd.pitch;
  send_robot_cmd_data_.distance = gimbal_cmd.distance;
  send_robot_cmd_data_.pitch_diff = gimbal_cmd.pitch_diff;
  send_robot_cmd_data_.yaw_diff = gimbal_cmd.yaw_diff;
  send_robot_cmd_data_.fire = gimbal_cmd.fire_advice;
  send_robot_cmd_data_.detect_color = detect_color_;
}
