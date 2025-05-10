// serial_port_config.hpp
#pragma once

#include <boost/asio/serial_port_base.hpp>
#include <boost/asio.hpp>

/// 串口参数配置结构
struct SerialPortConfig {
    unsigned int baud_rate;  ///< 波特率
    unsigned int char_size;  ///< 数据位
    boost::asio::serial_port_base::parity::type parity;
    boost::asio::serial_port_base::stop_bits::type stop_bits;
    boost::asio::serial_port_base::flow_control::type flow_control;

    SerialPortConfig(
        unsigned int baud = 115200,
        unsigned int csize = 8,
        boost::asio::serial_port_base::parity::type par = boost::asio::serial_port_base::parity::none,
        boost::asio::serial_port_base::stop_bits::type sb = boost::asio::serial_port_base::stop_bits::one,
        boost::asio::serial_port_base::flow_control::type fc = boost::asio::serial_port_base::flow_control::none
    )
      : baud_rate(baud),
        char_size(csize),
        parity(par),
        stop_bits(sb),
        flow_control(fc)
    {}
};
class SerialDriver {
    public:
      SerialDriver()
        : io_(), port_(io_)
      {}
    
      /// 初始化：仅保存设备名和配置
      void init_port(const std::string& device, const SerialPortConfig& config) {
        device_ = device;
        config_ = config;
      }
    
      /// 打开并应用配置
      void open() {
        port_.open(device_);
        port_.set_option(boost::asio::serial_port_base::baud_rate(config_.baud_rate));
        port_.set_option(boost::asio::serial_port_base::character_size(config_.char_size));
        port_.set_option(boost::asio::serial_port_base::parity(config_.parity));
        port_.set_option(boost::asio::serial_port_base::stop_bits(config_.stop_bits));
        port_.set_option(boost::asio::serial_port_base::flow_control(config_.flow_control));
      }
    
      /// 关闭串口
      void close() {
        if (port_.is_open()) port_.close();
      }
    
      /// 是否已打开
      bool is_open() const {
        return port_.is_open();
      }
    
      /// 接收数据，buf 必须预先 resize 好长度
      /// @return 实际接收到的字节数
      std::size_t receive(std::vector<uint8_t>& buf) {
        boost::system::error_code ec;
        std::size_t n = port_.read_some(boost::asio::buffer(buf), ec);
        if (ec) throw boost::system::system_error(ec);
        return n;
      }
    
      /// 发送一段数据
      void send(const std::vector<uint8_t>& data) {
        boost::system::error_code ec;
        boost::asio::write(port_, boost::asio::buffer(data), ec);
        if (ec) throw boost::system::system_error(ec);
      }
    
    private:
      boost::asio::io_service           io_;
      boost::asio::serial_port          port_;
      SerialPortConfig                  config_;
      std::string                       device_;
    };
