#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <chrono>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>


// 日志等级定义
enum class LogLevel {
    DEBUG = 0,
    INFO,
    WARN,
    ERROR,
};

// 日志等级转字符串
inline const char* levelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return " INFO";
        case LogLevel::WARN:  return " WARN";
        case LogLevel::ERROR: return "ERROR";
        default:              return "UNKN ";
    }
}

// 日志颜色
inline const char* colorForLevel(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "\033[36m";  // Cyan
        case LogLevel::INFO:  return "\033[32m";  // Green
        case LogLevel::WARN:  return "\033[33m";  // Yellow
        case LogLevel::ERROR: return "\033[31m";  // Red
        default:              return "\033[0m";
    }
}

inline const char* colorReset() {
    return "\033[0m";
}

// 时间戳
inline std::string getTimeStr() {
    auto now = std::chrono::system_clock::now();
    auto now_tt = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) % 1000;
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&now_tt), "%Y-%m-%d %H:%M:%S")
        << "." << std::setfill('0') << std::setw(3) << now_ms.count();
    return oss.str();
}

// ========== 核心 Logger 类 ==========
class Logger {
public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    void setLevel(LogLevel level) { log_level_ = level; }

    void enableFileOutput(const std::string& filename) {
        std::lock_guard<std::mutex> lock(mutex_);
        file_stream_.open(filename, std::ios::out | std::ios::app);
        file_output_enabled_ = file_stream_.is_open();
    }

    void disableColorOutput() { color_output_enabled_ = false; }

    LogLevel getLevel() const { return log_level_; }

    bool shouldLog(LogLevel level) const { return level >= log_level_; }

    std::ofstream& fileStream() { return file_stream_; }
    bool isFileOutputEnabled() const { return file_output_enabled_; }
    bool isColorOutputEnabled() const { return color_output_enabled_; }
    std::mutex& getMutex() { return mutex_; }

private:
    Logger() : log_level_(LogLevel::DEBUG), file_output_enabled_(false), color_output_enabled_(true) {}
    ~Logger() { if (file_stream_.is_open()) file_stream_.close(); }

    LogLevel log_level_;
    bool file_output_enabled_;
    bool color_output_enabled_;
    std::ofstream file_stream_;
    mutable std::mutex mutex_;
};

// ========== 流式日志类 ==========
class LoggerStream {
public:
    LoggerStream(LogLevel level, const std::string& node, const char* file, int line)
        : level_(level), node_name_(node), file_(file), line_(line) {}

    ~LoggerStream() {
        if (!Logger::getInstance().shouldLog(level_)) return;

        std::ostringstream full_msg;
        full_msg << "[" << getTimeStr() << "]"
                 << "[" << levelToString(level_) << "]"
                 << "[" << node_name_ << "]"
                 << "[" << file_ << ":" << line_ << "] "
                 << buffer_.str();

        std::lock_guard<std::mutex> lock(Logger::getInstance().getMutex());

        // 输出到控制台
        if (Logger::getInstance().isColorOutputEnabled()) {
            std::cout << colorForLevel(level_) << full_msg.str()
                      << colorReset() << std::endl;
        } else {
            std::cout << full_msg.str() << std::endl;
        }

        // 输出到文件
        if (Logger::getInstance().isFileOutputEnabled()) {
            Logger::getInstance().fileStream() << full_msg.str() << std::endl;
        }
    }

    template<typename T>
    LoggerStream& operator<<(const T& val) {
        buffer_ << val;
        return *this;
    }

private:
    LogLevel level_;
    std::ostringstream buffer_;
    std::string node_name_;
    const char* file_;
    int line_;
};

// ========== 宏定义 ==========
#define WUST_DEBUG(node) LoggerStream(LogLevel::DEBUG, node, __FILE__, __LINE__)
#define WUST_INFO(node)  LoggerStream(LogLevel::INFO,  node, __FILE__, __LINE__)
#define WUST_WARN(node)  LoggerStream(LogLevel::WARN,  node, __FILE__, __LINE__)
#define WUST_ERROR(node) LoggerStream(LogLevel::ERROR, node, __FILE__, __LINE__)


#include "NvInfer.h"

class TRTLogger : public nvinfer1::ILogger
{
public:
  explicit TRTLogger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kWARNING)
  : severity_(severity)
  {
  }
  void log(nvinfer1::ILogger::Severity severity, const char * msg) noexcept override
  {
    if (severity <= severity_) {
      std::cerr << msg << std::endl;
    }
  }
  nvinfer1::ILogger::Severity severity_;
};


