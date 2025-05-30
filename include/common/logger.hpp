#pragma once

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
namespace fs = std::filesystem;

// 日志等级定义
enum class LogLevel {
  DEBUG = 0,
  INFO,
  WARN,
  ERROR,
};

// 日志等级转字符串
inline const char *levelToString(LogLevel level) {
  switch (level) {
  case LogLevel::DEBUG:
    return "DEBUG";
  case LogLevel::INFO:
    return " INFO";
  case LogLevel::WARN:
    return " WARN";
  case LogLevel::ERROR:
    return "ERROR";
  default:
    return "UNKN ";
  }
}

// 日志颜色
inline const char *colorForLevel(LogLevel level) {
  switch (level) {
  case LogLevel::DEBUG:
    return "\033[36m"; // Cyan
  case LogLevel::INFO:
    return "\033[32m"; // Green
  case LogLevel::WARN:
    return "\033[33m"; // Yellow
  case LogLevel::ERROR:
    return "\033[31m"; // Red
  default:
    return "\033[0m";
  }
}

inline const char *colorReset() { return "\033[0m"; }

inline LogLevel logLevelFromString(const std::string &level_str) {
  std::string l = level_str;
  std::transform(l.begin(), l.end(), l.begin(), ::toupper);
  if (l == "DEBUG")
    return LogLevel::DEBUG;
  if (l == "INFO")
    return LogLevel::INFO;
  if (l == "WARN")
    return LogLevel::WARN;
  if (l == "ERROR")
    return LogLevel::ERROR;

  throw std::invalid_argument("Invalid log level string: " + level_str);
}

// 时间戳
inline std::string getTimeStr() {
  auto now = std::chrono::system_clock::now();
  auto now_tt = std::chrono::system_clock::to_time_t(now);
  auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()) %
                1000;
  std::ostringstream oss;
  oss << std::put_time(std::localtime(&now_tt), "%Y-%m-%d %H:%M:%S") << "."
      << std::setfill('0') << std::setw(3) << now_ms.count();
  return oss.str();
}

// ========== 核心 Logger 类 ==========
class Logger {
public:
  static Logger &getInstance() {
    static Logger instance;
    return instance;
  }

  void setLevel(const std::string &level_str) {
    setLevel(logLevelFromString(level_str));
  }

  void setLevel(LogLevel level) { log_level_ = level; }

  void enableFileOutput(const std::string &filename) {
    std::lock_guard<std::mutex> lock(mutex_);
    file_stream_.open(filename, std::ios::out | std::ios::app);
    file_output_enabled_ = file_stream_.is_open();
  }

  void disableColorOutput() { color_output_enabled_ = false; }

  void enableSimplifiedOutput(bool enabled) {
    simplified_output_enabled_ = enabled;
  }

  LogLevel getLevel() const { return log_level_; }

  bool shouldLog(LogLevel level) const { return level >= log_level_; }

  bool isSimplifiedOutputEnabled() const { return simplified_output_enabled_; }

  std::ofstream &fileStream() { return file_stream_; }
  bool isFileOutputEnabled() const { return file_output_enabled_; }
  bool isColorOutputEnabled() const { return color_output_enabled_; }
  std::mutex &getMutex() { return mutex_; }

private:
  Logger()
      : log_level_(LogLevel::DEBUG), file_output_enabled_(false),
        color_output_enabled_(true), simplified_output_enabled_(false) {}
  ~Logger() {
    if (file_stream_.is_open())
      file_stream_.close();
  }

  LogLevel log_level_;
  bool file_output_enabled_;
  bool color_output_enabled_;
  bool simplified_output_enabled_;
  std::ofstream file_stream_;
  mutable std::mutex mutex_;
};

// ========== 流式日志类 ==========
class LoggerStream {
public:
  LoggerStream(LogLevel level, const std::string &node, const char *file,
               int line)
      : level_(level), node_name_(node), file_(file), line_(line) {}

  ~LoggerStream() {
    // 构造输出内容
    std::ostringstream full_msg;
    if (Logger::getInstance().isSimplifiedOutputEnabled()) {
      // 简化格式： [LEVEL][node] message
      full_msg << "[" << levelToString(level_) << "]"
               << "[" << node_name_ << "] " << buffer_.str();
    } else {
      // 完整格式： [时间][LEVEL][node][文件:行号] message
      full_msg << "[" << getTimeStr() << "]"
               << "[" << levelToString(level_) << "]"
               << "[" << node_name_ << "]"
               << "[" << file_ << ":" << line_ << "] " << buffer_.str();
    }

    std::lock_guard<std::mutex> lock(Logger::getInstance().getMutex());

    // 控制台只打印等级满足的
    if (Logger::getInstance().shouldLog(level_)) {
      if (Logger::getInstance().isColorOutputEnabled()) {
        std::cout << colorForLevel(level_) << full_msg.str() << colorReset()
                  << std::endl;
      } else {
        std::cout << full_msg.str() << std::endl;
      }
    }

    // 文件日志写入全部，不受等级限制
    if (Logger::getInstance().isFileOutputEnabled()) {
      Logger::getInstance().fileStream() << full_msg.str() << std::endl;
    }
  }

  template <typename T> LoggerStream &operator<<(const T &val) {
    buffer_ << val;
    return *this;
  }

private:
  LogLevel level_;
  std::ostringstream buffer_;
  std::string node_name_;
  const char *file_;
  int line_;
};

// 初始化 Logger
inline void initLogger(const std::string &level_str,
                       const std::string &log_dir = "./logs",
                       bool use_logcli = true, bool use_logfile = true,
                       bool simplified_output = false) {
  Logger &logger = Logger::getInstance();

  // 设置日志等级
  logger.setLevel(logLevelFromString(level_str));

  // 设置简化日志输出选项
  logger.enableSimplifiedOutput(simplified_output);

  // 控制是否启用控制台输出（控制台输出由 shouldLog() + use_logcli 控制）
  // 这里我们通过改写 LoggerStream
  // 的打印条件或者加一个新的标志更复杂，简化处理：
  // 如果不启用控制台输出，则设置等级为 ERROR+，不打印普通信息即可。
  if (!use_logcli) {
    // 直接设为最高等级，避免打印
    logger.setLevel(LogLevel::ERROR);
  }

  // 处理日志目录为绝对路径
  fs::path dir_path(log_dir);
  if (!dir_path.is_absolute()) {
    dir_path = fs::absolute(dir_path);
  }

  // 获取当前时间作为文件名
  std::string timestamp = getTimeStr(); // 2025-05-24 13:30:45.123
  std::replace(timestamp.begin(), timestamp.end(), ':', '-');
  std::replace(timestamp.begin(), timestamp.end(), ' ', '_');

  // 拼接绝对路径文件名
  fs::path filename = dir_path / ("log_" + timestamp + ".txt");

  // 创建目录（C++17）
  fs::create_directories(dir_path);

  // 启用文件日志
  if (use_logfile) {
    logger.enableFileOutput(filename.string());
  }
}

// 宏定义
#define WUST_DEBUG(node) LoggerStream(LogLevel::DEBUG, node, __FILE__, __LINE__)
#define WUST_INFO(node) LoggerStream(LogLevel::INFO, node, __FILE__, __LINE__)
#define WUST_WARN(node) LoggerStream(LogLevel::WARN, node, __FILE__, __LINE__)
#define WUST_ERROR(node) LoggerStream(LogLevel::ERROR, node, __FILE__, __LINE__)

