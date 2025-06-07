#include "common/matplottools.hpp"
#include "common/logger.hpp"
#include <chrono>
#include <vector>
void plotYawThread() {
  try {
    matplotlibcpp::ion();
  } catch (const std::exception &e) {
    std::cerr << "Failed to initialize matplotlib interactive mode: "
              << e.what();
    return;
  }

  bool figureClosed = false;

  while (is_inited_) {
    std::vector<double> time_list, yaw_list;
    {
      std::lock_guard<std::mutex> lock(yaw_log_mutex_);
      for (const auto &[t, yaw] : target_yaw_log_) {
        time_list.push_back(t);
        yaw_list.push_back(yaw);
      }
    }

    matplotlibcpp::clf();
    if (!time_list.empty())
      matplotlibcpp::plot(time_list, yaw_list);
    matplotlibcpp::title("Target Yaw Over Time");
    matplotlibcpp::xlabel("Time (s)");
    matplotlibcpp::ylabel("Yaw (rad)");
    matplotlibcpp::grid(true);
    matplotlibcpp::pause(0.001);

    std::this_thread::sleep_for(std::chrono::milliseconds(50)); // 20 Hz
  }

  WUST_DEBUG("matplot") << "[plotYawThread] Ending...";

  try {
    if (!figureClosed) {
      matplotlibcpp::close();
      // 给窗口关闭一点时间
      for (int i = 0; i < 10; ++i)
        matplotlibcpp::pause(0.05);

      // Python 层彻底关闭所有窗口并切后端，避免卡死
      matplotlibcpp::detail::_interpreter::get();
      PyRun_SimpleString("import matplotlib.pyplot as plt\n"
                         "plt.close('all')\n"
                         "plt.switch_backend('agg')\n");

      figureClosed = true;
      WUST_DEBUG("matplot") << "[plotYawThread] Figure closed successfully.";
    }
  } catch (const std::exception &e) {
    std::cerr << "[plotYawThread] Exception on close: " << e.what();
  }

  try {
    matplotlibcpp::detail::_interpreter::kill();
  } catch (...) {
    // 忽略异常
  }

  WUST_DEBUG("matplot") << "[plotYawThread] Fully terminated.";
}

void plotRobotCmdThread() {

  matplotlibcpp::ion();
  bool figureClosed = false;

  while (is_inited_) {
    std::vector<double> time_list, yaw_list, pitch_list;
    {
      std::lock_guard<std::mutex> lock(robot_cmd_mutex_);
      for (size_t i = 0; i < time_log_.size(); ++i) {
        time_list.push_back(time_log_[i]);
        yaw_list.push_back(cmd_yaw_log_[i]);
        pitch_list.push_back(cmd_pitch_log_[i]);
      }
    }

    matplotlibcpp::clf();

    matplotlibcpp::named_plot("Yaw", time_list, yaw_list, "r-");
    matplotlibcpp::named_plot("Pitch", time_list, pitch_list, "b-");

    matplotlibcpp::legend();
    matplotlibcpp::title("Robot Command Yaw and Pitch Over Time");
    matplotlibcpp::xlabel("Time (s)");
    matplotlibcpp::ylabel("Angle (rad)");
    matplotlibcpp::grid(true);
    matplotlibcpp::pause(0.001);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  WUST_DEBUG("matplot") << "plotRobotCmdThread ending";

  // 多次尝试关闭图形窗口
  try {
    if (!figureClosed) {
      // 1. 显式关闭图形窗口
      matplotlibcpp::close();

      // 2. 确保事件循环完全停止
      // 多次处理剩余事件
      for (int i = 0; i < 10; i++) {
        matplotlibcpp::pause(0.05);
      }

      // 3. 强制关闭所有 Matplotlib 资源
      // 尝试通过 Python 命令彻底关闭
      matplotlibcpp::detail::_interpreter::get();
      PyRun_SimpleString("import matplotlib.pyplot as plt\n"
                         "plt.close('all')\n"
                         "plt.switch_backend('agg')\n"); // 切换到非交互式后端

      figureClosed = true;
      WUST_DEBUG("matplot") << "Figure closed successfully";
    }
  } catch (const std::exception &e) {
    std::cerr << "Exception on close: " << e.what();
  }

  // 4. 确保 Python 解释器清理资源
  try {
    matplotlibcpp::detail::_interpreter::kill();
  } catch (...) {
    // 忽略可能的异常
  }

  WUST_DEBUG("matplot") << "plotRobotCmdThread fully terminated";
}

void plotAllThread() {
  try {
    matplotlibcpp::ion(); // 启用交互模式
  } catch (const std::exception &e) {
    std::cerr << "Failed to initialize matplotlib interactive mode: "
              << e.what();
    return;
  }

  bool figureClosed = false;

  while (is_inited_) {
    std::vector<double> time_list_target, target_yaw_list;
    std::vector<double> time_list_cmd, cmd_yaw_list, cmd_pitch_list;

    {
      std::lock_guard<std::mutex> lock(yaw_log_mutex_);
      for (const auto &[t, yaw] : target_yaw_log_) {
        time_list_target.push_back(t);
        target_yaw_list.push_back(yaw * 180.0 / M_PI); // 转为角度显示
      }
    }

    {
      std::lock_guard<std::mutex> lock(robot_cmd_mutex_);
      time_list_cmd = time_log_;
      cmd_yaw_list = cmd_yaw_log_;
      cmd_pitch_list = cmd_pitch_log_;
    }

    matplotlibcpp::clf();

    if (!time_list_target.empty())
      matplotlibcpp::named_plot("Target Yaw", time_list_target, target_yaw_list,
                                "g-");

    if (!time_list_cmd.empty()) {
      matplotlibcpp::named_plot("Cmd Yaw", time_list_cmd, cmd_yaw_list, "r-");
      matplotlibcpp::named_plot("Cmd Pitch", time_list_cmd, cmd_pitch_list,
                                "b-");
    }

    matplotlibcpp::legend();
    matplotlibcpp::title("Yaw & Pitch Over Time");
    matplotlibcpp::xlabel("Time (s)");
    matplotlibcpp::ylabel("Angle (deg)");
    matplotlibcpp::grid(true);
    matplotlibcpp::pause(0.001);

    std::this_thread::sleep_for(std::chrono::milliseconds(50)); // 20Hz
  }

  WUST_DEBUG("matplot") << "[plotAllThread] Ending...";

  // 正确关闭图形窗口
  try {
    if (!figureClosed) {
      matplotlibcpp::close(); // C++ 层调用
      for (int i = 0; i < 10; ++i)
        matplotlibcpp::pause(0.05); // 给窗口时间关闭

      // Python 层彻底关闭所有窗口并切后端
      matplotlibcpp::detail::_interpreter::get();
      PyRun_SimpleString("import matplotlib.pyplot as plt\n"
                         "plt.close('all')\n"
                         "plt.switch_backend('agg')\n");

      figureClosed = true;
      WUST_DEBUG("matplot") << "[plotAllThread] Figure closed successfully.";
    }
  } catch (const std::exception &e) {
    std::cerr << "[plotAllThread] Exception on close: " << e.what();
  }

  try {
    matplotlibcpp::detail::_interpreter::kill(); // 释放 Python
  } catch (...) {
    // 忽略 kill 异常
  }

  WUST_DEBUG("matplot") << "[plotAllThread] Fully terminated.";
}
