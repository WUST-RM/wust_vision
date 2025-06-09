#pragma once
#include "common/gobal.hpp"
#include "common/matplotlibcpp.h"
#include "common/toolsgobal.hpp"
#include "type/type.hpp"
#include <thread>

void plotYawThread();
void plotRobotCmdThread();
void plotAllThread();
void write_cmd_log_to_json();
void robotCmdLoggerThread();