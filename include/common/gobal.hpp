#pragma once

#include "common/tf.hpp"
#include "detect/mono_measure_tool.hpp"


extern TfTree tf_tree_;
extern std::unique_ptr<MonoMeasureTool> measure_tool_;
extern int detect_color_;
extern bool debug_mode_ ;
extern int debug_w;
extern int debug_h;
extern double controller_delay;
extern double velocity ;