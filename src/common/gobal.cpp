#include "common/gobal.hpp"
TfTree tf_tree_;
std::unique_ptr<MonoMeasureTool> measure_tool_;
int detect_color_;
bool debug_mode_ = false;
int debug_w;
int debug_h;
double controller_delay = 0.0;
double velocity = 15.0;
bool if_manual_reset = false;
int control_rate=1000;