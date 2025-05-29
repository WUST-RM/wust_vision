#pragma once
#include "type/type.hpp"
#include <vector>

double orientationToYaw(const tf2::Quaternion &orientation);
void command_callback(Armors &armors);
void ex(double &a, double &min, double &max);