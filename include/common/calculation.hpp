#pragma once
#include <vector>
#include "type/type.hpp"


double orientationToYaw(const tf2::Quaternion& orientation);
void command_callback(Armors& armors);
void ex(double& a,double& min,double& max);