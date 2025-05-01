#!/bin/bash

export MVCAM_SDK_PATH=/opt/MVS
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export MVCAM_GENICAM_CLPROTOCOL=/opt/MVS/lib/CLProtocol
export ALLUSERSPROFILE=/opt/MVS/MVFG

export LD_LIBRARY_PATH=/opt/MVS/lib/64:/opt/MVS/lib/32:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/hy/TensorRT-8.5.2.2/lib:$LD_LIBRARY_PATH

blue="\033[1;34m"
yellow="\033[1;33m"
reset="\033[0m"

if [ ! -d "build" ]; then 
    mkdir build
fi

echo -e "${yellow}<--- Start CMake --->${reset}"
cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=YES .. 
if [ $? -ne 0 ]; then
    echo -e "${red}\n--- CMake Failed ---${reset}"
    exit 1
fi

echo -e "${yellow}\n<--- Start Make --->${reset}"
max_threads=$(grep -c "processor" /proc/cpuinfo)
make -j "$max_threads"
if [ $? -ne 0 ]; then
    echo -e "${red}\n--- Make Failed ---${reset}"
    exit 1
fi

# 计算代码总行数（排除指定目录）
echo -e "${yellow}\n<--- Total Lines --->${reset}"
total=$(find .. \
    -type d \( \
        -path ../build -o \
        -path ../hikSDK -o \
        -path ../model \
    \) -prune -o \
    -type f \( \
        -name "*.cpp" -o \
        -name "*.hpp" -o \
        -name "*.c" -o \
        -name "*.h" \
    \) -exec wc -l {} + | awk 'END{print $1}')
echo -e "${blue}        $total${reset}"

echo -e "${yellow}\n<--- Run Code --->${reset}"
echo -e "${blue}\n-----WUST-VISION-----${reset}"
./wust_vision
if [ $? -ne 0 ]; then
    echo -e "${red}\n--- Program exited with error ---${reset}"
    exit 1
fi

echo -e "${yellow}<----- OVER ----->${reset}"