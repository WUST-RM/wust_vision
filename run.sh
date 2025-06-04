#!/bin/bash
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export MVCAM_SDK_PATH=/opt/MVS
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export MVCAM_GENICAM_CLPROTOCOL=/opt/MVS/lib/CLProtocol
export ALLUSERSPROFILE=/opt/MVS/MVFG

export LD_LIBRARY_PATH=/opt/MVS/lib/aarch64:/opt/MVS/lib/32:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

blue="\033[1;34m"
yellow="\033[1;33m"
reset="\033[0m"
red="\033[1;31m"

if [ ! -d "build" ]; then 
    mkdir build
fi

echo -e "${yellow}<--- Start CMake --->${reset}"
cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=YES ..  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="--gcc-toolchain=/usr"
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

if [ "$1" == "build" ]; then
    echo -e "${yellow}\n<--- Only building and copying both executables --->${reset}"
    # Copy the executables to /usr/local/bin
    sudo cp ./wust_vision_trt /usr/local/bin/
    if [ $? -ne 0 ]; then
        echo -e "${red}\n--- Failed to copy wust_vision_trt to /usr/local/bin ---${reset}"
        exit 1
    fi
    #sudo cp ./wust_vision_openvino /usr/local/bin/
    if [ $? -ne 0 ]; then
        echo -e "${red}\n--- Failed to copy wust_vision_openvino to /usr/local/bin ---${reset}"
        exit 1
    fi
    echo -e "${blue}\n----- Both executables copied to /usr/local/bin -----${reset}"
    exit 0
fi

sudo cp ./wust_vision_trt /usr/local/bin/
if [ $? -ne 0 ]; then
    echo -e "${red}\n--- Failed to copy wust_vision_trt to /usr/local/bin ---${reset}"
    exit 1
fi
#sudo cp ./wust_vision_openvino /usr/local/bin/
if [ $? -ne 0 ]; then
    echo -e "${red}\n--- Failed to copy wust_vision_openvino to /usr/local/bin ---${reset}"
    exit 1
fi
echo -e "${blue}\n----- Both executables copied to /usr/local/bin -----${reset}"



# Check input argument to decide which program to run
if [ "$1" == "trt" ]; then
    echo -e "${yellow}\n<--- Running TensorRT version --->${reset}"
    echo -e "${blue}\n-----WUST-VISION-TENSORRT-----${reset}"
    ./wust_vision_trt
    program_name="wust_vision_trt"
    if [ $? -ne 0 ]; then
        echo -e "${red}\n--- TensorRT program crashed, running guard_trt.sh ---${reset}"
        ./congig/guard_trt.sh
        exit 1
    fi
elif [ "$1" == "openvino" ]; then
    echo -e "${yellow}\n<--- Running OpenVINO version --->${reset}"
    echo -e "${blue}\n-----WUST-VISION-OPENVINO-----${reset}"
    ./wust_vision_openvino
    program_name="wust_vision_openvino"
    if [ $? -ne 0 ]; then
        echo -e "${red}\n--- OpenVINO program crashed, running guard_openvino.sh ---${reset}"
        ./config/guard_openvino.sh
        exit 1
    fi
else
    echo -e "${red}\n--- Invalid argument: Please specify 'trt' or 'openvino' ---${reset}"
    exit 1
fi

echo -e "${yellow}<----- OVER ----->${reset}"