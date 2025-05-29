#!/bin/bash

sleep 5

while true; do
    pkill wust_vision_openvino
    wust_vision_openvino
    sleep 1
done