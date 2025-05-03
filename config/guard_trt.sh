#!/bin/bash

sleep 5

while true; do
    pkill wust_vision_trt
    wust_vision_trt
    sleep 1
done