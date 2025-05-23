#!/bin/bash

trap "echo 'Exiting...'; exit 0" SIGINT SIGTERM

sleep 5

while true; do
    pkill wust_vision_trt
    wust_vision_trt
    sleep 1
done
