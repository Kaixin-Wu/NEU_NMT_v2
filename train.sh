#!/bin/sh

cuda=$1
log=$2
CUDA_VISIBLE_DEVICES=$cuda nohup python3 -u src/train.py > $log &
