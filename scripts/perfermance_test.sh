#!/bin/bash

now=`date +'%Y-%m-%d %H:%M:%S'`
start_time=$(date --date="$now" +%s);


../bin/train  feat_cfg=../config/feature_config.json train=../../train valid=../../train threads=4  verbose=0 epoch=30

now=`date +'%Y-%m-%d %H:%M:%S'`
end_time=$(date --date="$now" +%s);
echo "used time:"$((end_time-start_time))"s"

