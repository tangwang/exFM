ps -ef|grep train | grep feat_cfg | awk '{print $2}' | xargs kill -9
