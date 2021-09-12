ps -ef|grep train | grep feature_config | awk '{print $2}' | xargs kill -9
