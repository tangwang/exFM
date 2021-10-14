rm -rf feat_config_0811
rm -rf fea_config_08_all
cat /home/SanJunipero/rd/tangwang/dj/rank_new/fmdata_new/20210811 | python3 make_feat_conf.py -o feat_config_0811 > log.fea_statis 2>log.fea_statis.err
cat /home/SanJunipero/rd/tangwang/dj/rank_new/fmdata_new/202108*  | python3 make_feat_conf.py -o fea_config_08_all > log.fea_statis_08_all 2>log.fea_statis_08_all.err
