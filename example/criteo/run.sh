

# make config
cd config
cat ../data/criteo_sampled_data.csv | python3 make_feat_conf.py -o criteo --cpu_num 3
cd -

# train
bin/train data_formart=csv feat_sep=, feat_cfg=criteo train=data/criteo_sampled_data.csv.train valid=data/criteo_sampled_data.csv.test threads=4 verbose=0 epoch=11 solver=adam batch_size=2000 mf=txt om=model4

# pred
bin/pred data_formart=csv feat_sep=, feat_cfg=criteo train=data/criteo_sampled_data.csv.train
