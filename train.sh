#!/bin/bash

cpu_num=30

bin/train \
    data_formart=csv \
    feat_sep=, \
    feat_cfg=conf1 \
    train=data/train.csv \
    valid=data/test.csv \
    threads=$cpu_num \
    verbose=1 \
    epoch=1 \
    solver=ftrl \
    batch_size=10 \
    mf=txt \
    om=model_output


