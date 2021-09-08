# cat data/20210711 > data/orig.train
# cat data/20210712 >> data/orig.train
# cat data/20210713 >> data/orig.train
# cat data/20210714 >> data/orig.train
# cat data/20210715 >> data/orig.train
# cat data/20210716 >> data/orig.train
# cat data/20210717 >> data/orig.train
# cat data/20210718 >> data/orig.train
# cat data/20210719 >> data/orig.train
# cat data/20210720 >> data/orig.train
# cat data/20210721 >> data/orig.train
# cat data/20210722 >> data/orig.train
# cat data/20210723 >> data/orig.train
# cat data/20210724 >> data/orig.train
# cat data/20210725 >> data/orig.train
# cat data/20210726 >> data/orig.train
# cat data/20210727 >> data/orig.train
# cat data/20210728 >> data/orig.train
# cat data/20210729 >> data/orig.train
# cat data/20210730 >> data/orig.train
# cat data/20210731 >> data/orig.train
# cat data/20210801 >> data/orig.train
# cat data/20210802 >> data/orig.train
# cat data/20210803 >> data/orig.train
# cat data/20210804 >> data/orig.train
# cat data/20210805 >> data/orig.train
# cat data/20210806 >> data/orig.train
# cat data/20210807 >> data/orig.train
# cat data/20210808 >> data/orig.train
# cat data/20210809 >> data/orig.train
# cat data/20210810 >> data/orig.train
# cat data/20210811 >> data/orig.train
# cat data/20210812 > data/orig.test
# cat data/20210813 > data/orig.valid
# 
mkdir -p npdata1/train 
mkdir -p npdata1/test
mkdir -p npdata1/valid

cat data/orig.test |  python3 feamap.py  --mode='to_np' -o=npdata1/test  > log/feamap.test
cat data/orig.valid |  python3 feamap.py  --mode='to_np' -o=npdata1/valid > log/feamap.valid
cat data/orig.train |  python3 feamap.py  --mode='to_np' -o=npdata1/train > log/feamap.train

