
g++ -I.. -lpthread  test_sync.cc
exit

g++ -std=c++11  -I../../third_party -I../ -I../feature -I../ftrl -I../utils ../feature/dense_fea.cc ../feature/common_fea.cc ../ftrl/param.cc ../ftrl/train_opt.cc ../feature/sparse_fea.cc ../feature/varlen_sparse_fea.cc ../feature/fea_manager.cc  ../ftrl/ftrl_trainer.cc  test_fea_manager.cc -o test_fea 
exit


g++ -std=c++11  -I../../third_party test_fea_config.cc

g++ -std=c++11  -I../ test_dict.cc

