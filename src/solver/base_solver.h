/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/feat_manager.h"
#include "solver/parammeter_container.h"
#include "utils/base.h"
#include "train/train_opt.h"

class Sample {
 public:
  Sample() {}
  ~Sample() {}

  real_t forward();

  void backward();

  size_t fm_layer_nodes_size;
  vector<FmLayerNode> fm_layer_nodes;
  
  real_t logit;
  real_t loss;
  real_t grad;
  real_t sum[DIM];
  real_t sum_sqr[DIM];
  union {
    int i;    // for classification
    real_t f; // for regression
  } label;
};

class BaseSolver {
 public:
  BaseSolver(const FeatManager &feat_manager);

  virtual ~BaseSolver() {}

  void train(const string & line, int &y, real_t &logit, real_t & loss, real_t & grad);
  void test(const string & line, int &y, real_t &logit);

protected:
  real_t feedLine_libSVM(const string & aline);
  real_t feedLine_CSV(const string & aline);
  real_t (BaseSolver::*lineProcessor)(const string & aline);
  
  virtual void update() {}

  void rotateSampleIdx();

  void batchReduce(FMParamUnit &grad, int count);

protected:
  const FeatManager &feat_manager_;
  vector<DenseFeatContext> dense_feats;
  vector<SparseFeatContext> sparse_feats;
  vector<VarlenSparseFeatContext> varlen_feats;
  std::unordered_map<string, CommonFeatContext *> feat_map; // libsvm格式数据的特征索引
  vector<pair<size_t, CommonFeatContext *>> feat_entries;                 // csv格式数据的特征索引
  vector<string> line_split_buff;   // csv格式数据的解析中间变量
  const size_t batch_size;
  size_t sample_idx;
  vector<Sample> batch_samples;


  std::unordered_map<FMParamUnit *, ParamNode> batch_params;
};
