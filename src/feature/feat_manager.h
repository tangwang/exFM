/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/dense_feat.h"
#include "feature/sparse_feat.h"
#include "feature/varlen_sparse_feat.h"
#ifdef _MSC_VER
#include <direct.h> // mdkir for windows
#define mkdir(dir, mode) mkdir(dir)
#else
#include <sys/stat.h> // mkdir for linux
#endif

class FeatManager {
  public:
  FeatManager() {}
  ~FeatManager() {}

  bool loadByFeatureConfig(string config_file_name);

  bool dumpModel();
  
  // TODO 改变feedSample的方式，最好x按列解析好，直接跟featCfgs对应起来，而不是把整行数据直接丢给各个feaCfg。 以下几个feas直接暴露给外部不太好。
  vector<DenseFeatConfig> dense_feas;
  vector<SparseFeatConfig> sparse_feas;
  vector<VarlenSparseFeatConfig> varlen_feas;

  // key: 如果有shared_embedding_name则用shared_embedding_name，否则用特征的名称. 用于共享embedding
  unordered_map<string, shared_ptr<ParamContainerInterface>> shared_param_container_map;

private:
  bool initModelParams(bool show_cfg = false);

};