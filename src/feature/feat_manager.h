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

  bool loadByFeatureConfig(string config_path);

  bool dumpModel();
  
  vector<DenseFeatConfig> dense_feat_cfgs;
  vector<SparseFeatConfig> sparse_feat_cfgs;
  vector<VarlenSparseFeatConfig> varlen_feat_cfgs;

  // key: 如果有shared_embedding_name则用shared_embedding_name，否则用特征的名称. 用于共享embedding
  unordered_map<string, shared_ptr<ParamContainerInterface>> shared_param_container_map;

private:
  bool initModelParams(bool show_cfg = false);

};
