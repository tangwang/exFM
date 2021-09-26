/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/dense_fea.h"
#include "feature/sparse_fea.h"
#include "feature/varlen_sparse_fea.h"
#if WINDOWS_VER_
#include <direct.h> // mdkir for windows
#define mkdir(dir, mode) mkdir(dir)
#else
#include <sys/stat.h> // mkdir for linux
#endif

class FeaManager {
  public:
  FeaManager() {}
  ~FeaManager() {}

   int loadByFeatureConfig(string config_file_name);

  int dumpModel(bool show_cfg = false);
  
  // TODO 改变feedSample的方式，最好x按列解析好，直接跟featCfgs对应起来，而不是把整行数据直接丢给各个feaCfg。 以下几个feas直接暴露给外部不太好。
  vector<DenseFeaConfig> dense_feas;
  vector<SparseFeaConfig> sparse_feas;
  vector<VarlenSparseFeaConfig> varlen_feas;

  // key: 如果有shared_embedding_name则用shared_embedding_name，否则用特征的名称. 用于共享embedding
  map<string, shared_ptr<ParamContainerInterface>> shared_param_container_map;

private:
  void initModelParams(bool show_cfg = false);

};
