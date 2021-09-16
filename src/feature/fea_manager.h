/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/dense_fea.h"
#include "feature/sparse_fea.h"
#include "feature/varlen_sparse_fea.h"

class FeaManager {
 public:
  vector<DenseFeaConfig> dense_feas;
  vector<SparseFeaConfig> sparse_feas;
  vector<VarlenSparseFeaConfig> varlen_feas;

  void initModelParams(bool show_cfg = false) {
    for (auto &fea : dense_feas) {
      fea.init();
      if (show_cfg) fea.dump();
    }
    for (auto &fea : sparse_feas) {
      fea.init();
      if (show_cfg) fea.dump();
    }
    for (auto &fea : varlen_feas) {
      fea.init();
      if (show_cfg) fea.dump();
    }
  }

 public:
  // configure file
  int parse_fea_config(string config_file_name);
  int parse_one_fea(const char *json_string);

  FeaManager();
  ~FeaManager();
};
