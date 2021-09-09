/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "feature/fea_manager.h"


FeaManager::FeaManager() {}
FeaManager::~FeaManager() {}

int FeaManager::parse_fea_config(const char* config_file_name) {
  std::ifstream fin(config_file_name);

  json cfg_json;
  fin >> cfg_json;

  dense_feas = cfg_json.at(train_opt.fea_type_dense)
                   .get<vector<DenseFeaConfig>>();
  sparse_feas = cfg_json.at(train_opt.fea_type_sparse)
                    .get<vector<SparseFeaConfig>>();
  varlen_feas = cfg_json.at(train_opt.fea_type_varlen_sparse)
                    .get<vector<VarlenSparseFeaConfig>>();
  fin.close();
  return 0;
}
