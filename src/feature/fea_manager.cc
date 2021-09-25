/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "feature/fea_manager.h"


int FeaManager::loadByFeatureConfig(string config_file_name) {
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

  initModelParams(train_opt.verbose > 0);

  return 0;
}

void FeaManager::initModelParams(bool show_cfg) {
  for (auto &fea : dense_feas) {
    fea.init(param_containers);
    if (show_cfg) cout << fea << endl;
  }
  for (auto &fea : sparse_feas) {
    fea.init(param_containers);
    if (show_cfg) cout << fea << endl;
  }
  for (auto &fea : varlen_feas) {
    fea.init(param_containers);
    if (show_cfg) cout << fea << endl;
  }
}

int FeaManager::dumpModel(bool show_cfg) {
  // TODO 参数可能要加锁，保证内存序，以保证读取的参数是最新的
  // 调用dump_model时，所有的worker进程以及结束，只留下一个进程所以不存在内存序的问题导致读取的内存错误，但是
  // 能否确认缓存都已更新到内存？如果不能确认，这里需要全部加一次锁，或者考虑full
  // barrier
  int ret = 0;
  if (!train_opt.model_path.empty()) {
    if (0 != access(train_opt.model_path.c_str(), 0)) {
      int mkdir_ok = mkdir(train_opt.model_path.c_str(), 0755);
      if (mkdir_ok != 0) {
        std::cerr << "mkdir for save model faild !! << " << train_opt.model_path << std::endl;
        return -1;
      }
    }
    cout << "begin to dump model: " << endl;
    for (auto &fea : dense_feas) {
      if (ret != 0) break;
      ret = fea.dumpModel();
    }
    for (auto &fea : sparse_feas) {
      if (ret != 0) break;
      ret = fea.dumpModel();
    }
    for (auto &fea : varlen_feas) {
      if (ret != 0) break;
      ret = fea.dumpModel();
    }
    if (ret == 0) {
        cout << "dump model all finished" << endl;
    } else {
        cerr << "dump model faild" << endl;
    }
  }
  return ret;
}