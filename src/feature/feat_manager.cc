/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "feature/feat_manager.h"
#include <exception>


bool FeatManager::loadByFeatureConfig(string config_file_name) {
  bool ret = false;
  std::ifstream fin(config_file_name);

  json cfg_json;
  fin >> cfg_json;

  const char *parsing = NULL;
  try {
    parsing = "dense";
    dense_feas =
        cfg_json.at(train_opt.fea_type_dense).get<vector<DenseFeatConfig>>();

    parsing = "sparse";
    sparse_feas =
        cfg_json.at(train_opt.fea_type_sparse).get<vector<SparseFeatConfig>>();

    parsing = "varlen_sparse";
    varlen_feas = cfg_json.at(train_opt.fea_type_varlen_sparse)
                      .get<vector<VarlenSparseFeatConfig>>();
    
    ret = true;
  } catch (char *err_msg) {
    std::cerr << "exception occured while parse " << parsing
              << " features : " << err_msg << endl;
  } catch (std::exception &ex) {
    std::cerr << "exception occured while parse " << parsing
              << " features : " << ex.what() << endl;
  } catch (...) {
    std::cerr << "unknown exception occured while parse " << parsing
              << " features." << endl;
  }
  fin.close();

  if (ret) {
    ret = initModelParams(train_opt.verbose > 0);
  }

  return ret;
}

bool FeatManager::initModelParams(bool show_cfg) {
  for (auto &feat : dense_feas) {
    if (!feat.initParams(shared_param_container_map)) {
      cerr << " feature config " << feat.name << " init failed" << endl;
      return false;
    }
    cout << "feature config " << feat.name << " init ok" << endl;
    if (show_cfg) cout << feat << endl;
  }
  for (auto &feat : sparse_feas) {
    if (!feat.initParams(shared_param_container_map)) {
      cerr << " feature config " << feat.name << " init failed" << endl;
      return false;
    }
    cout << "feature config " << feat.name << " init ok" << endl;
    if (show_cfg) cout << feat << endl;
  }
  for (auto &feat : varlen_feas) {
    if (!feat.initParams(shared_param_container_map)) {
      cerr << " feature config " << feat.name << " init failed" << endl;
      return false;
    }
    cout << "feature config " << feat.name << " init ok" << endl;
    if (show_cfg) cout << feat << endl;
  }
  return true;
}

bool FeatManager::dumpModel() {
  // TODO 参数可能要加锁，保证内存序，以保证读取的参数是最新的
  // 调用dump_model时，所有的worker进程以及结束，只留下一个进程所以不存在内存序的问题导致读取的内存错误，但是
  // 能否确认缓存都已更新到内存？如果不能确认，这里需要全部加一次锁，或者考虑full
  // barrier
  bool ret = true;
  if (!train_opt.model_path.empty()) {
    if (0 != access(train_opt.model_path.c_str(), 0)) {
      int mkdir_ok = mkdir(train_opt.model_path.c_str(), 0755);
      if (mkdir_ok != 0) {
        std::cerr << "mkdir for save model faild !! << " << train_opt.model_path << std::endl;
        return false;
      }
    }
    cout << "begin to dump model: " << endl;
    for (auto &feat : dense_feas) {
      if (!ret) break;
      ret = feat.dumpModel();
    }
    for (auto &feat : sparse_feas) {
      if (!ret) break;
      ret = feat.dumpModel();
    }
    for (auto &feat : varlen_feas) {
      if (!ret) break;
      ret = feat.dumpModel();
    }
    if (ret) {
        cout << "dump model all finished" << endl;
    } else {
        cerr << "dump model faild" << endl;
    }
  }
  return ret;
}