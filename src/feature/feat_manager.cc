/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "feature/feat_manager.h"
#include <exception>


bool FeatManager::loadByFeatureConfig(string config_path) {
  bool ret = false;
  std::ifstream fin(config_path);

  json cfg_json;
  fin >> cfg_json;

  const char *parsing = NULL;
  try {
    parsing = "dense";
    if (cfg_json.find(train_opt.feat_type_dense) != cfg_json.end()) {
      dense_feat_cfgs =
        cfg_json.at(train_opt.feat_type_dense).get<vector<DenseFeatConfig>>();
    }

    if (cfg_json.find(train_opt.feat_type_sparse) != cfg_json.end()) {
      sparse_feat_cfgs =
        cfg_json.at(train_opt.feat_type_sparse).get<vector<SparseFeatConfig>>();
    }

    if (cfg_json.find(train_opt.feat_type_varlen_sparse) != cfg_json.end()) {
      varlen_feat_cfgs = cfg_json.at(train_opt.feat_type_varlen_sparse)
                      .get<vector<VarlenSparseFeatConfig>>();
    }
    ret = true;
  } catch (const char * err_msg) {
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
  for (auto &feat : dense_feat_cfgs) {
    if (!feat.initParams(shared_param_container_map)) {
      cerr << " feature config " << feat.name << " init failed" << endl;
      return false;
    }
    cout << "feature config " << feat.name << " init ok" << endl;
    if (show_cfg) cout << feat << endl;
  }
  for (auto &feat : sparse_feat_cfgs) {
    if (!feat.initParams(shared_param_container_map)) {
      cerr << " feature config " << feat.name << " init failed" << endl;
      return false;
    }
    cout << "feature config " << feat.name << " init ok" << endl;
    if (show_cfg) cout << feat << endl;
  }
  for (auto &feat : varlen_feat_cfgs) {
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
  bool ret = true;
  if (!train_opt.model_path.empty()) {
    if (0 != access(train_opt.model_path.c_str(), 0)) {
      int mkdir_ok = mkdir(train_opt.model_path.c_str(), 0755);
      if (mkdir_ok != 0) {
        std::cerr << "mkdir faild: " << train_opt.model_path << std::endl;
        return false;
      }
    }

    string feat_id_dict_path = train_opt.model_path + "/feat_id_mapping";
    if (0 != access(feat_id_dict_path.c_str(), 0)) {
      int mkdir_ok = mkdir(feat_id_dict_path.c_str(), 0755);
      if (mkdir_ok != 0) {
        std::cerr << "mkdir faild: " << feat_id_dict_path << std::endl;
        return false;
      }
    }

    cout << "begin to dump model: " << endl;
    for (auto &feat : dense_feat_cfgs) {
      if (!ret) break;
      ret = feat.dumpModel();
    }
    for (auto &feat : sparse_feat_cfgs) {
      if (!ret) break;
      ret = feat.dumpModel() && feat.dumpFeatIdDict(feat_id_dict_path);
    }
    for (auto &feat : varlen_feat_cfgs) {
      if (!ret) break;
      ret = feat.dumpModel() && feat.dumpFeatIdDict(feat_id_dict_path);
    }
    if (ret) {
        cout << "dump model all finished" << endl;
    } else {
        cerr << "dump model faild" << endl;
    }
  }
  return ret;
}