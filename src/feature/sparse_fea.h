/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/common_fea.h"
#include "utils/dict.hpp"

using utils::Dict;

class SparseFeaConfig : public CommonFeaConfig {
 public:
  int vocab_size;
  int max_id;
  int default_value;
  int use_id_mapping;
  bool use_hash;
  mutable shared_ptr<ParamContainer> param_container;
  mutable vector<Mutex_t> mutexes;
  int mutex_nums;
  Mutex_t* GetMutexByFeaID(feaid_t id) const {
    return &mutexes[id % mutex_nums];
  }

  string id_mapping_dict_path;

  // TODO 是否需要支持字符串类型特征，兼容性更好，但是影响性能
  // Dict<std::string, feaid_t> fea_id_mapping;
  Dict<feaid_t, feaid_t> fea_id_mapping;

  int initParams();

  void dump() const {
    cout << "------------------------------------- \n";
    cout << " SparseFeaConfig name " << name << ">\n";
    cout << " use_hash <" << use_hash << ">\n";
    cout << " max_id <" << max_id << ">\n";
    cout << " use_id_mapping <" << use_id_mapping << ">\n";
    cout << " default_value <" << default_value << ">\n";
    cout << " id_mapping_dict_path <" << id_mapping_dict_path << ">\n";
    cout << " fea_id_mapping size <" << fea_id_mapping.size() << ">\n";
    cout << " vocab_size <" << vocab_size << ">\n";
    cout << " use_hash <" << use_hash << ">\n";
    cout << " default_value <" << default_value << ">\n";
  }

  SparseFeaConfig();
  ~SparseFeaConfig();
};

void to_json(json& j, const SparseFeaConfig& p);
void from_json(const json& j, SparseFeaConfig& p);

class SparseFeaContext : public CommonFeaContext {
 public:
  const SparseFeaConfig& cfg_;

  feaid_t orig_fea_id;
  feaid_t fea_id;

  bool valid() const { return fea_id != cfg_.default_value; }

  void forward(vector<ParamContext>& forward_params);
  void backward();

  int feedRawData(const char* line, vector<ParamContext>& forward_params,
                  vector<ParamContext>& backward_params);

  SparseFeaContext(const SparseFeaConfig& cfg);
  ~SparseFeaContext();
};
