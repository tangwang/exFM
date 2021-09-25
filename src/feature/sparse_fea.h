/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/common_fea.h"
#include "utils/dict.hpp"

using utils::Dict;

class SparseFeaConfig : public CommonFeaConfig {
 public:
  feaid_t vocab_size;
  feaid_t max_id;
  feaid_t default_value;
  int use_id_mapping;
  bool use_hash;
  string shared_embedding_name;

  string id_mapping_dict_path;

  // TODO 是否需要支持字符串类型特征，兼容性更好，但是影响性能
  // Dict<std::string, feaid_t> fea_id_mapping;
  Dict<feaid_t, feaid_t> fea_id_mapping;

  int initParams(map<string, shared_ptr<ParamContainerInterface>> & param_containers);

  friend ostream & operator << (ostream &out, const SparseFeaConfig & cfg) {
    out << "------------------------------------- " << endl;
    out << " SparseFeaConfig name " << cfg.name << ">" << endl;
    out << " use_hash <" << cfg.use_hash << ">" << endl;
    out << " max_id <" << cfg.max_id << ">" << endl;
    out << " use_id_mapping <" << cfg.use_id_mapping << ">" << endl;
    out << " default_value <" << cfg.default_value << ">" << endl;
    out << " id_mapping_dict_path <" << cfg.id_mapping_dict_path << ">" << endl;
    out << " fea_id_mapping size <" << cfg.fea_id_mapping.size() << ">" << endl;
    out << " vocab_size <" << cfg.vocab_size << ">" << endl;
    out << " use_hash <" << cfg.use_hash << ">" << endl;
    out << " default_value <" << cfg.default_value << ">" << endl;
    out << " shared_embedding_name <" << cfg.shared_embedding_name << ">" << endl;
    return out;
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

  int feedSample(const char* line, vector<ParamContext>& forward_params,
                  vector<ParamContext>& backward_params);

  SparseFeaContext(const SparseFeaConfig& cfg);
  ~SparseFeaContext();
};
