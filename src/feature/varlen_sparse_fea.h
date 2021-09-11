/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/common_fea.h"
#include "feature/sparse_fea.h"


class VarlenSparseFeaConfig : public CommonFeaConfig {
 public:
  SparseFeaConfig sparse_cfg;
  string pooling_type;
  SeqPoolType pooling_type_id;
  int max_len;

  int initParams();

  void dump() const {
    cout << "------------------------------------- \n";
    cout << " VarlenSparseFeaConfig name <" << name << ">\n";
    cout << " max_len <" << max_len << ">\n";
    cout << " pooling_type <" << pooling_type << ">\n";
    cout << " max_id <" << sparse_cfg.max_id << ">\n";
    cout << " use_id_mapping <" << sparse_cfg.use_id_mapping << ">\n";
    cout << " use_hash <" << sparse_cfg.use_hash << ">\n";
    cout << " default_value <" << sparse_cfg.default_value << ">\n";
    cout << " id_mapping_dict_path <" << sparse_cfg.id_mapping_dict_path << ">\n";
    cout << " fea_id_mapping size <" << sparse_cfg.fea_id_mapping.size() << ">\n";
    cout << " vocab_size <" << sparse_cfg.vocab_size << ">\n";
  }

  VarlenSparseFeaConfig();
  ~VarlenSparseFeaConfig();
};

void to_json(json& j, const VarlenSparseFeaConfig& p);
void from_json(const json& j, VarlenSparseFeaConfig& p);

class VarlenSparseFeaContext : public CommonFeaContext {
 public:
  const VarlenSparseFeaConfig& cfg_;

  vector<feaid_t> orig_fea_ids;
  //以下两个vector每个元素一一对应。 TODO ， 后面可以去掉fea_ids，只保留fea_params
  vector<feaid_t> fea_ids;
  vector<FtrlParamUnit *> fea_params;

  void forward(vector<ParamContext> &forward_params);
  void backward();

  bool valid() const { return !fea_ids.empty(); }

  int feedSample(const char* line, vector<ParamContext> & forward_params, vector<ParamContext> & backward_params);

  VarlenSparseFeaContext(const VarlenSparseFeaConfig& cfg);
  ~VarlenSparseFeaContext();
};
