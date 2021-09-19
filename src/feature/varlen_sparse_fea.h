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

  friend ostream & operator << (ostream &out, const VarlenSparseFeaConfig & cfg) {
    out << "------------------------------------- " << endl;
    out << " VarlenSparseFeaConfig name <" << cfg.name << ">" << endl;
    out << " max_len <" << cfg.max_len << ">" << endl;
    out << " pooling_type <" << cfg.pooling_type << ">" << endl;
    out << " max_id <" << cfg.sparse_cfg.max_id << ">" << endl;
    out << " use_id_mapping <" << cfg.sparse_cfg.use_id_mapping << ">" << endl;
    out << " use_hash <" << cfg.sparse_cfg.use_hash << ">" << endl;
    out << " default_value <" << cfg.sparse_cfg.default_value << ">" << endl;
    out << " id_mapping_dict_path <" << cfg.sparse_cfg.id_mapping_dict_path << ">" << endl;
    out << " fea_id_mapping size <" << cfg.sparse_cfg.fea_id_mapping.size() << ">" << endl;
    out << " vocab_size <" << cfg.sparse_cfg.vocab_size << ">" << endl;
    return out;
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
  vector<FMParamUnit *> fea_params;

  void forward(vector<ParamContext> &forward_params);
  void backward();

  bool valid() const { return !fea_ids.empty(); }

  int feedSample(const char* line, vector<ParamContext> & forward_params, vector<ParamContext> & backward_params);

  VarlenSparseFeaContext(const VarlenSparseFeaConfig& cfg);
  ~VarlenSparseFeaContext();
};
