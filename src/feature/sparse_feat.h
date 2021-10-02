/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/common_feat.h"
#include "utils/dict.hpp"

using utils::Dict;

class SparseFeatConfig : public CommonFeatConfig {
 public:
  feaid_t vocab_size;
  feaid_t default_id;
  feaid_t unknown_id;
  
  enum MappingType {
    mapping_by_orig_id,
    mapping_by_dict_int32,
    mapping_by_dict_int64,
    mapping_by_dict_str,
    mapping_by_hash_int32,
    mapping_by_hash_int64,
    mapping_by_hash_str
  } mapping_type;

  static const uint32_t hash_seed = 0;

  feaid_t featMapping(const string& orig_fea_id) const;

  string shared_embedding_name;

  string mapping_dict_name;

  Dict<int, feaid_t> i32_feat_id_dict;
  Dict<long, feaid_t> i64_feat_id_dict;
  Dict<string, feaid_t> str_feat_id_dict;

  bool initParams(map<string, shared_ptr<ParamContainerInterface>> & shared_param_container_map);

  friend ostream & operator << (ostream &out, const SparseFeatConfig & cfg) {
    out << " SparseFeatConfig name " << cfg.name << ">" << endl;
    out << " mapping_dict_name <" << cfg.mapping_dict_name << ">" << endl;
    out << " i32_feat_id_dict size <" << cfg.i32_feat_id_dict.size() << ">" << endl;
    out << " i64_feat_id_dict size <" << cfg.i64_feat_id_dict.size() << ">" << endl;
    out << " str_feat_id_dict size <" << cfg.str_feat_id_dict.size() << ">" << endl;
    out << " vocab_size <" << cfg.vocab_size << ">" << endl;
    out << " default_id <" << cfg.default_id << ">" << endl;
    out << " unknown_id <" << cfg.unknown_id << ">" << endl;
    out << " shared_embedding_name <" << cfg.shared_embedding_name << ">" << endl;
    return out;
  }

  SparseFeatConfig();
  ~SparseFeatConfig();
};

void to_json(json& j, const SparseFeatConfig& p);
void from_json(const json& j, SparseFeatConfig& p);

class SparseFeatContext : public CommonFeatContext {
 public:
  const SparseFeatConfig& cfg_;

  string orig_fea_id;
  feaid_t feat_id;

  bool valid() const { return feat_id != cfg_.default_id; }

  void forward(vector<ParamContext>& forward_params);
  void backward();

  int feedSample(const char* line, vector<ParamContext>& forward_params,
                  vector<ParamContext>& backward_params);

  SparseFeatContext(const SparseFeatConfig& cfg);
  ~SparseFeatContext();
};
