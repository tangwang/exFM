/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/common_feat.h"
#include "utils/dict.hpp"

using utils::Dict;

class SparseFeatConfig : public CommonFeatConfig {
 public:
  feat_id_t max_id;
  feat_id_t ids_num;

  // mapping_type == "orig_id" 时，vocab_size由max_id确定
  // mapping_type == "hash" 时，vocab_size由ids_num确定
  // mapping_type == "dict" 时，vocab_size由配置词典大小确定
  feat_id_t vocab_size;

  // mapping_type == "orig_id" 时，default_id设为max_id, unknown_id设为max_id+1
  // mapping_type == "hash" 时，default_id设为0, unknown_id设为1
  // mapping_type == "dict" 时，default_id设为0, unknown_id设为1
  feat_id_t default_id;
  feat_id_t unknown_id;
  
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
  static const size_t max_hash_buckets = 10000000;
  static const size_t min_hash_buckets = 200;

  feat_id_t featMapping(const char * orig_feat_id, size_t str_len) const;

  string shared_embedding_name;

  string mapping_dict_name;

  Dict<int, feat_id_t> i32_feat_id_dict;
  Dict<long, feat_id_t> i64_feat_id_dict;
  Dict<string, feat_id_t> str_feat_id_dict;

  bool initParams(unordered_map<string, shared_ptr<ParamContainerInterface>> & shared_param_container_map);

  friend ostream & operator << (ostream &out, const SparseFeatConfig & cfg) {
    out << " SparseFeatConfig name " << cfg.name << ">" << endl;
    out << " mapping_dict_name <" << cfg.mapping_dict_name << ">" << endl;
    out << " i32_feat_id_dict size <" << cfg.i32_feat_id_dict.size() << ">" << endl;
    out << " i64_feat_id_dict size <" << cfg.i64_feat_id_dict.size() << ">" << endl;
    out << " str_feat_id_dict size <" << cfg.str_feat_id_dict.size() << ">" << endl;
    out << " max_id <" << cfg.max_id << ">" << endl;
    out << " ids_num <" << cfg.ids_num << ">" << endl;
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

  feat_id_t feat_id;

  bool valid() const { return feat_id != cfg_.default_id; }

  int feedSample(const char *feat_str, size_t feat_str_len, FmLayerNode & fm_node);

  SparseFeatContext(const SparseFeatConfig& cfg);
  ~SparseFeatContext();
};
