/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/common_feat.h"
#include "feature/sparse_feat.h"


class VarlenSparseFeatConfig : public CommonFeatConfig {
 public:
  SparseFeatConfig sparse_cfg;

  enum SeqPoolType {
    SeqPoolTypeSUM = 0,
    SeqPoolTypeAVG = 1,
    SeqPoolTypeMAX = 2,
    SeqPoolTypeFlatern = 3,
    SeqPoolTypeGRU = 4,
  };

  SeqPoolType pooling_type_id;
  size_t max_len;

  bool initParams(unordered_map<string, shared_ptr<ParamContainerInterface>> & shared_param_container_map);

  friend ostream & operator << (ostream &out, const VarlenSparseFeatConfig & cfg) {
    out << " VarlenSparseFeatConfig name <" << cfg.name << ">" << endl;
    out << " max_len <" << cfg.max_len << ">" << endl;
    out << " pooling_type_id <" << cfg.pooling_type_id << ">" << endl;
    out << cfg.sparse_cfg;
    return out;
  }

  VarlenSparseFeatConfig();
  ~VarlenSparseFeatConfig();
};

void to_json(json& j, const VarlenSparseFeatConfig& p);
void from_json(const json& j, VarlenSparseFeatConfig& p);

class VarlenSparseFeatContext : public CommonFeatContext {
 public:
  const VarlenSparseFeatConfig& cfg_;

  vector<feat_id_t>  feat_ids;

  bool valid() const { return !feat_ids.empty(); }

  int feedSample(const char *feat_str, size_t feat_str_len, FmLayerNode & fm_node);

  char feat_id_buff[128]; // temp variable for parse single feat_id

  VarlenSparseFeatContext(const VarlenSparseFeatConfig& cfg);
  ~VarlenSparseFeatContext();
};
