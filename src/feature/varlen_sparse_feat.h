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

  bool initParams(map<string, shared_ptr<ParamContainerInterface>> & shared_param_container_map);

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

  vector<string> orig_fea_ids;

  vector<feaid_t> fea_ids;

  bool valid() const { return !fea_ids.empty(); }

  int feedSample(const char *line, FmLayerNode & fm_node);

  VarlenSparseFeatContext(const VarlenSparseFeatConfig& cfg);
  ~VarlenSparseFeatContext();
};
