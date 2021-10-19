/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/common_feat.h"
#include "utils/dict.hpp"
#include "synchronize/mutex_adapter.h"

using utils::Dict;

class SparseFeatConfig : public CommonFeatConfig {
 public:
  feat_id_t max_id;

  // mapping_type == "orig_id" 时，vocab_size由max_id确定
  // mapping_type == "hash" 时，vocab_size从配置文件读取
  // mapping_type == "dict" 时，vocab_size由配置词典大小确定
  feat_id_t vocab_size;
  // default_id 约定永远为0
  // unknown_id 约定永远为vocab_size-1
  feat_id_t default_id;
  feat_id_t unknown_id;
  
  enum MappingType {
    mapping_by_orig_id,
    mapping_by_dict_int32,
    mapping_by_dict_int64,
    mapping_by_dict_str,
    mapping_by_dynamic_dict_int32,
    mapping_by_dynamic_dict_int64,
    mapping_by_dynamic_dict_str,
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
  string mapping_dict_path;

  // feat value to id mapping dicts. 
  mutable Dict<int, feat_id_t> i32_feat_id_dict;
  mutable Dict<long, feat_id_t> i64_feat_id_dict;
  mutable Dict<string, feat_id_t> str_feat_id_dict;
  mutable feat_id_t max_feat_id_of_mapping_dict;
  mutable shared_ptr<RW_Mutex_t> mapping_dict_lock;

  template <typename DictKeyType>
  bool loadFeatIdDict(const string& mapping_dict_path,
                      Dict<DictKeyType, feat_id_t>& feat_value2id_dict) const {
    bool ret = false;
    feat_value2id_dict.setNullValue(unknown_id);
    if (feat_value2id_dict.create(mapping_dict_path,
                                  train_opt.feat_id_dict_seperator)) {
      std::cout << "load dict <" << mapping_dict_path << "> ok, size <"
                << feat_value2id_dict.size() << ">" << std::endl;
      for (auto iter = feat_value2id_dict.begin();
           iter != feat_value2id_dict.end(); iter++) {
        if (max_feat_id_of_mapping_dict < iter->second)
          max_feat_id_of_mapping_dict = iter->second;
      }
      ret = true;
    } else {
      std::cerr << "load dict <" << mapping_dict_path << "> failed!"
                << std::endl;
    }
    return ret;
  }

  template <typename DictKeyType>
  bool dumpFeatIdDict(const string& output_dict_path,
                      const Dict<DictKeyType, feat_id_t>& feat_value2id_dict) const {
    ofstream ofs(output_dict_path);
    if (!ofs) {
      cerr << "cannot open " << output_dict_path << " for write " << endl;
      return false;
    }

    vector<pair<DictKeyType, feat_id_t>> dict_items;

    mapping_dict_lock->readLock();
    for (auto iter = feat_value2id_dict.begin();
         iter != feat_value2id_dict.end(); iter++) {
      dict_items.push_back(*iter);
    }
    mapping_dict_lock->unlock();

    sort(dict_items.begin(), dict_items.end(),
         utils::judgeByPairSecond<DictKeyType, feat_id_t>);

    for (const auto& v : dict_items) {
      ofs << v.first << train_opt.feat_id_dict_seperator << v.second
          << std::endl;
    }

    return true;
  }

  bool dumpFeatIdDict(const string & path) const {

    string feat_id_dict_path = path + "/" + name;

    switch (mapping_type) {
      case mapping_by_dynamic_dict_int32: {
        return dumpFeatIdDict(feat_id_dict_path, i32_feat_id_dict);
      } break;
      case mapping_by_dynamic_dict_int64: {
        return dumpFeatIdDict(feat_id_dict_path, i64_feat_id_dict);
      } break;
      case mapping_by_dynamic_dict_str: {
        return dumpFeatIdDict(feat_id_dict_path, str_feat_id_dict);
      } break;
      default:
        break;
    }
    return true;
  }

  template <typename DictKeyType>
  feat_id_t getAndSetFeatID(const DictKeyType& orig_feat_value,
                            Dict<DictKeyType, feat_id_t>& feat_value2id_dict) const {

    mapping_dict_lock->readLock();
    feat_id_t temp_max_feat_id_of_mapping_dict = max_feat_id_of_mapping_dict;
    feat_id_t feat_id = feat_value2id_dict.get(orig_feat_value);
    bool is_new_feat_id = (feat_id == unknown_id &&
                           temp_max_feat_id_of_mapping_dict < unknown_id - 1);
    mapping_dict_lock->unlock();
    
    if (is_new_feat_id) {
      mapping_dict_lock->writeLock();
      // need get again when dict changed (check dict size change or max_feat_id change)
      if (max_feat_id_of_mapping_dict != temp_max_feat_id_of_mapping_dict) {
        feat_id = feat_value2id_dict.get(orig_feat_value);
      }
      if (feat_id == unknown_id &&
          max_feat_id_of_mapping_dict < unknown_id - 1) {
        feat_value2id_dict.set(orig_feat_value, ++max_feat_id_of_mapping_dict);
        feat_id = max_feat_id_of_mapping_dict;
      }
      mapping_dict_lock->unlock();
    }
    return feat_id;
  }

  bool initParams(unordered_map<string, shared_ptr<ParamContainerInterface>> & shared_param_container_map);

  friend ostream & operator << (ostream &out, const SparseFeatConfig & cfg) {
    out << " SparseFeatConfig name " << cfg.name << ">" << endl;
    out << " mapping_dict_name <" << cfg.mapping_dict_name << ">" << endl;
    out << " i32_feat_id_dict size <" << cfg.i32_feat_id_dict.size() << ">" << endl;
    out << " i64_feat_id_dict size <" << cfg.i64_feat_id_dict.size() << ">" << endl;
    out << " str_feat_id_dict size <" << cfg.str_feat_id_dict.size() << ">" << endl;
    out << " max_id <" << cfg.max_id << ">" << endl;
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
