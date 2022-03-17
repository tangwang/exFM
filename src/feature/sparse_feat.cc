/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "feature/sparse_feat.h"
#include "solver/solver_factory.h"
#include "murmur_hash3/MurmurHash3.h"

SparseFeatConfig::SparseFeatConfig() {
  vocab_size = 0;
  max_id = 0;
  default_id = 0;
  max_feat_id_of_mapping_dict = 0;
}

SparseFeatConfig::~SparseFeatConfig() {}

feat_id_t SparseFeatConfig::featMapping(const char * orig_feat_id, size_t str_len) const {
  feat_id_t feat_id = default_id;
  if (likely(orig_feat_id[0] != 0)) {
    switch (mapping_type) {
      case mapping_by_dict_int32: {
        // int i_orig_feat_id = atoi(orig_feat_id);
        int i_orig_feat_id = int(std::round(atof(orig_feat_id))); // 兼容float
        feat_id = i32_feat_id_dict.get(i_orig_feat_id);
      } break;
      case mapping_by_dict_int64: {
        // long i_orig_feat_id = atol(orig_feat_id);
        long i_orig_feat_id = long(std::round(atof(orig_feat_id)));  // 兼容float
        feat_id = i64_feat_id_dict.get(i_orig_feat_id);
      } break;
      case mapping_by_dict_str: {
        feat_id = str_feat_id_dict.get(orig_feat_id);
      } break;
      case mapping_by_dynamic_dict_int32: {
        int i_orig_feat_id = atoi(orig_feat_id);
        // int i_orig_feat_id = int(std::round(atof(orig_feat_id)));  // 兼容float
        feat_id = getAndSetFeatID(i_orig_feat_id, i32_feat_id_dict);
      } break;
      case mapping_by_dynamic_dict_int64: {
        long i_orig_feat_id = atol(orig_feat_id);
        // long i_orig_feat_id = long(std::round(atof(orig_feat_id)));  // 兼容float
        feat_id = getAndSetFeatID(i_orig_feat_id, i64_feat_id_dict);
      } break;
      case mapping_by_dynamic_dict_str: {
        feat_id = getAndSetFeatID(string(orig_feat_id), str_feat_id_dict);
      } break;
      case mapping_by_hash_int32: {
        int i_orig_feat_id = atoi(orig_feat_id);
        // int i_orig_feat_id = int(std::round(atof(orig_feat_id))); // 兼容float
        feat_id = MurmurHash3_x86_32((void *)&i_orig_feat_id, sizeof(i_orig_feat_id), hash_seed);
        feat_id %= vocab_size;
      } break;
      case mapping_by_hash_int64: {
        long i_orig_feat_id = atol(orig_feat_id);
        // long i_orig_feat_id = long(std::round(atof(orig_feat_id))); // 兼容float
        feat_id = MurmurHash3_x86_32((void *)&i_orig_feat_id, sizeof(i_orig_feat_id), hash_seed);
        feat_id %= vocab_size;
      } break;
      case mapping_by_hash_str: {
        feat_id =
            MurmurHash3_x86_32(orig_feat_id, str_len, hash_seed);
        feat_id %= vocab_size;
      } break;
      default:
        int i_orig_feat_id = atoi(orig_feat_id);
        // int i_orig_feat_id = int(std::round(atof(orig_feat_id))); // 兼容float
        feat_id = (i_orig_feat_id < 0 || i_orig_feat_id > (int)max_id) ? unknown_id : (feat_id_t)i_orig_feat_id;
        break;
    }
  }
  return feat_id;
}

bool SparseFeatConfig::initParams(unordered_map<string, shared_ptr<ParamContainerInterface>> & shared_param_container_map) {

  bool ret = true;

  // 加载词典， 确定vocab_size，default_id, unknown_id
  switch (mapping_type) {
    case mapping_by_dict_int32:
      ret = loadFeatIdDict(mapping_dict_path, (i32_feat_id_dict));
      vocab_size = std::max(vocab_size, max_feat_id_of_mapping_dict + 2);
      break;

    case mapping_by_dict_int64:
      ret = loadFeatIdDict(mapping_dict_path, (i64_feat_id_dict));
      vocab_size = std::max(vocab_size, max_feat_id_of_mapping_dict + 2);
      break;

    case mapping_by_dict_str:
      ret = loadFeatIdDict(mapping_dict_path, (str_feat_id_dict));
      vocab_size = std::max(vocab_size, max_feat_id_of_mapping_dict + 2);
      break;

    case mapping_by_dynamic_dict_int32:
      if (!mapping_dict_path.empty()) {
        ret = loadFeatIdDict(mapping_dict_path, (i32_feat_id_dict));
      }
      mapping_dict_lock = std::make_shared<RW_Mutex_t>();
      break;

    case mapping_by_dynamic_dict_int64:
      if (!mapping_dict_path.empty()) {
        ret = loadFeatIdDict(mapping_dict_path, (i64_feat_id_dict));
      }
      mapping_dict_lock = std::make_shared<RW_Mutex_t>();
      break;

    case mapping_by_dynamic_dict_str:
      if (!mapping_dict_path.empty()) {
        ret = loadFeatIdDict(mapping_dict_path, (str_feat_id_dict));
      }
      mapping_dict_lock = std::make_shared<RW_Mutex_t>();
      break;

    case mapping_by_orig_id:
      vocab_size = max_id + 2;
      break;

    case mapping_by_hash_int32:
    case mapping_by_hash_int64:
    case mapping_by_hash_str:
      break;

    default:
      break;
  }
  if (!ret) return ret;

  vocab_size = std::max(vocab_size, max_feat_id_of_mapping_dict + 2);
  unknown_id = vocab_size - 1;
  i32_feat_id_dict.setNullValue(unknown_id);
  i64_feat_id_dict.setNullValue(unknown_id);
  str_feat_id_dict.setNullValue(unknown_id);

  // initail mutexes
  feat_id_t mutex_nums = vocab_size;
  if (mutex_nums > 10000) {
    mutex_nums = std::max(10000, (int)std::pow((float)mutex_nums, 0.8));
    mutex_nums = std::min(mutex_nums, (feat_id_t)1000000);
  }

  bool embedding_existed = false;
  if (!shared_embedding_name.empty()) {
    auto iter = shared_param_container_map.find(shared_embedding_name);
    if (iter != shared_param_container_map.end()) {
      param_container = iter->second;
      embedding_existed = true;
    }
  }

  if (!embedding_existed) {
    param_container = creatParamContainer(vocab_size, mutex_nums);
    if (!shared_embedding_name.empty()) shared_param_container_map[shared_embedding_name] = param_container;
    // 共享embedding的feature，param_container的创建者负责warmup，以保证只warm_start()一次
    ret = loadModel();
    if (!ret) return ret;
  }

  return ret;
}

void to_json(json &j, const SparseFeatConfig &p) {
  j = json{{"name", p.name},
           {"max_id", p.max_id},
           {"mapping_dict_name", p.mapping_dict_name},
           {"mapping_type", p.mapping_type},
           {"shared_embedding_name", p.shared_embedding_name}};
}

void from_json(const json &j, SparseFeatConfig &p) {
  if (!j.contains("name")) {
    throw "feature config err : no attr \"name\" in sparse feature.";
  }
  j.at("name").get_to(p.name);

  if (!j.contains("mapping_type")) {
    throw "feature config err : no attr \"mapping_type\" in sparse feature.";
    return;
  }
  if (!j.contains("value_type")) {
    throw "feature config err : no attr \"value_type\" in sparse feature.";
    return;
  }

  string str_mapping_type;
  string feat_value_type;
  j.at("mapping_type").get_to(str_mapping_type);
  j.at("value_type").get_to(feat_value_type);

  // mapping_type=="orig_id"时需填写max_id。将根据max_id确定特征ID总数。
  if (str_mapping_type == "orig_id") {
    if (!j.contains("max_id")) {
      throw "feature config err : no attr \"max_id\" in sparse feature.";
      return;
    }
    j.at("max_id").get_to(p.max_id);
  }
  // mapping_type=="hash"时需填写ids_num。将根据ids_num确定hash桶个数。
  if (str_mapping_type == "hash") {
    if (!j.contains("vocab_size")) {
      throw "feature config err : no attr \"vocab_size\" in sparse feature.";
      return;
    }
    j.at("vocab_size").get_to(p.vocab_size);
  }

  if (feat_value_type != "int32" &&
      feat_value_type != "int64" &&
      feat_value_type != "str") {
    throw "unknown mapping_type of feature " + p.name +
        ". only supoort int32/int64/str";
    return;
  }

  if (str_mapping_type == "dynamic_dict") {
    j.at("vocab_size").get_to(p.vocab_size);
   if (feat_value_type == "int32") {
      p.mapping_type = SparseFeatConfig::mapping_by_dynamic_dict_int32;
    } else if (feat_value_type == "int64") {
      p.mapping_type = SparseFeatConfig::mapping_by_dynamic_dict_int64;
    } else if (feat_value_type == "str") {
      p.mapping_type = SparseFeatConfig::mapping_by_dynamic_dict_str;
    }
  } else if (str_mapping_type == "dict") {
    if (feat_value_type == "int32") {
      p.mapping_type = SparseFeatConfig::mapping_by_dict_int32;
    } else if (feat_value_type == "int64") {
      p.mapping_type = SparseFeatConfig::mapping_by_dict_int64;
    } else if (feat_value_type == "str") {
      p.mapping_type = SparseFeatConfig::mapping_by_dict_str;
    }
  } else if (str_mapping_type == "hash") {
    if (feat_value_type == "int32") {
      p.mapping_type = SparseFeatConfig::mapping_by_hash_int32;
    } else if (feat_value_type == "int64") {
      p.mapping_type = SparseFeatConfig::mapping_by_hash_int64;
    } else if (feat_value_type == "str") {
      p.mapping_type = SparseFeatConfig::mapping_by_hash_str;
    }
  } else {
    p.mapping_type = SparseFeatConfig::mapping_by_orig_id;
  }

  if (j.contains("mapping_dict_name")) {
    j.at("mapping_dict_name").get_to(p.mapping_dict_name);
    if (!p.mapping_dict_name.empty()) {
      p.mapping_dict_path = train_opt.mapping_dict_path + p.mapping_dict_name;
    }
  }

  if (j.contains("shared_embedding_name"))   j.at("shared_embedding_name").get_to(p.shared_embedding_name);
}

SparseFeatContext::SparseFeatContext(const SparseFeatConfig &cfg) : cfg_(cfg) {
  feat_cfg = &cfg_;
}

SparseFeatContext::~SparseFeatContext() {}

int SparseFeatContext::feedSample(const char *feat_str, size_t feat_str_len, FmLayerNode & fm_node) {
  feat_id = cfg_.featMapping(feat_str, feat_str_len);

  if (!valid()) {
    fm_node.forward.clear();
    fm_node.backward_nodes.clear();
    return -1;  // TODO 这里可以要去掉。之前默认值是-1，现在默认值改成了0，采用默认值的ID
  }

  DEBUG_OUT << "feedSample " << cfg_.name << " feat_str " << feat_str << " feat_id " << feat_id << endl;

  FMParamUnit *feat_param = cfg_.param_container->get(feat_id);
  ParamMutex_t *param_mutex = cfg_.param_container->GetMutexByFeatID(feat_id);

  param_mutex->readLock();
  fm_node.forward = *feat_param;
  param_mutex->unlock();

  fm_node.backward_nodes.clear();
  fm_node.backward_nodes.emplace_back(feat_param, param_mutex, 1.0);

  return 0;
}

