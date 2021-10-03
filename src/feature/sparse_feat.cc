/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "feature/sparse_feat.h"
#include "solver/solver_factory.h"
#include "murmur_hash3/MurmurHash3.h"

SparseFeatConfig::SparseFeatConfig() {
  default_id = 0;
  unknown_id = 1;
}

SparseFeatConfig::~SparseFeatConfig() {}

feaid_t SparseFeatConfig::featMapping(const string& orig_fea_id) const {
  feaid_t feat_id = default_id;
  if (!orig_fea_id.empty()) {
    switch (mapping_type) {
      case mapping_by_dict_int32: {
        int i_orig_fea_id = atoi(orig_fea_id.c_str());
        feat_id = i32_feat_id_dict.get(i_orig_fea_id);
      } break;
      case mapping_by_dict_int64: {
        long i_orig_fea_id = atol(orig_fea_id.c_str());
        feat_id = i64_feat_id_dict.get(i_orig_fea_id);
      } break;
      case mapping_by_dict_str: {
        feat_id = str_feat_id_dict.get(orig_fea_id);
      } break;
      case mapping_by_hash_int32: {
        int i_orig_fea_id = atoi(orig_fea_id.c_str());
        feat_id = MurmurHash3_x86_32((void *)&i_orig_fea_id, sizeof(i_orig_fea_id), hash_seed);
        feat_id %= vocab_size;
      } break;
      case mapping_by_hash_int64: {
        long i_orig_fea_id = atol(orig_fea_id.c_str());
        feat_id = MurmurHash3_x86_32((void *)&i_orig_fea_id, sizeof(i_orig_fea_id), hash_seed);
        feat_id %= vocab_size;
      } break;
      case mapping_by_hash_str: {
        feat_id =
            MurmurHash3_x86_32(orig_fea_id.c_str(), orig_fea_id.size(), hash_seed);
        feat_id %= vocab_size;
      } break;
      default:
        feat_id = atoi(orig_fea_id.c_str());
        if (feat_id > max_id) feat_id = unknown_id;
        break;
    }
  }
  return feat_id;
}

bool SparseFeatConfig::initParams(map<string, shared_ptr<ParamContainerInterface>> & shared_param_container_map) {

  bool ret = true;

  // 加载词典， 确定vocab_size，default_id, unknown_id
#define LOAD_FEAT_ID_DICT(dict)                                             \
  do {                                                                      \
    dict.setNullValue(unknown_id);                                          \
    if (dict.create(mapping_dict_name, train_opt.feat_id_dict_seperator)) { \
      std::cout << "load dict <" << mapping_dict_name << "> ok, size <"     \
                << dict.size() << ">" << std::endl;                         \
    } else {                                                                \
      ret = false;                                                          \
      std::cerr << "load dict <" << mapping_dict_name << "> failed!!!"      \
                << std::endl;                                               \
    }                                                                       \
  } while (0)

  switch (mapping_type) {
    case mapping_by_dict_int32:
      LOAD_FEAT_ID_DICT(i32_feat_id_dict);
      vocab_size = i32_feat_id_dict.size() + 2;
      break;

    case mapping_by_dict_int64:
      LOAD_FEAT_ID_DICT(i64_feat_id_dict);
      vocab_size = i64_feat_id_dict.size() + 2;
      break;

    case mapping_by_dict_str:
      LOAD_FEAT_ID_DICT(str_feat_id_dict);
      vocab_size = str_feat_id_dict.size() + 2;
      break;

    case mapping_by_orig_id:
      vocab_size = max_id + 2;
      default_id = max_id;
      unknown_id = max_id + 1;
      break;

    case mapping_by_hash_int32:
    case mapping_by_hash_int64:
    case mapping_by_hash_str:
      vocab_size = ids_num + 10000;
      default_id = 0;
      unknown_id = 1;
      break;

    default:
      break;
  }
  if (!ret) return ret;

  // initail mutexes
  feaid_t mutex_nums = vocab_size;
  if (mutex_nums > 10000) {
    mutex_nums = std::max(10000, (int)std::pow((float)mutex_nums, 0.8));
    mutex_nums = std::min(mutex_nums, (feaid_t)1000000);
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
    loadModel();
  }

  return ret;
}

  feaid_t max_id;
  feaid_t ids_num;

void to_json(json &j, const SparseFeatConfig &p) {
  j = json{{"name", p.name},
           {"max_id", p.max_id},
           {"ids_num", p.ids_num},
           {"mapping_dict_name", p.mapping_dict_name},
           {"mapping_type", p.mapping_type},
           {"shared_embedding_name", p.shared_embedding_name},
           {"default_id", p.default_id}};
}

void from_json(const json &j, SparseFeatConfig &p) {
  if (j.find("name") == j.end()) {
    throw "feature config err : no attr \"name\" in dense feature.";
  }
  j.at("name").get_to(p.name);

  if (j.find("mapping_type") == j.end()) {
    throw "feature config err : no attr \"mapping_type\" in dense feature.";
    return;
  }
  if (j.find("value_type") == j.end()) {
    throw "feature config err : no attr \"value_type\" in dense feature.";
    return;
  }

  string str_mapping_type;
  string feat_value_type;
  j.at("mapping_type").get_to(str_mapping_type);
  j.at("value_type").get_to(feat_value_type);

  // mapping_type=="orig_id"时需填写max_id。将根据max_id确定特征ID总数。
  if (str_mapping_type == "orig_id") {
    if (j.find("max_id") == j.end()) {
      throw "feature config err : no attr \"max_id\" in dense feature.";
      return;
    }
    j.at("max_id").get_to(p.max_id);
  }
  // mapping_type=="hash"时需填写ids_num。将根据ids_num确定hash桶个数。
  if (str_mapping_type == "hash") {
    if (j.find("ids_num") == j.end()) {
      throw "feature config err : no attr \"ids_num\" in dense feature.";
      return;
    }
    j.at("ids_num").get_to(p.ids_num);
  }

  if (str_mapping_type == "dict") {
    if (feat_value_type == "int32") {
      p.mapping_type = SparseFeatConfig::mapping_by_dict_int32;
    } else if (feat_value_type == "int64") {
      p.mapping_type = SparseFeatConfig::mapping_by_dict_int64;
    } else if (feat_value_type == "str") {
      p.mapping_type = SparseFeatConfig::mapping_by_dict_str;
    } else {
      throw "unknown mapping_type of feature " + p.name +
          ". only supoort int32/int64/str";
      return;
    }
  } else if (str_mapping_type == "hash") {
    if (feat_value_type == "int32") {
      p.mapping_type = SparseFeatConfig::mapping_by_hash_int32;
    } else if (feat_value_type == "int64") {
      p.mapping_type = SparseFeatConfig::mapping_by_hash_int64;
    } else if (feat_value_type == "str") {
      p.mapping_type = SparseFeatConfig::mapping_by_hash_str;
    } else {
      throw "unknown mapping_type of feature " + p.name +
          ". only supoort int32/int64/str";
      return;
    }
  } else {
    p.mapping_type = SparseFeatConfig::mapping_by_orig_id;
  }

  if (j.find("default_id") != j.end())                 j.at("default_id").get_to(p.default_id);
  if (j.find("unknown_id") != j.end())                 j.at("unknown_id").get_to(p.unknown_id);
  if (j.find("mapping_dict_name") != j.end())       j.at("mapping_dict_name").get_to(p.mapping_dict_name);
  if (j.find("shared_embedding_name") != j.end())      j.at("shared_embedding_name").get_to(p.shared_embedding_name);
}

SparseFeatContext::SparseFeatContext(const SparseFeatConfig &cfg) : cfg_(cfg) {}

SparseFeatContext::~SparseFeatContext() {}

void SparseFeatContext::forward(vector<ParamContext> &forward_params) {}

int SparseFeatContext::feedSample(const char *line,
                                  vector<ParamContext> &forward_params,
                                  vector<ParamContext> &backward_params) {
  cfg_.parseStr(line, orig_fea_id);
  feat_id = cfg_.featMapping(orig_fea_id);

  if (!valid()) {
    return -1; // TODO 0929 这里要去掉。 之前默认值是-1，现在默认值改成了0，采用默认值的ID
  }

  DEBUG_OUT << "feedSample " << cfg_.name << " orig_fea_id " << orig_fea_id << " feat_id " << feat_id << endl;

  FMParamUnit *fea_param = cfg_.param_container->get(feat_id);
  Mutex_t *param_mutex = cfg_.param_container->GetMutexByFeaID(feat_id);

  FMParamUnit *forward_param = forward_param_container->get();
  param_mutex->lock();
  cfg_.param_container->cpParam(forward_param, fea_param);
  param_mutex->unlock();

  forward_params.push_back(ParamContext((ParamContainerInterface*)cfg_.param_container.get(), forward_param, NULL, 1.0));

  real_t grad_from_forward2backward = 1.0;
  backward_params.push_back(ParamContext((ParamContainerInterface*)cfg_.param_container.get(), fea_param, param_mutex, 1.0, (int)forward_params.size()-1, grad_from_forward2backward));

  return 0;
}

void SparseFeatContext::backward() {
  // FMParamUnit *p = backward_param_container->get();

  // FMParamUnit *fea_param = cfg_.param_container->get(feat_id);
  // cfg_.sparse_cfg.param_container->addWeightsTo(p, fea_param);
}
