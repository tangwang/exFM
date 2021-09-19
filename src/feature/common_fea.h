/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
// #include "solver/solver_factory.h"
#include "solver/parammeter_container.h"
#include "nlohmann/json.hpp"
#include "synchronize/mutex_adapter.h"
#include "utils/Hash.h"
#include "utils/List.h"
#include "utils/base.h"
#include "utils/utils.h"

using json = nlohmann::json;

enum SeqPoolType {
  SeqPoolTypeSUM = 0,
  SeqPoolTypeAVG = 1,
  SeqPoolTypeMAX = 2,
  SeqPoolTypeFlatern = 3,
  SeqPoolTypeGRU = 4,
};

class CommonFeaConfig {
 public:
  string name;
  string identifier;  // fea_seperator + name + kv_seperator
  int identifier_len;

  mutable shared_ptr<ParamContainerInterface> param_container;

  int warm_start() {
    int ret = 0;
    if (!train_opt.init_model_path.empty()) {
      ret = param_container->load(train_opt.init_model_path + "/" + name,
                                  train_opt.model_format);
    }
    return ret;
  }

  int dumpModel() {
    int ret = 0;
    if (!train_opt.model_path.empty()) {
      cout << "dump model for " << name << " ... ";
      if (param_container) {
        ret = param_container->dump(train_opt.model_path + "/" + name,
                                    train_opt.model_format);
        if (ret == 0) {
          cout << " ok " << endl;
        } else {
          cout << " faild " << endl;
        }
      } else {
        cout << " param_container is empty! " << endl;
      }
    }
    return ret;
  }

  virtual int initParams() = 0;

  void init() {
    identifier = train_opt.fea_seperator + name + train_opt.fea_kv_seperator;
    identifier_len = identifier.length();

    initParams();
  }

  int parseReal(const char *line, real_t &x, real_t default_v) const {
    x = default_v;
    const char *pos = strstr(line, identifier.c_str());
    if (pos != NULL) {
      x = atof(pos + identifier_len);
    }
    return 0;
  }

  int parseID(const char *line, feaid_t &feaid, feaid_t default_v) const {
    feaid = default_v;
    const char *pos = strstr(line, identifier.c_str());
    if (pos != NULL) {
      feaid = atol(pos + identifier_len);
    }
    return 0;
  }

  int parseFeaIdList(const char *line, vector<feaid_t> &feaid_list) const {
    feaid_list.clear();
    const char *pos = strstr(line, identifier.c_str());
    if (pos != NULL) {
      pos += identifier_len;

      utils::split_string(pos, train_opt.fea_multivalue_seperator,
                          train_opt.fea_seperator, feaid_list);
    }
    return 0;
  }
};

class CommonFeaContext {
 public:
  shared_ptr<ParamContainerInterface> forward_param_container;
  shared_ptr<ParamContainerInterface> backward_param_container;

  // forward_params中的参数指针只用于前向传递值，计算loss，所以为性能考虑，都从参数容器中取出来后放入线程本地内存（FeaContext中的本地参数容器），计算loss时不会对forward_params中的元素加锁
  virtual int feedSample(const char *line,
                          vector<ParamContext> &forward_params,
                          vector<ParamContext> &backward_params) = 0;
  virtual bool valid() const = 0;

  // TODO改造
  virtual void forward(vector<ParamContext> &forward_params) = 0;
  virtual void backward() = 0;

  CommonFeaContext();

  virtual ~CommonFeaContext() {}
};
