/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "ftrl/ftrl_param.h"
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

struct ParamContext {
  ParamContext(FtrlParamUnit *_param = NULL, Mutex_t *_mutex = NULL,
               real_t _x = 0.0)
      : param(_param), mutex(_mutex), x(_x) {}
  FtrlParamUnit *param;
  Mutex_t *mutex;
  real_t x;
};

class CommonFeaConfig {
 public:
  string name;
  string identifier;  // fea_seperator + name + kv_seperator
  int identifier_len;

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
  shared_ptr<FtrlParamContainer> forward_param_container;
  shared_ptr<FtrlParamContainer> backward_param_container;
  shared_ptr<FtrlParamContainer> local_buff_container;

  virtual int feedSample(const char *line,
                          vector<ParamContext> &forward_params,
                          vector<ParamContext> &backward_params) = 0;
  virtual bool valid() const = 0;

  // TODO改造
  virtual void forward(vector<ParamContext> &forward_params) = 0;
  virtual void backward() = 0;

  CommonFeaContext()
      : forward_param_container(new FtrlParamContainer(1)),
        backward_param_container(new FtrlParamContainer(1)),
        local_buff_container(new FtrlParamContainer(1))
  {}

  virtual ~CommonFeaContext() {}
};
