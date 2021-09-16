/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "train/train_opt.h"
#include "utils/base.h"
#include "utils/utils.h"
#include "synchronize/mutex_adapter.h"


// 通用参数的头部结构
struct ParamUnitHead {
  real_t w;
  real_t V[];
};

class ParamContainerInterface;

struct ParamContext {
  ParamContext(ParamContainerInterface *_container = NULL, ParamUnitHead *_param = NULL, Mutex_t *_mutex = NULL,
               real_t _x = 0.0)
      : container(_container), param(_param), mutex(_mutex), x(_x) {}
  ParamContainerInterface *container;
  ParamUnitHead *param;
  Mutex_t *mutex;
  real_t x;
};


class ParamContainerInterface {
 public:
  ParamContainerInterface(feaid_t total_fea_num, int _param_size_of_one_fea)
      : v_dim(train_opt.factor_num),
        param_size_of_one_fea(_param_size_of_one_fea),
        fea_num(total_fea_num)
  {
    param_base_addr =
        (unsigned char *)malloc((fea_num + 1) * param_size_of_one_fea);
  }

  virtual ~ParamContainerInterface() {
    if (param_base_addr) free(param_base_addr);
  }

  virtual void init_params() = 0;

  void clear_weights(ParamUnitHead * param_addr) {
    param_addr->w = 0.0;
    for (int i=0; i < v_dim; i++) {
      param_addr->V[i] = 0.0;
    }
  }

  bool isBadID(feaid_t id) const { return id > fea_num || id < 0; }
  feaid_t getUNKnownID() const { return fea_num; }

  ParamUnitHead *get() {
    return (ParamUnitHead *)param_base_addr;
  }

  ParamUnitHead *get(feaid_t id) {
    if (UNLIKELY(isBadID(id))) {
      id = getUNKnownID();
    }
    return (ParamUnitHead *)(param_base_addr + param_size_of_one_fea * id);
  }

  int get_param_unit_size() {
    return param_size_of_one_fea;
  }

  // forward
  bool cp_param(ParamUnitHead * dest, feaid_t fid) {
    const ParamUnitHead * param_addr = get(fid);
    cp_param(dest, param_addr);
  }

  bool cp_param(ParamUnitHead * dest, const ParamUnitHead * param_addr) {
    memcpy((void *)dest, (const void *)param_addr, param_size_of_one_fea);
  }

  bool cp_weights(ParamUnitHead * dest, feaid_t fid) {
    const ParamUnitHead * param_addr = get(fid);
    cp_weights(dest, param_addr);
  }

  bool cp_weights(ParamUnitHead * dest, const ParamUnitHead * param_addr) {
    memcpy((void *)dest, (const void *)param_addr, sizeof(real_t) * (1 + v_dim));
  }

  bool add_weights_to(feaid_t fid, ParamUnitHead * dest);
  bool add_weights_to(const ParamUnitHead * param_addr, ParamUnitHead * dest) {
    dest->w += param_addr->w;
    for (int i=0; i < v_dim; i++) {
      dest->V[i] += param_addr->V[i];
    }
  }

  // // backward  现在update是靠solver中的虚函数实现的，没有用这里的update。
  // // 如果放在这里，则不需要solver的多态，update的多态在这里实现，只需要一个solver的实现就行。但是不方便保存update的一些中间变量。需要一个context。
  // void update_param(feaid_t fid, real_t grad) {
  //   ParamUnitHead *backward_param = get(fid);
  //   update_param(backward_param, grad);
  // }
  // virtual void update_param(ParamUnitHead *backward_param, real_t grad) = 0;

  const feaid_t fea_num;
  const int v_dim;

  const int param_size_of_one_fea;

  unsigned char *param_base_addr; // TODO做成
  // unsigned char *const param_base_addr;

 private:
  // 禁用拷贝
  ParamContainerInterface(const ParamContainerInterface &ohter);
  ParamContainerInterface &operator=(const ParamContainerInterface &that);
};
