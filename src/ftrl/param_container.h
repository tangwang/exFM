/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "ftrl/train_opt.h"
#include "utils/base.h"
#include "utils/utils.h"

class FTRLParamUnit {
 public:
  static int offset_vn;
  static int offset_vz;
  static int factor_num;
  static int full_size;

  /* data */
  real_t w;
  real_t wn;
  real_t wz;
  real_t buff[];

  real_t &multabel_v(int factor) { return buff[factor]; }
  real_t &multabel_vn(int factor) { return buff[offset_vn + factor]; }
  real_t &multabel_vz(int factor) { return buff[offset_vz + factor]; }
  const real_t &v(int factor) const { return buff[factor]; }
  const real_t &vn(int factor) const { return buff[offset_vn + factor]; }
  const real_t &vz(int factor) const { return buff[offset_vz + factor]; }

  static void static_init() {
    factor_num = train_opt.factor_num;
    offset_vn = train_opt.factor_num;
    offset_vz = train_opt.factor_num + train_opt.factor_num;
    full_size = sizeof(FTRLParamUnit) + train_opt.factor_num * 3 * sizeof(real_t);
  }

  void param_init() {
    w = wn = wz = 0.0;

    for (int f = 0; f < FTRLParamUnit::factor_num; ++f) {
      multabel_v(f) = utils::gaussian(train_opt.init_mean, train_opt.init_stdev);
      multabel_vn(f) = 0.0;
      multabel_vz(f) = 0.0;
    }
  }

  void operator=(const FTRLParamUnit &rhs) {
    memcpy((void *)this, (const void *)&rhs, full_size);
  }

  void operator+(const FTRLParamUnit &rhs) {
    w += rhs.w;
    wn += rhs.wn;
    wz += rhs.wz;
    for (int i = 0; i < factor_num * 3; i++) {
      multabel_v(i) += rhs.v(i);
    }
  }

  void clear_weights() {
    w = 0.0;
    for (int i = 0; i < factor_num; i++) {
      multabel_v(i) = 0.0;
    }
  }

  void plus_weights(const FTRLParamUnit &rhs) {
    w += rhs.w;
    for (int i = 0; i < factor_num; i++) {
      multabel_v(i) += rhs.v(i);
    }
  }

  void plus_params(const FTRLParamUnit &rhs) {
    wz += rhs.wz;
    wn += rhs.wn;
    for (int i = 0; i < factor_num; i++) {
      multabel_vz(i) += rhs.vz(i);
      multabel_vn(i) += rhs.vn(i);
    }
  }

  inline void calc_w() {
    if (fabs(wz) <= train_opt.l1_reg_w) {
      w = 0.0;
    } else if (wz > 0.0000000001)  // TODO 跟0有区别吗
    {
      w = -(wz - train_opt.l1_reg_w) /
          (train_opt.l2_reg_w + (train_opt.w_beta + sqrt(wn)) / train_opt.w_alpha);
    } else {
      w = -(wz + train_opt.l1_reg_w) /
          (train_opt.l2_reg_w + (train_opt.w_beta + sqrt(wn)) / train_opt.w_alpha);
    }
  }

  void calc_v() {
    for (int f = 0; f < factor_num; ++f) {
      real_t &vf = multabel_v(f);
      const real_t &vnf = vn(f);
      const real_t &vzf = vz(f);
      if (vnf > 0) {
        if (fabs(vzf) <= train_opt.l1_reg_V) {
          vf = 0.0;
        } else if (vzf > 0.0000000001)  // TODO 跟0有区别吗
        {
          vf = -(vzf - train_opt.l1_reg_V) /
               (train_opt.l2_reg_V + (train_opt.v_beta + sqrt(vnf)) / train_opt.v_alpha);
        } else {
          vf = -(vzf + train_opt.l1_reg_V) /
               (train_opt.l2_reg_V + (train_opt.v_beta + sqrt(vnf)) / train_opt.v_alpha);
        }
      }
    }
  }

  void calc_param() {
    calc_w();
    calc_v();
  }

  FTRLParamUnit() {}
  ~FTRLParamUnit() {}
};

class ParamContainer {
 public:
  ParamContainer(feaid_t total_fea_num) {
    fea_num = total_fea_num;
    // fid为0到vocab_size，最后一个为 UNKNOWN
    param = (unsigned char *)malloc((fea_num + 1) * FTRLParamUnit::full_size);
    // param = memalign(sizeof(real_t), fea_num * FTRLParamUnit::full_size);

    param_init();
  }

  void param_init() {
    for (feaid_t i = 0; i < fea_num + 1; i++) {
      get(i)->param_init();
    }
  }

  void Destroy() {
    if (param) free(param);
  }

  ~ParamContainer() {
    if (param) free(param);
  }

  bool isBadID(feaid_t id) const { return id > fea_num || id < 0; }
  feaid_t getUNKnownID() const { return fea_num; }

  int read(feaid_t id, FTRLParamUnit *p) {
    if (UNLIKELY(isBadID(id))) {
      id = getUNKnownID();
    }
    FTRLParamUnit *param_addr = get(id);
    *p = *param_addr;
    return 0;
  }

  int write(feaid_t id, FTRLParamUnit *p) {
    if (UNLIKELY(isBadID(id))) {
      id = getUNKnownID();
    }
    FTRLParamUnit *param_addr = get(id);
    *param_addr = *p;
    return 0;
  }

  FTRLParamUnit *get() {
    return (FTRLParamUnit *)param;
  }

  FTRLParamUnit *get(feaid_t id) {
    if (UNLIKELY(isBadID(id))) {
      id = getUNKnownID();
    }
    return (FTRLParamUnit *)(param + FTRLParamUnit::full_size * id);
  }

  void set(feaid_t id, const FTRLParamUnit &v) {
    if (UNLIKELY(isBadID(id))) {
      id = getUNKnownID();
    }

    FTRLParamUnit *fea_unit = get(id);
    *fea_unit = v;
  }

  unsigned char *param;
  feaid_t fea_num;

 private:
  // 禁用拷贝
  ParamContainer(const ParamContainer &ohter);
  ParamContainer &operator=(const ParamContainer &that);
};
