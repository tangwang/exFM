/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "train/train_opt.h"
#include "utils/base.h"
#include "utils/utils.h"

class FtrlParamUnit {
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
    full_size = sizeof(FtrlParamUnit) + train_opt.factor_num * 3 * sizeof(real_t);
  }

  void param_init() {
    w = wn = wz = 0.0;

    for (int f = 0; f < FtrlParamUnit::factor_num; ++f) {
      multabel_v(f) = utils::gaussian(train_opt.ftrl.init_mean, train_opt.ftrl.init_stdev);
      multabel_vn(f) = 0.0;
      multabel_vz(f) = 0.0;
    }
  }

  void operator=(const FtrlParamUnit &rhs) {
    memcpy((void *)this, (const void *)&rhs, full_size);
  }

  void operator+(const FtrlParamUnit &rhs) {
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

  void plus_weights(const FtrlParamUnit &rhs) {
    w += rhs.w;
    for (int i = 0; i < factor_num; i++) {
      multabel_v(i) += rhs.v(i);
    }
  }

  void plus_params(const FtrlParamUnit &rhs) {
    wz += rhs.wz;
    wn += rhs.wn;
    for (int i = 0; i < factor_num; i++) {
      multabel_vz(i) += rhs.vz(i);
      multabel_vn(i) += rhs.vn(i);
    }
  }

  inline void calc_w() {
    if (fabs(wz) <= train_opt.ftrl.l1_reg_w) {
      w = 0.0;
    } else if (wz > 1e-10)  // TODO 跟0有区别吗
    {
      w = -(wz - train_opt.ftrl.l1_reg_w) /
          (train_opt.ftrl.l2_reg_w + (train_opt.ftrl.w_beta + sqrt(wn)) / train_opt.ftrl.w_alpha);
    } else {
      w = -(wz + train_opt.ftrl.l1_reg_w) /
          (train_opt.ftrl.l2_reg_w + (train_opt.ftrl.w_beta + sqrt(wn)) / train_opt.ftrl.w_alpha);
    }
  }

  void calc_v() {
    for (int f = 0; f < factor_num; ++f) {
      real_t &vf = multabel_v(f);
      const real_t &vnf = vn(f);
      const real_t &vzf = vz(f);
      if (vnf > 0) {
        if (fabs(vzf) <= train_opt.ftrl.l1_reg_V) {
          vf = 0.0;
        } else if (vzf > 1e-10)  // TODO 跟0有区别吗
        {
          vf = -(vzf - train_opt.ftrl.l1_reg_V) /
               (train_opt.ftrl.l2_reg_V + (train_opt.ftrl.v_beta + sqrt(vnf)) / train_opt.ftrl.v_alpha);
        } else {
          vf = -(vzf + train_opt.ftrl.l1_reg_V) /
               (train_opt.ftrl.l2_reg_V + (train_opt.ftrl.v_beta + sqrt(vnf)) / train_opt.ftrl.v_alpha);
        }
      }
    }
  }

  void calc_param() {
    calc_w();
    calc_v();
  }

  FtrlParamUnit() {}
  ~FtrlParamUnit() {}
};

class FtrlParamContainer {
 public:
  FtrlParamContainer(feaid_t total_fea_num) {
    fea_num = total_fea_num;
    // fid为0到vocab_size，最后一个为 UNKNOWN
    param = (unsigned char *)malloc((fea_num + 1) * FtrlParamUnit::full_size);
    // param = memalign(sizeof(real_t), fea_num * FtrlParamUnit::full_size);

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

  ~FtrlParamContainer() {
    if (param) free(param);
  }

  bool isBadID(feaid_t id) const { return id > fea_num || id < 0; }
  feaid_t getUNKnownID() const { return fea_num; }

  int read(feaid_t id, FtrlParamUnit *p) {
    if (UNLIKELY(isBadID(id))) {
      id = getUNKnownID();
    }
    FtrlParamUnit *param_addr = get(id);
    *p = *param_addr;
    return 0;
  }

  int write(feaid_t id, FtrlParamUnit *p) {
    if (UNLIKELY(isBadID(id))) {
      id = getUNKnownID();
    }
    FtrlParamUnit *param_addr = get(id);
    *param_addr = *p;
    return 0;
  }

  FtrlParamUnit *get() {
    return (FtrlParamUnit *)param;
  }

  FtrlParamUnit *get(feaid_t id) {
    if (UNLIKELY(isBadID(id))) {
      id = getUNKnownID();
    }
    return (FtrlParamUnit *)(param + FtrlParamUnit::full_size * id);
  }

  void set(feaid_t id, const FtrlParamUnit &v) {
    if (UNLIKELY(isBadID(id))) {
      id = getUNKnownID();
    }

    FtrlParamUnit *fea_unit = get(id);
    *fea_unit = v;
  }

  unsigned char *param;
  feaid_t fea_num;

 private:
  // 禁用拷贝
  FtrlParamContainer(const FtrlParamContainer &ohter);
  FtrlParamContainer &operator=(const FtrlParamContainer &that);
};
