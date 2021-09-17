/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "solver/parammeter_interface.h"

class SgdmParamUnit {
public:
  static int offset_vm;
  static int factor_num;
  static int full_size;
  
  /* data */
  // w, V[0~factor_num-1], wn, wz, vn[0~factor_num-1], vz[0~factor_num-1]
  ParamUnitHead head;

  real_t &multabel_wm() { return head.V[factor_num]; }
  real_t &multabel_vm(int factor) { return head.V[offset_vm + factor]; }
  const real_t &wm() { return head.V[factor_num]; }
  const real_t &vm(int factor) { return head.V[offset_vm + factor]; }

  static void static_init() {
    factor_num = train_opt.factor_num;
    offset_vm = train_opt.factor_num + 1;
    full_size = (2 + train_opt.factor_num * 2) * sizeof(real_t);
  }

  void init_params() {
    head.w = 0.0;
    multabel_wm() = 0.0;
    for (int f = 0; f < SgdmParamUnit::factor_num; ++f) {
      head.V[f] = utils::gaussian(train_opt.ftrl.init_mean, train_opt.ftrl.init_stdev);
      multabel_vm(f) = 0.0;
    }
  }

  void operator=(const SgdmParamUnit &rhs) {
    memcpy((void *)this, (const void *)&rhs, full_size);
  }

};

class SgdmParamContainer : public ParamContainerInterface {
 public:
  SgdmParamContainer(feaid_t total_fea_num)
      : ParamContainerInterface(total_fea_num, SgdmParamUnit::full_size) {
    init_params();
  }

  virtual ~SgdmParamContainer() {}

  virtual void init_params() {
    for (feaid_t i = 0; i < fea_num + 1; i++) {
      SgdmParamUnit * p = (SgdmParamUnit *)get(i);
      p->init_params();
    }
  }

  // virtual void update_param(ParamUnitHead *backward_param, real_t grad) {
  //     real_t lr = train_opt.sgdm.lr;

  //     real_t xi = 1.0;
  //     real_t wg = grad * xi;
  //     backward_param->w -= lr *  wg;

  //     for (int f = 0; f < train_opt.factor_num; ++f) {
  //       real_t &vf = backward_param->V[f];
  //       real_t vgf = wg * (sum[f] - vf * xi);
  //       vf -= lr * vgf;
  //     }
  // }
};
