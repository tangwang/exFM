/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "solver/parammeter_interface.h"

class AdamParamUnit {
 public:

 public:
  static int offset_vm;
  static int offset_vv;
  static int factor_num;
  static int full_size;
  /* data */
  // w, V[0~factor_num-1], wn, wz, vn[0~factor_num-1], vz[0~factor_num-1]
  ParamUnitHead head;

  real_t &multabel_v(int factor) { return head.V[factor]; }
  real_t &multabel_vm(int factor) { return head.V[offset_vm + factor]; }
  real_t &multabel_vv(int factor) { return head.V[offset_vv + factor]; }
  real_t &multabel_wm() { return head.V[factor_num]; }
  real_t &multabel_wv() { return head.V[factor_num+1]; }
  const real_t &v(int factor) const { return head.V[factor]; }
  const real_t &vn(int factor) const { return head.V[offset_vm + factor]; }
  const real_t &vz(int factor) const { return head.V[offset_vv + factor]; }
  const real_t &wn() const { return head.V[factor_num]; }
  const real_t &wz() const { return head.V[factor_num+1]; }

  static void static_init() {
    factor_num = train_opt.factor_num;
    offset_vm = train_opt.factor_num + 2;
    offset_vv = train_opt.factor_num *2 + 2;
    full_size = sizeof(AdamParamUnit) +  (2 + train_opt.factor_num * 3 )* sizeof(real_t);
  }

  void init_params() {
    head.w = 0.0;
    multabel_wm() = 0.0;
    multabel_wv() = 0.0;
    for (int f = 0; f < AdamParamUnit::factor_num; ++f) {
      multabel_v(f) = utils::gaussian(train_opt.ftrl.init_mean, train_opt.ftrl.init_stdev);
      multabel_vm(f) = 0.0;
      multabel_vv(f) = 0.0;
    }
  }

  AdamParamUnit() {}
  ~AdamParamUnit() {}
};

class AdamParamContainer : public ParamContainerInterface {
 public:
  AdamParamContainer(feaid_t total_fea_num)
      : ParamContainerInterface(
            total_fea_num,
            sizeof(AdamParamUnit) +
                (2 + train_opt.factor_num * 3) * sizeof(real_t)) {
    init_params();
  }
  virtual ~AdamParamContainer() {}

  virtual void init_params() {
    for (feaid_t i = 0; i < fea_num + 1; i++) {
      AdamParamUnit * p = (AdamParamUnit *)get(i);
      p->init_params();
    }
  }

  // virtual void update_param(ParamUnitHead *_backward_param, real_t grad) {
  //   AdamParamUnit *backward_param = (AdamParamUnit *)_backward_param;

  //   real_t xi = 1.0;
  //   real_t wg = mult * xi;
  //   real_t & w = backward_param->head.w;
  //   real_t & wm = backward_param->multabel_wm();
  //   real_t & wv = backward_param->multabel_wv();
    
  //   wm = train_opt.adam.beta1 * wm + (1-train_opt.adam.beta1)*wg;
  //   wv = train_opt.adam.beta2 * wv + (1-train_opt.adam.beta2)*wg*wg;

  //   wm = wm / (1-beta1_pow);
  //   wv = wv / (1-beta2_pow);
    
  //   w -= (train_opt.adam.step_size * wm/ (math.powf(wv, 0.5) + train_opt.adam.eps));

  //   for (int f = 0; f < train_opt.factor_num; ++f) {
  //     const real_t &vf = backward_param->head.V[f];
  //     real_t &vmf = backward_param->multabel_vm(f);
  //     real_t &vvf = backward_param->multabel_vv(f);
  //     vmf = train_opt.adam.beta1 * vmf + (1 - train_opt.adam.beta1) * wg;
  //     vvf = train_opt.adam.beta2 * vvf + (1 - train_opt.adam.beta2) * wg * wg;

  //     vmf = vmf / (1 - beta1_pow);
  //     vvf = vvf / (1 - beta2_pow);

  //     vf -= (train_opt.adam.step_size * vmf /
  //             (math.powf(vvf, 0.5) + train_opt.adam.eps));
  //   }
    
  //   beta1_pow *= train_opt.adam.beta1;
  //   beta2_pow *= train_opt.adam.beta2;
  // }
};
