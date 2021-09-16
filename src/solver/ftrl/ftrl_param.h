/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "solver/parammeter_interface.h"

class FtrlParamUnit {
 public:
  static int offset_vn;
  static int offset_vz;
  static int factor_num;
  static int full_size;
  /* data */
  // w, V[0~factor_num-1], wn, wz, vn[0~factor_num-1], vz[0~factor_num-1]
  ParamUnitHead head;

  real_t &multabel_v(int factor) { return head.V[factor]; }
  real_t &multabel_vn(int factor) { return head.V[offset_vn + factor]; }
  real_t &multabel_vz(int factor) { return head.V[offset_vz + factor]; }
  real_t &multabel_wn() { return head.V[factor_num]; }
  real_t &multabel_wz() { return head.V[factor_num+1]; }
  const real_t &v(int factor) const { return head.V[factor]; }
  const real_t &vn(int factor) const { return head.V[offset_vn + factor]; }
  const real_t &vz(int factor) const { return head.V[offset_vz + factor]; }
  const real_t &wn() const { return head.V[factor_num]; }
  const real_t &wz() const { return head.V[factor_num+1]; }

  static void static_init() {
    factor_num = train_opt.factor_num;
    offset_vn = train_opt.factor_num + 2;
    offset_vz = train_opt.factor_num *2 + 2;
    full_size = sizeof(FtrlParamUnit) +  (2 + train_opt.factor_num * 3 )* sizeof(real_t);
  }

  void init_params() {
    head.w = 0.0;
    multabel_wn() = 0.0;
    multabel_wz() = 0.0;
    for (int f = 0; f < FtrlParamUnit::factor_num; ++f) {
      multabel_v(f) = utils::gaussian(train_opt.ftrl.init_mean, train_opt.ftrl.init_stdev);
      multabel_vn(f) = 0.0;
      multabel_vz(f) = 0.0;
    }
  }

  void operator=(const FtrlParamUnit &rhs) {
    memcpy((void *)this, (const void *)&rhs, full_size);
  }

  inline void calc_w() {
    real_t _wz = wz();
    real_t _wn = wn();
    if (fabs(_wz) <= train_opt.ftrl.l1_reg_w) {
      head.w = 0.0;
    } else // TODO check
    {
      head.w = -(_wz - utils::sign_a_multiply_b(_wz, train_opt.ftrl.l1_reg_w)) /
          (train_opt.ftrl.l2_reg_w + (train_opt.ftrl.w_beta + sqrt(_wn)) / train_opt.ftrl.w_alpha);
    }
  }

  void calc_v() {
    for (int f = 0; f < factor_num; ++f) {
      real_t &vf = head.V[f];
      const real_t &vnf = vn(f);
      const real_t &vzf = vz(f);
      if (vnf > 0) {
        if (fabs(vzf) <= train_opt.ftrl.l1_reg_V) {
          vf = 0.0;
        } 
        else
        {
          vf = -(vzf - utils::sign_a_multiply_b(vzf, train_opt.ftrl.l1_reg_V)) /
               (train_opt.ftrl.l2_reg_V + (train_opt.ftrl.v_beta + sqrt(vnf)) / train_opt.ftrl.v_alpha);
        } 
      }
    }
  }

  // TODO 对于adam不需要calc_param，到时候把调用处注释掉，看性能是否有变化
  void calc_param() {
    calc_w();
    calc_v();
  }

  FtrlParamUnit() {}
  ~FtrlParamUnit() {}
};

class FtrlParamContainer : public ParamContainerInterface {
 public:
  FtrlParamContainer(feaid_t total_fea_num) : 
  ParamContainerInterface(total_fea_num, sizeof(FtrlParamUnit) + (2 + train_opt.factor_num * 3) * sizeof(real_t)),
  factor_num(train_opt.factor_num),
  offset_vn(train_opt.factor_num + 2),
  offset_vz(train_opt.factor_num * 2 + 2)
   {
    init_params();
  }
  virtual ~FtrlParamContainer() {}

  virtual void init_params() {
    for (feaid_t i = 0; i < fea_num + 1; i++) {
      FtrlParamUnit * p = (FtrlParamUnit *)get(i);
      p->init_params();
    }
  }

  // virtual void update_param(ParamUnitHead *_backward_param, real_t grad) {

  //   // 需要保存sum等
  //   // FtrlParamUnit *backward_param = (FtrlParamUnit *)_backward_param;

  //   // real_t & wz = backward_param->multabel_wz();
  //   // real_t & wn = backward_param->multabel_wn();

  //   // real_t xi = 1.0;
  //   // real_t wg = grad * xi;
  //   // real_t ws =
  //   //     1 / train_opt.ftrl.w_alpha *
  //   //     (sqrt(wn + wg * wg) - sqrt(wn));

  //   // wz += wg - ws * backward_param->head.w;
  //   // wn += wg * wg;

  //   // for (int f = 0; f < train_opt.factor_num; ++f) {
  //   //   const real_t &vf = backward_param->head.V[f];
  //   //   real_t &vnf = backward_param->multabel_vn(f);
  //   //   real_t &vzf = backward_param->multabel_vz(f);
  //   //   real_t vgf = grad * (sum[f] * xi - vf * xi * xi);
  //   //   real_t vsf =
  //   //       1 / train_opt.ftrl.v_alpha * (sqrt(vnf + vgf * vgf) - sqrt(vnf));

  //   //   vzf += vgf - vsf * vf;
  //   //   vnf += vgf * vgf;
  //   // }

  //   // backward_param->calc_param();

  // }

  const int offset_vn;
  const int offset_vz;
  const int factor_num;
};
