/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/fea_manager.h"
#include "solver/adam/adam_param.h"
#include "solver/base_solver.h"
#include "train/train_opt.h"

class AdamSolver : public BaseSolver {
 public:
  AdamSolver(const FeaManager &fea_manager)
      : BaseSolver(fea_manager),
        t(0),
        beta1_pow(1.0),
        beta2_pow(1.0),
        step_size(train_opt.adam.step_size),
        bias_correct(train_opt.adam.bias_correct != 0),
        eps(train_opt.adam.eps),
        beta1(train_opt.adam.beta1),
        beta2(train_opt.adam.beta2),
        weight_decay_w(train_opt.adam.weight_decay_w),
        weight_decay_V(train_opt.adam.weight_decay_V) {}
  virtual ~AdamSolver() {}

  virtual void update(real_t grad) {

    // TODO 这里的pow(beta1_pow, t), t是取总步数，该是取该参数更新的次数？
    beta1_pow *= beta1;
    beta2_pow *= beta2;

    for (auto param_context : backward_params) {
      AdamParamUnit *backward_param = (AdamParamUnit *)param_context.param;
      param_context.mutex->lock();
      real_t xi = 1.0;
      real_t wg = grad * xi;
      real_t & w = backward_param->head.w;
      real_t & wm = backward_param->multabel_wm();
      real_t & wv = backward_param->multabel_wv();
      
      wm = beta1 * wm + (1-beta1)*wg;
      wv = beta2 * wv + (1-beta2)*wg*wg;

      real_t corrected_wm = bias_correct ? wm : wm / (1-beta1_pow);
      real_t corrected_wv = bias_correct ? wv : wv / (1-beta2_pow);
      
      w -= (step_size * corrected_wm/ (std::pow(corrected_wv, 0.5) + eps) + weight_decay_w * w);

      for (int f = 0; f < train_opt.factor_num; ++f) {

        real_t &vf = backward_param->head.V[f];
        real_t &vmf = backward_param->multabel_vm(f);
        real_t &vvf = backward_param->multabel_vv(f);

        real_t vgf = wg * (sum[f]  - vf * xi );

        vmf = beta1 * vmf + (1 - beta1) * vgf;
        vvf = beta2 * vvf + (1 - beta2) * vgf * vgf;

        real_t corrected_vmf = vmf / (1 - beta1_pow);
        real_t corrected_vvf = vvf / (1 - beta2_pow);

        vf -= (step_size * corrected_vmf /
               (std::pow(corrected_vvf, 0.5) + eps)  + weight_decay_V * vf);
      }

      param_context.mutex->unlock();
    }
  }
  const bool bias_correct;
  const real_t eps;
  const real_t step_size;
  const real_t beta1;
  const real_t beta2;
  const real_t weight_decay_w;
  const real_t weight_decay_V;
  unsigned long t;
  real_t beta1_pow;
  real_t beta2_pow;
};
