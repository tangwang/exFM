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
        lr(train_opt.adam.lr),
        bias_correct(train_opt.adam.bias_correct != 0),
        eps(train_opt.adam.eps),
        beta1(train_opt.adam.beta1),
        beta2(train_opt.adam.beta2),
        weight_decay_w(train_opt.adam.weight_decay_w),
        weight_decay_V(train_opt.adam.weight_decay_V) {}
  virtual ~AdamSolver() {}

  virtual void update(real_t grad) {
    if (y == 1) grad *= 7.2816; // TODO 正负样本loss均衡。暂时写死

    // TODO 这里的pow(beta1_pow, t), t是取总步数，该是取该参数更新的次数？

    for (auto param_context : backward_params) {
      AdamParamUnit *backward_param = (AdamParamUnit *)param_context.param;
      param_context.mutex->lock();
      real_t xi = 1.0;
      // grad *= xi; //暂时都是离散特征，不支持连续值特征，所以此处关闭
      real_t & w = backward_param->head.w;
      real_t & wm = backward_param->multabel_wm();
      real_t & wv = backward_param->multabel_wv();

      real_t & _beta1power_t = backward_param->multabel_beta1power_t();
      real_t & _beta2power_t = backward_param->multabel_beta2power_t();
      _beta1power_t *= beta1;
      _beta2power_t *= beta2;

      wm = beta1 * wm + (1-beta1)*grad;
      wv = beta2 * wv + (1-beta2)*grad*grad;

      real_t corrected_wm = bias_correct ? wm : (wm / (1-_beta1power_t));
      real_t corrected_wv = bias_correct ? wv : (wv / (1-_beta2power_t));
      
      w -= lr * (corrected_wm/ (std::sqrt(corrected_wv) + eps) + weight_decay_w * w);

      for (int f = 0; f < train_opt.factor_num; ++f) {

        real_t &vf = backward_param->head.V[f];
        real_t &vmf = backward_param->multabel_vm(f);
        real_t &vvf = backward_param->multabel_vv(f);

        real_t vgf = grad * (sum[f]  - vf * xi );

        vmf = beta1 * vmf + (1 - beta1) * vgf;
        vvf = beta2 * vvf + (1 - beta2) * vgf * vgf;

        real_t corrected_vmf = bias_correct ? vmf : (vmf / (1 - beta1_pow));
        real_t corrected_vvf = bias_correct ? vvf : (vvf / (1 - beta2_pow));

        vf -= lr * (corrected_vmf /
               (std::sqrt(corrected_vvf) + eps)  + weight_decay_V * vf);
      }

      param_context.mutex->unlock();
    }
  }
  const bool bias_correct;
  const real_t eps;
  const real_t lr;
  const real_t beta1;
  const real_t beta2;
  const real_t weight_decay_w;
  const real_t weight_decay_V;
  unsigned long t;
  real_t beta1_pow;
  real_t beta2_pow;
};
