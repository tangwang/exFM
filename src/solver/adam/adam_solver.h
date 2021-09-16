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
  AdamSolver(const FeaManager &fea_manager) : BaseSolver(fea_manager) , t(0), beta1_pow(1.0), beta2_pow(1.0) {}
  virtual ~AdamSolver() {}

  virtual void update(real_t grad) {

    beta1_pow *= train_opt.adam.beta1;
    beta2_pow *= train_opt.adam.beta2;

    for (auto param_context : backward_params) {
      AdamParamUnit *backward_param = (AdamParamUnit *)param_context.param;
      param_context.mutex->lock();
      real_t xi = 1.0;
      real_t wg = grad * xi;
      real_t & w = backward_param->head.w;
      real_t & wm = backward_param->multabel_wm();
      real_t & wv = backward_param->multabel_wv();
      
      wm = train_opt.adam.beta1 * wm + (1-train_opt.adam.beta1)*wg;
      wv = train_opt.adam.beta2 * wv + (1-train_opt.adam.beta2)*wg*wg;

      wm = wm / (1-beta1_pow);
      wv = wv / (1-beta2_pow);
      
      w -= (train_opt.adam.step_size * wm/ (std::pow(wv, 0.5) + train_opt.adam.eps));

      for (int f = 0; f < train_opt.factor_num; ++f) {

        real_t &vf = backward_param->head.V[f];
        real_t &vmf = backward_param->multabel_vm(f);
        real_t &vvf = backward_param->multabel_vv(f);

        real_t vgf = wg * (sum[f]  - vf * xi );

        vmf = train_opt.adam.beta1 * vmf + (1 - train_opt.adam.beta1) * vgf;
        vvf = train_opt.adam.beta2 * vvf + (1 - train_opt.adam.beta2) * vgf * vgf;

        vmf = vmf / (1 - beta1_pow);
        vvf = vvf / (1 - beta2_pow);

        vf -= (train_opt.adam.step_size * vmf /
               (std::pow(vvf, 0.5) + train_opt.adam.eps));
      }

      param_context.mutex->unlock();
    }
    
  }
  unsigned long t;
  real_t beta1_pow;
  real_t beta2_pow;
};
