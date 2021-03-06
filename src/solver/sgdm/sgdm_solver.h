/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/feat_manager.h"
#include "solver/sgdm/sgdm_param.h"
#include "solver/base_solver.h"
#include "train/train_opt.h"

class SgdmSolver : public BaseSolver {
 public:
  SgdmSolver(const FeatManager &feat_manager)
      : BaseSolver(feat_manager),
        lr(train_opt.sgdm.lr),
        beta1(train_opt.sgdm.beta1),
        l1_reg_w(train_opt.sgdm.l1_reg_w),
        l1_reg_V(train_opt.sgdm.l1_reg_V),
        l2_reg_w(train_opt.sgdm.l2_reg_w),
        l2_reg_V(train_opt.sgdm.l2_reg_V)
        {}
  virtual ~SgdmSolver() {}

  virtual void update() {

    for (auto & kv : batch_params) {
      ParamNode & param_node = kv.second;
      FMParamUnit & grad = param_node.fm_grad;
      batchReduce(grad, param_node.count);

      SgdmParamUnit *backward_param = (SgdmParamUnit *)param_node.param;
      param_node.mutex->writeLock();

      real_t & w = backward_param->fm_param.w;
      real_t & wm = backward_param->momentum.w;

      wm = beta1 * wm + (1-beta1) * grad.w;
      w -= lr * (wm  + w * l2_reg_w);

      for (int f = 0; f < DIM; ++f) {
        real_t &vf = backward_param->fm_param.V[f];
        real_t & vmf = backward_param->momentum.V[f];
        real_t vgf = grad.V[f];

        vmf = beta1 * vmf + (1-beta1) * vgf;
        vf -= lr * (vmf + vf * l2_reg_V);
      }
      param_node.mutex->unlock();
    }
  }

  const real_t lr;
  const real_t beta1;
  const real_t l1_reg_w;
  const real_t l1_reg_V;
  const real_t l2_reg_w;
  const real_t l2_reg_V;
};
