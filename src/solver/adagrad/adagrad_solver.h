/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/fea_manager.h"
#include "solver/adagrad/adagrad_param.h"
#include "solver/base_solver.h"
#include "train/train_opt.h"

class AdagradSolver : public BaseSolver {
 public:
  AdagradSolver(const FeaManager &fea_manager)
      : BaseSolver(fea_manager),
        lr(train_opt.adagrad.lr),
        l2_norm_w(train_opt.adagrad.l2_norm_w),
        l2_norm_V(train_opt.adagrad.l2_norm_V) {}

  virtual ~AdagradSolver() {}

  virtual void update() {
    // TODO 这里的pow(beta1_pow, t), t是取总步数，该是取该参数更新的次数？

    for (auto &kv : batch_params) {
      ParamContext &param_context = kv.second;
      FMParamUnit grad = param_context.fm_grad;
      batchReduce(grad, param_context.count);

      AdagradParamUnit *backward_param = (AdagradParamUnit *)param_context.param;
      param_context.mutex->lock();

      // update w
      real_t &w = backward_param->fm_param.w;
      real_t &wv = backward_param->variance_m.w;

      wv += grad.w * grad.w;

      DEBUG_OUT << "adagrad_solver: grad:" << grad << " decayed_lr" << lr / (std::sqrt(wv) + eps)
                << " count " << param_context.count
                << " wv:" << wv << " update:" 
                << lr * (grad.w + l2_norm_w * w) / (std::sqrt(wv) + eps)
                << endl;

      w -= lr * (grad.w + l2_norm_w * w)  / (std::sqrt(wv) + eps);

      // update V
      for (int f = 0; f < DIM; ++f) {
        real_t &vf = backward_param->fm_param.V[f];
        real_t &vvf = backward_param->variance_m.V[f];
        real_t vgf = grad.V[f];

        vvf += vgf * vgf;
        vf -= lr * (vgf + l2_norm_V * vf)  / (std::sqrt(vvf) + eps);
      }
      param_context.mutex->unlock();
    }
  }

  const real_t lr;
  const real_t l2_norm_w;
  const real_t l2_norm_V;
  static constexpr real_t eps = 1e-7;
};
