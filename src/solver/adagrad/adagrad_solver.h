/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/feat_manager.h"
#include "solver/adagrad/adagrad_param.h"
#include "solver/base_solver.h"
#include "train/train_opt.h"

class AdagradSolver : public BaseSolver {
 public:
  AdagradSolver(const FeatManager &feat_manager)
      : BaseSolver(feat_manager),
        lr(train_opt.adagrad.lr),
        l2_norm_w(train_opt.adagrad.l2_norm_w),
        l2_norm_V(train_opt.adagrad.l2_norm_V)
        {}

  virtual ~AdagradSolver() {}

  virtual void update() {

    for (auto &kv : batch_params) {
      ParamContext &param_context = kv.second;
      FMParamUnit grad = param_context.fm_grad;
      batchReduce(grad, param_context.count);

      AdagradParamUnit *backward_param = (AdagradParamUnit *)param_context.param;
      param_context.mutex->lock();

      // update w
      real_t &w = backward_param->fm_param.w;
      real_t &wv = backward_param->squared_sum.w;

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
        real_t &vvf = backward_param->squared_sum.V[f];
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
  static constexpr bool amsgrad = false; // 保留历史最大的v_t，记为v_{max}，每次计算都是用最大的v_{max}，否则是用当前v_t
  // amsgrad需要多一个保存一份历史最大值平方梯度v_{max}。 暂未实现
  // avg_grads = beta1 * avg_grads + (1-beta1) * w.grad
  // squared_sum = beta2 * (squared_sum) + (1-beta2) * (w.grad * w.grad)
  // max_squared = max(squared_sum, max_squared)
  // w = w - lr * avg_grads / sqrt(max_squared)
};
