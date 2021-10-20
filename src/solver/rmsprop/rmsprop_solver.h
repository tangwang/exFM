/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/feat_manager.h"
#include "solver/rmsprop/rmsprop_param.h"
#include "solver/base_solver.h"
#include "train/train_opt.h"

class RmspropSolver : public BaseSolver {
 public:
  RmspropSolver(const FeatManager &feat_manager)
      : BaseSolver(feat_manager),
        lr(train_opt.rmsprop.lr),
        l2_norm_w(train_opt.rmsprop.l2_norm_w),
        l2_norm_V(train_opt.rmsprop.l2_norm_V),
        beta2(train_opt.rmsprop.beta2) {}

  virtual ~RmspropSolver() {}

  virtual void update() {

    for (auto &kv : batch_params) {
      ParamNode &param_node = kv.second;
      FMParamUnit & grad = param_node.fm_grad;
      batchReduce(grad, param_node.count);

      RmspropParamUnit *backward_param = (RmspropParamUnit *)param_node.param;
      param_node.mutex->writeLock();

      // update w
      real_t &w = backward_param->fm_param.w;
      real_t &wv = backward_param->avg_squared.w;

      wv = beta2 * wv + (1 - beta2) * grad.w * grad.w;

      DEBUG_OUT << "rmsprop_solver: grad:" << grad << " decayed_lr" << lr / (std::sqrt(wv) + eps)
                << " count " << param_node.count
                << " wv:" << wv << " update:" 
                << lr * (grad.w + l2_norm_w * w) / (std::sqrt(wv) + eps)
                << endl;

      w -= lr * (grad.w + l2_norm_w * w)  / (std::sqrt(wv) + eps);

      // update V
      for (int f = 0; f < DIM; ++f) {
        real_t &vf = backward_param->fm_param.V[f];
        real_t &vvf = backward_param->avg_squared.V[f];
        real_t vgf = grad.V[f];

        vvf = beta2 * vvf + (1 - beta2) * vgf * vgf;
        vf -= lr * (vgf + l2_norm_V * vf)  / (std::sqrt(vvf) + eps);
      }
      param_node.mutex->unlock();
    }
  }

  const real_t lr;
  const real_t l2_norm_w;
  const real_t l2_norm_V;
  const real_t beta2;
  static constexpr real_t eps = 1e-7;
  static constexpr bool amsgrad = false; // 保留历史最大的v_t，记为v_{max}，每次计算都是用最大的v_{max}，否则是用当前v_t
  // amsgrad需要多一个保存一份历史最大值平方梯度v_{max}。 暂未实现
  // avg_grads = beta1 * avg_grads + (1-beta1) * w.grad
  // avg_squared = beta2 * (avg_squared) + (1-beta2) * (w.grad * w.grad)
  // max_squared = max(avg_squared, max_squared)
  // w = w - lr * avg_grads / sqrt(max_squared)
};
