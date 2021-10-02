/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/feat_manager.h"
#include "solver/adam/adam_param.h"
#include "solver/base_solver.h"
#include "train/train_opt.h"

class AdamSolver : public BaseSolver {
 public:
  AdamSolver(const FeatManager &feat_manager)
      : BaseSolver(feat_manager),
        lr(train_opt.adam.lr),
        beta1(train_opt.adam.beta1),
        beta2(train_opt.adam.beta2),
        beta1_pow(1.0),
        beta2_pow(1.0),
        weight_decay_w(train_opt.adam.weight_decay_w),
        weight_decay_V(train_opt.adam.weight_decay_V) {}

  virtual ~AdamSolver() {}

  virtual void update() {
    // TODO 这里的pow(beta1_pow, t), t是取总步数，该是取该参数更新的次数？

    for (auto &kv : batch_params) {
      ParamContext &param_context = kv.second;
      FMParamUnit grad = param_context.fm_grad;
      batchReduce(grad, param_context.count);

      AdamParamUnit *backward_param = (AdamParamUnit *)param_context.param;
      param_context.mutex->lock();
      // calc corection_lr
      backward_param->beta1power_t *= beta1;
      backward_param->beta2power_t *= beta2;
      real_t bias_correction1 = (1 - backward_param->beta1power_t);
      real_t bias_correction2 = (1 - backward_param->beta2power_t);
      real_t corection_lr = bias_correction ? (lr * std::sqrt(bias_correction2) / bias_correction1) : lr;

      // update w
      real_t &w = backward_param->fm_param.w;
      real_t &wm = backward_param->avg_grad.w;
      real_t &wv = backward_param->avg_squared.w;

      wm = beta1 * wm + (1 - beta1) * grad.w;
      real_t avg_squared = beta2 * wv + (1 - beta2) * grad.w * grad.w;
      wv = amsgrad ? std::max(wv, avg_squared) : avg_squared;

      DEBUG_OUT << "adam_solver: grad:" << grad << " corection_lr:" << corection_lr
                << " count " << param_context.count << corection_lr << " wm:" << wm
                << " wv:" << wv << " update:"
                << corection_lr * (wm / (std::sqrt(wv) + eps) + weight_decay_w * w)
                << endl
                << "fm_param:" << backward_param->fm_param.w << ","
                << backward_param->fm_param.V[0] << ","
                << backward_param->fm_param.V[1] << endl
                << "avg_grad:" << backward_param->avg_grad.w << ","
                << backward_param->avg_grad.V[0] << ","
                << backward_param->avg_grad.V[1] << endl
                << "avg_squared:" << backward_param->avg_squared.w << ","
                << backward_param->avg_squared.V[0] << ","
                << backward_param->avg_squared.V[1] << endl
                << "fm_param.V_0_1 " << backward_param->fm_param.V[0] << ","
                << backward_param->fm_param.V[1] << endl;

      // adamW:
      // Just adding the square of the weights to the loss function is *not*
      // the correct way of using L2 regularization/weight decay with Adam,
      // since that will interact with the m and v parameters in strange ways.
      w -= corection_lr * (wm / (std::sqrt(wv) + eps) + weight_decay_w * w);

      // update V
      for (int f = 0; f < DIM; ++f) {
        real_t &vf = backward_param->fm_param.V[f];
        real_t &vmf = backward_param->avg_grad.V[f];
        real_t &vvf = backward_param->avg_squared.V[f];

        real_t vgf = grad.V[f];

        vmf = beta1 * vmf + (1 - beta1) * vgf;
        vvf = beta2 * vvf + (1 - beta2) * vgf * vgf;
        vf -= corection_lr * (vmf / (std::sqrt(vvf) + eps) + weight_decay_V * vf);
      }
      param_context.mutex->unlock();
    }
  }

  const real_t lr;
  const real_t beta1;
  const real_t beta2;
  real_t beta1_pow;
  real_t beta2_pow;
  const real_t weight_decay_w;
  const real_t weight_decay_V;

  static constexpr real_t eps = 1e-8;
  static constexpr real_t tolerance = 1e-5;
  static constexpr bool resetPolicy = true;
  static constexpr bool exactObjective = false;
  static constexpr bool bias_correction = true;

  static constexpr bool amsgrad = false; // 保留历史最大的v_t，记为v_{max}，每次计算都是用最大的v_{max}，否则是用当前v_t
  // amsgrad需要多一个保存一份历史最大值平方梯度v_{max}。 暂未实现
  // avg_grads = beta1 * avg_grads + (1-beta1) * w.grad
  // avg_squared = beta2 * (avg_squared) + (1-beta2) * (w.grad * w.grad)
  // max_squared = max(avg_squared, max_squared)
  // w = w - lr * avg_grads / sqrt(max_squared)
};
