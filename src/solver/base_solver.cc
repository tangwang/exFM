/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "solver/base_solver.h"
#include "feature/fea_manager.h"

BaseSolver::BaseSolver(const FeaManager &fea_manager)
    : fea_manager_(fea_manager) {
  for (auto &iter : fea_manager_.dense_feas) {
    dense_feas.push_back(std::move(DenseFeaContext(iter)));
  }
  for (auto &iter : fea_manager_.sparse_feas) {
    sparse_feas.push_back(std::move(SparseFeaContext(iter)));
  }
  for (auto &iter : fea_manager_.varlen_feas) {
    varlen_feas.push_back(std::move(VarlenSparseFeaContext(iter)));
  }
}


int BaseSolver::feedSample(const char *line) {
  // label统一为1， -1的形式
  // y = atoi(line) > 0 ? 1 : -1;
  if (UNLIKELY(*line == 0)) {
    return -1;
  }
  y = line[0] == '1' ? 1 : -1;

  do {
    ++line;
  } while (*line != train_opt.fea_seperator);

  forward_params.clear();
  backward_params.clear();
  for (auto &iter : dense_feas) {
    iter.feedSample(line, forward_params, backward_params);
  }
  for (auto &iter : sparse_feas) {
    iter.feedSample(line, forward_params, backward_params);
  }
  for (auto &iter : varlen_feas) {
    iter.feedSample(line, forward_params, backward_params);
  }
  return 0;
}

void BaseSolver::train(int &out_y, real_t &out_logit) {
  out_logit = predict();
  out_y = y;
  real_t grad = y * (1 / (1 + exp(-logit * y)) - 1);
  DEBUG_OUT << "BaseSolver::train " << " y " << y << " logit " << logit  << " grad " << grad << endl;
  update(grad);
}

real_t BaseSolver::predict() {
  real_t sum_sqr = 0.0;
  logit = 0.0;
  for (int f = 0; f < DIM; ++f) {
    sum[f] = sum_sqr = 0.0;
    for (size_t i = 0; i < forward_params.size(); i++) {
      real_t d = forward_params[i].param->V[f];
      sum[f] += d;
      sum_sqr += d * d;
    }
    logit += (sum[f] * sum[f] - sum_sqr);
  }
  logit *= 0.5;

  for (size_t i = 0; i < forward_params.size(); i++) {
    logit += forward_params[i].param->w;
  }
  return logit;
}

// void BaseSolver::update__by_container(real_t grad) {

//   for (auto param_context : backward_params) {
//     param_context.mutex->lock();
//     param_context.container->update_param(param_context.param, grad);
//     param_context.mutex->unlock();
//   }
// }