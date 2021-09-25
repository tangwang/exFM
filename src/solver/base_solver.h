/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/fea_manager.h"
#include "solver/parammeter_container.h"
#include "utils/base.h"
#include "train/train_opt.h"

class Sample {
 public:
  Sample() {}
  ~Sample() {}

  real_t forward();

  void backward();

  vector<ParamContext> forward_params;
  vector<ParamContext> backward_params;

  real_t logit;
  real_t loss;
  real_t grad;
  real_t sum[DIM];
  real_t sum_sqr[DIM];
  int y;
};

class BaseSolver {
 public:
  BaseSolver(const FeaManager &fea_manager);

  virtual ~BaseSolver() {}

  real_t forward(const char *line) {
    feedSample(line);
    return batch_samples[sample_idx].forward();
  }

  void train(const char *line, int &y, real_t &logit, real_t & loss, real_t & grad) {
    forward(line);

    batch_samples[sample_idx].backward();
    
    y = batch_samples[sample_idx].y;
    logit = batch_samples[sample_idx].logit;
    loss = batch_samples[sample_idx].loss;
    grad = batch_samples[sample_idx].grad;

    rotateSampleIdx();

    DEBUG_OUT << "BaseSolver::train "
              << " y " << y << " logit " << logit << endl;
  }

  void test(const char *line, int &y, real_t &logit) {
    forward(line);
    y = batch_samples[sample_idx].y;
    logit = batch_samples[sample_idx].logit;
  }

protected:
  virtual void update() = 0;

  int feedSample(const char *line);

  void rotateSampleIdx() {
    ++sample_idx;
    if (sample_idx == batch_size) {
      // merge grads
      batch_params.clear();
      for (size_t i = 0; i < batch_size; i++) {
        Sample &sample = batch_samples[i];
        for (const auto &param_unit : sample.backward_params) {
          auto ins_ret = batch_params.insert({param_unit.param, param_unit});
          if (!ins_ret.second) {
            ins_ret.first->second.fm_grad += param_unit.fm_grad;
            ins_ret.first->second.count += 1;
          }
        }
      }
      DEBUG_OUT << "batch update :" << batch_params.size() << endl;
      update();
      sample_idx = 0;
    }
  }

  void batchReduce(FMParamUnit &grad, int count) {
    switch (train_opt.batch_grad_reduce_type) {
      case TrainOption::BatchGradReduceType_Sum:
        break;
      case TrainOption::BatchGradReduceType_AvgByOccurrences:
        grad /= count;
        break;
      case TrainOption::BatchGradReduceType_AvgByOccurrencesSqrt:
        grad /= std::sqrt(count);
        break;
      default: 
      // BatchGradReduceType_AvgByBatchSize by default
        grad /= train_opt.batch_size;
        break;
    }
  }

protected:
  const FeaManager &fea_manager_;
  vector<DenseFeaContext> dense_feas;
  vector<SparseFeaContext> sparse_feas;
  vector<VarlenSparseFeaContext> varlen_feas;

  const size_t batch_size;
  size_t sample_idx;
  vector<Sample> batch_samples;

  std::map<FMParamUnit *, ParamContext> batch_params;
};
