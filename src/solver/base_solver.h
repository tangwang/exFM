/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/feat_manager.h"
#include "solver/parammeter_container.h"
#include "utils/base.h"
#include "train/train_opt.h"

class Sample {
 public:
  Sample() {}
  ~Sample() {}

  real_t forward();

  void backward();

  size_t fm_layer_nodes_size;
  vector<FmLayerNode> fm_layer_nodes;
  
  real_t logit;
  real_t loss;
  real_t grad;
  real_t sum[DIM];
  real_t sum_sqr[DIM];
  int y;
};

class BaseSolver {
 public:
  BaseSolver(const FeatManager &feat_manager);

  virtual ~BaseSolver() {}

  real_t forward(const string & aline) {
    // label统一为1， -1的形式
    // y = atoi(line) > 0 ? 1 : -1;
    char line[aline.size() + 1];
    memcpy(line, aline.c_str(), aline.size() + 1);
    char * pos = line;
    char * feat_beg = NULL;
    char * feat_end = strchr(pos, train_opt.feat_seperator);
    char * feat_kv_pos = NULL;
    if (unlikely(*pos == '\0' || feat_end == NULL)) {
      return -1;
    }

    Sample &sample = batch_samples[sample_idx];

    // parse label
    sample.y = pos[0] == '1' ? 1 : -1;

    // parse featrues
    size_t fm_node_idx = 0;
    do {
      feat_beg = feat_end + 1;
      feat_end = strchr(feat_beg, train_opt.feat_seperator);
      feat_kv_pos = strchr(feat_beg, train_opt.feat_kv_seperator);
      if (likely(feat_kv_pos != NULL && (feat_end == NULL || feat_kv_pos + 1 < feat_end))) {
        *feat_kv_pos = '\0';
        if (feat_end) *feat_end = '\0';
        auto got = feat_map.find(feat_beg);
        if (got != feat_map.end()) {
          DEBUG_OUT << " feed : "<< fm_node_idx << " " << feat_beg << " " << feat_kv_pos + 1 << endl;
          got->second->feedSample(feat_kv_pos + 1, sample.fm_layer_nodes[fm_node_idx++]);
        }
      }
    } while (feat_end != NULL);
    sample.fm_layer_nodes_size = fm_node_idx;

    return sample.forward();
  }

  void train(const string & line, int &y, real_t &logit, real_t & loss, real_t & grad) {
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

  void test(const string & line, int &y, real_t &logit) {
    forward(line);
    y = batch_samples[sample_idx].y;
    logit = batch_samples[sample_idx].logit;
  }

protected:
  virtual void update() = 0;

  void rotateSampleIdx() {
    ++sample_idx;
    if (sample_idx == batch_size) {
      // merge gradient
      batch_params.clear();
      for (size_t i = 0; i < batch_size; i++) {
        const Sample &sample = batch_samples[i];

        for (size_t feat_idx = 0; feat_idx < sample.fm_layer_nodes_size; feat_idx++) {
          const auto & fm_node = sample.fm_layer_nodes[feat_idx];
          for (const auto & param_node : fm_node.backward_nodes) {
            auto ins_ret = batch_params.insert({param_node.param, param_node});
            if (!ins_ret.second) {
              ins_ret.first->second.fm_grad += param_node.fm_grad;
              ins_ret.first->second.count += 1;
            }
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
  const FeatManager &feat_manager_;
  vector<DenseFeatContext> dense_feas;
  vector<SparseFeatContext> sparse_feas;
  vector<VarlenSparseFeatContext> varlen_feas;
  std::unordered_map<string, CommonFeatContext *> feat_map;

  const size_t batch_size;
  size_t sample_idx;
  vector<Sample> batch_samples;

  std::unordered_map<FMParamUnit *, ParamNode> batch_params;
};
