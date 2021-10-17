/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "solver/base_solver.h"
#include "feature/feat_manager.h"

real_t Sample::forward() {
  logit = 0.0;
  
  for (int f = 0; f < DIM; ++f) {
    sum[f] = sum_sqr[f] = 0.0;
  }

  for (size_t i = 0; i < fm_layer_nodes_size; i++) {
    const auto & node = fm_layer_nodes[i];
    real_t x = 1.0;
    logit += node.forward.w * x;
    for (int f = 0; f < DIM; ++f) {
      real_t d = node.forward.V[f] * x;
      sum[f] += d;
      sum_sqr[f] += d * d;
    }
  }
  real_t sum_factors_score = 0.0;
  for (int f = 0; f < DIM; ++f) {
    sum_factors_score += (sum[f] * sum[f] - sum_sqr[f]);
  }

  logit += (0.5 * sum_factors_score);

  return logit;
}

void Sample::backward() {
  // 计算整体的梯度:
  // crossEntropyLoss = -log( sigmoid(label.i * fm_score(x) ) ) ， label.i = {-1, 1}
  // partitial(loss) / partitial(fm_score(x)) = -label.i * sigmoid( - fm_score(x) * label.i )，y*score
  real_t exp_y_logit = std::exp(logit * label.i);
  grad = -label.i / (1 + exp_y_logit);
  loss = - std::log(1 - 1/(1+std::max(exp_y_logit, 1e-10)));
  
  FMParamUnit backward;
  for (size_t i = 0; i < fm_layer_nodes_size; i++) {
    auto & node = fm_layer_nodes[i];
    //  partitial(fm_score(x)) / partitial(fm_node)
    real_t xi = 1.0;
    real_t grad_i = grad * xi;
    backward.w = grad_i;
    for (int f = 0; f < DIM; ++f) {
      real_t &vf = node.forward.V[f];
      real_t vgf = grad_i * (sum[f] - vf * xi);
      backward.V[f] = vgf;
    }
    // 计算每个fmParamUnit的梯度： partitial(fm_score(x)) / partitial(\theta),  theata = {w_i, V_i1, Vi2, ... Vif} for i in {0, 1, ... N }
    for (auto & param_node : node.backward_nodes) {
      param_node.fm_grad = backward;
      param_node.fm_grad *= param_node.grad_from_fm_node;
    }
  }
}

BaseSolver::BaseSolver(const FeatManager &feat_manager)
    : feat_manager_(feat_manager), batch_size(train_opt.batch_size), sample_idx(0), batch_samples(train_opt.batch_size) {
  for (const auto &iter : feat_manager_.dense_feat_cfgs) {
    dense_feats.emplace_back(iter);
  }
  for (const auto &iter : feat_manager_.sparse_feat_cfgs) {
    sparse_feats.emplace_back(iter);
  }
  for (const auto &iter : feat_manager_.varlen_feat_cfgs) {
    varlen_feats.emplace_back(iter);
  }
  for (auto & sample : batch_samples) {
    sample.fm_layer_nodes.resize(dense_feats.size() + sparse_feats.size() + varlen_feats.size());
  }

  for (auto &iter : dense_feats) {
    feat_map[iter.feat_cfg->name] = &iter;
  }
  for (auto &iter : sparse_feats) {
    feat_map[iter.feat_cfg->name] = &iter;
  }
  for (auto &iter : varlen_feats) {
    feat_map[iter.feat_cfg->name] = &iter;
  }
  if (train_opt.data_formart == TrainOption::DataFormart_CSV) {
    for (size_t i = 0; i < train_opt.csv_columns.size(); i++) {
      const string &column = train_opt.csv_columns[i];
      auto got = feat_map.find(column);
      if (got != feat_map.end()) {
        feat_entries.push_back(make_pair(i, got->second));
     }
    }
    lineProcessor = &BaseSolver::feedLine_CSV;
  } else {
    lineProcessor = &BaseSolver::feedLine_libSVM;
  }
}

void BaseSolver::rotateSampleIdx() {
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

void BaseSolver::batchReduce(FMParamUnit &grad, int count) {
  switch (train_opt.batch_grad_reduce_type) {
    case TrainOption::BatchGradReduceType_AvgByOccurrences:
      grad /= count;
      break;
    case TrainOption::BatchGradReduceType_AvgByOccurrencesSqrt:
      grad /= std::sqrt(count);
      break;
    case TrainOption::BatchGradReduceType_Sum:
      break;
    default: 
    // BatchGradReduceType_AvgByBatchSize by default
      grad /= train_opt.batch_size;
      break;
  }
}

real_t BaseSolver::feedLine_libSVM(const string & aline) {
  // label统一为1， -1的形式
  // label.i = atoi(line) > 0 ? 1 : -1;
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
  sample.label.i = pos[0] == '1' ? 1 : -1;

  // parse featrues
  size_t fm_node_idx = 0;
  do {
    feat_beg = feat_end + 1;
    feat_end = strchr(feat_beg, train_opt.feat_seperator);
    feat_kv_pos = strchr(feat_beg, train_opt.feat_kv_seperator);
    int feat_len = (feat_end == NULL) ? (int)strlen(feat_kv_pos + 1) : feat_end - (feat_kv_pos + 1);
    if (likely(feat_kv_pos != NULL && feat_len > 0)) {
      *feat_kv_pos = '\0';
      if (feat_end) *feat_end = '\0';
      auto got = feat_map.find(feat_beg);
      if (got != feat_map.end()) {
        DEBUG_OUT << " feed : "<< fm_node_idx << " " << feat_beg << " " << feat_kv_pos + 1 << endl;
        got->second->feedSample(feat_kv_pos + 1, feat_len, sample.fm_layer_nodes[fm_node_idx++]);
      }
    }
  } while (feat_end != NULL);
  sample.fm_layer_nodes_size = fm_node_idx;

  return sample.forward();
}

real_t BaseSolver::feedLine_CSV(const string & aline) {
  // label统一为1， -1的形式
  // label.i = atoi(line) > 0 ? 1 : -1;
  line_split_buff.clear();
  utils::split_string(aline, train_opt.feat_seperator, line_split_buff);

  if (unlikely(line_split_buff.size() < train_opt.csv_columns.size() || line_split_buff[0].empty())) {
    return -1;
  }
  
  Sample &sample = batch_samples[sample_idx];

  // parse label
  sample.label.i = line_split_buff[0][0] == '1' ? 1 : -1;

  // parse featrues
  size_t fm_node_idx = 0;
  for (const auto &feat_entrie : feat_entries) {
    const string & feat_str = line_split_buff[feat_entrie.first];
        feat_entrie.second->feedSample(feat_str.c_str(), feat_str.size(), sample.fm_layer_nodes[fm_node_idx++]);
  }
  sample.fm_layer_nodes_size = fm_node_idx;

  return sample.forward();
}

void BaseSolver::train(const string & line, int &y, real_t &logit, real_t & loss, real_t & grad) {
  (this->*lineProcessor)(line);

  batch_samples[sample_idx].backward();
  
  y = batch_samples[sample_idx].label.i;
  logit = batch_samples[sample_idx].logit;
  loss = batch_samples[sample_idx].loss;
  grad = batch_samples[sample_idx].grad;

  rotateSampleIdx();

  DEBUG_OUT << "BaseSolver::train "
            << " y " << y << " logit " << logit << endl;
}

void BaseSolver::test(const string & line, int &y, real_t &logit) {
  (this->*lineProcessor)(line);
  y = batch_samples[sample_idx].label.i;
  logit = batch_samples[sample_idx].logit;
}

#if 0 // 单个样本的sgdm, adam, ftrl参数更新

  void update_by_sgdm() {
    for (auto & param_node : backward_params) {
      real_t grad = param_node.grad;
      real_t xi = param_node.xi;

      SgdmParamUnit *backward_param = (SgdmParamUnit *)param_node.param;
      param_node.mutex->lock();

      real_t & w = backward_param->fm_param.w;
      real_t & wm = backward_param->momentum.w;

      wm = beta1 * wm + (1-beta1) * grad;
      w -= lr * (wm  + w * l2_reg_w);

      for (int f = 0; f < DIM; ++f) {
        real_t &vf = backward_param->fm_param.V[f];
        real_t & vmf = backward_param->momentum.V[f];

        real_t vgf = grad * (sum[f]  - vf * xi );

        vmf = beta1 * vmf + (1-beta1) * vgf;

        vf -= lr * (vmf + vf * l2_reg_V);
      }
      param_node.mutex->unlock();
    }
  }

  virtual void update_by_adam() {
    for (auto & param_node : backward_params) {
      real_t grad = param_node.grad;
      real_t xi = param_node.xi;

      AdamParamUnit *backward_param = (AdamParamUnit *)param_node.param;
      param_node.mutex->lock();
      // calc fixed_lr
      backward_param->beta1power_t *= beta1;
      backward_param->beta2power_t *= beta2;
      real_t bias_correction1 = (1 - backward_param->beta1power_t);
      real_t bias_correction2 = (1 - backward_param->beta2power_t);
      real_t fixed_lr = lr * std::sqrt(bias_correction2) / bias_correction1;

      // update w
      real_t & w = backward_param->fm_param.w;
      real_t & wm = backward_param->momentum.w;
      real_t & wv = backward_param->avg_squared.w;

      wm = beta1 * wm + (1-beta1)*grad;
      wv = beta2 * wv + (1-beta2)*grad*grad;

      DEBUG_OUT << "adam_solver: grad:" << grad << " w:" << w << " fixed_lr: " << fixed_lr
                << " wm:" << wm << " wv:" << wv << " update:"
                << fixed_lr * (wm / (std::sqrt(wv) + eps) + weight_decay_w * w) << endl
                << "fm_param: " << backward_param->fm_param.w << "," << backward_param->fm_param.V[0] << "," << backward_param->fm_param.V[1] << endl
                << "momentum: " << backward_param->momentum.w << "," << backward_param->momentum.V[0] << "," << backward_param->momentum.V[1] << endl
                << "avg_squared: " << backward_param->avg_squared.w << "," << backward_param->avg_squared.V[0] << "," << backward_param->avg_squared.V[1] << endl
                << "sum_0_1 " << sum[0] <<"," << sum[1] << endl
                << "fm_param.V_0_1 " << backward_param->fm_param.V[0] <<"," << backward_param->fm_param.V[1] << endl
                << "vgf_0 " << grad * (sum[0]  - backward_param->fm_param.V[0] * xi ) << endl
                << "vgf_1 " << grad * (sum[1]  - backward_param->fm_param.V[1] * xi ) << endl;

      w -= fixed_lr * (wm / (std::sqrt(wv) + eps) + weight_decay_w * w);

      // update V
      for (int f = 0; f < DIM; ++f) {

        real_t &vf = backward_param->fm_param.V[f];
        real_t &vmf = backward_param->momentum.V[f];
        real_t &vvf = backward_param->avg_squared.V[f];

        real_t vgf = grad * (sum[f]  - vf * xi );

        vmf = beta1 * vmf + (1 - beta1) * vgf;
        vvf = beta2 * vvf + (1 - beta2) * vgf * vgf;
        vf -= fixed_lr * (vmf / (std::sqrt(vvf) + eps) + weight_decay_V * vf);
      }
      param_node.mutex->unlock();
    }
  }

  void update_by_adam_raw(real_t grad) {
    for (auto & param_node : backward_params) {
      real_t grad = param_node.grad;
      real_t xi = param_node.xi;

      AdamParamUnit *backward_param = (AdamParamUnit *)param_node.param;
      param_node.mutex->lock();

      real_t & w = backward_param->fm_param.w;
      real_t & wm = backward_param->momentum.w;
      real_t & wv = backward_param->avg_squared.w;

      wm = beta1 * wm + (1-beta1)*grad;
      wv = beta2 * wv + (1-beta2)*grad*grad;

      real_t corrected_wm = wm;
      real_t corrected_wv = wv;
      if (bias_correct) {
        backward_param->beta1power_t *= beta1;
        backward_param->beta2power_t *= beta2;
        wm /= (1-backward_param->beta1power_t);
        wv /= (1-backward_param->beta2power_t);
      }
      
      w -= lr * (corrected_wm / (std::sqrt(corrected_wv) + eps) + weight_decay_w * w);

      for (int f = 0; f < DIM; ++f) {

        real_t &vf = backward_param->fm_param.V[f];
        real_t &vmf = backward_param->momentum.V[f];
        real_t &vvf = backward_param->avg_squared.V[f];

        real_t vgf = grad * (sum[f]  - vf * xi );

        vmf = beta1 * vmf + (1 - beta1) * vgf;
        vvf = beta2 * vvf + (1 - beta2) * vgf * vgf;

        real_t corrected_vmf = bias_correct ? vmf : (vmf / (1 - beta1_pow));
        real_t corrected_vvf = bias_correct ? vvf : (vvf / (1 - beta2_pow));

        vf -= lr * (corrected_vmf /
               (std::sqrt(corrected_vvf) + eps)  + weight_decay_V * vf);
      }

      param_node.mutex->unlock();
    }
  }


  void update_by_ftrl() {
    for (auto & param_node : backward_params) {
      real_t grad = param_node.grad;
      real_t xi = param_node.xi;

      FtrlParamUnit *backward_param = (FtrlParamUnit *)param_node.param;
      param_node.mutex->lock();
      real_t w_sigama =
          1 / train_opt.ftrl.w_alpha *
          (std::sqrt(backward_param->n.w + grad * grad) - std::sqrt(backward_param->n.w));

      backward_param->z.w += grad - w_sigama * backward_param->fm_param.w;
      backward_param->n.w += grad * grad;

      for (int f = 0; f < DIM; ++f) {
        real_t vgf = grad * (sum[f]  - backward_param->fm_param.V[f] * xi);
        real_t v_sigma_f =
            1 / train_opt.ftrl.v_alpha * (std::sqrt(backward_param->n.V[f] + vgf * vgf) - std::sqrt(backward_param->n.V[f]));

        backward_param->z.V[f] += vgf - v_sigma_f * backward_param->fm_param.V[f];
        backward_param->n.V[f] += vgf * vgf;
      }

      backward_param->calcFmWeights();

      param_node.mutex->unlock();
    }
  }

#endif
