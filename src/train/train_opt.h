/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include <sstream>

#include "utils/base.h"
#include "utils/utils.h"
#include "utils/args_parser.h"

class TrainOption;
extern TrainOption train_opt;


class TrainOption {
 public:
  TrainOption() {}
  ~TrainOption() {}

  const char* config_file_path = "config/train.conf";

  bool parse_cfg_and_cmdlines(int argc, char* argv[]);

 public:
  std::string train_path;
  std::string valid_path;
  std::string feature_config_path;
  std::string model_path;
  std::string model_format;
  std::string init_model_path;
  std::string model_number_type;
  int epoch;

  enum BatchGradReduceType {
    BatchGradReduceType_Sum,
    BatchGradReduceType_AvgByBatchSize,
    BatchGradReduceType_AvgByOccurrences,
    BatchGradReduceType_AvgByOccurrencesSqrt    
  };

  static const BatchGradReduceType batch_grad_reduce_type = BatchGradReduceType_AvgByOccurrencesSqrt;

  int threads_num;
  int time_interval_of_validation;
  static const long n_sample_per_output = 50000;
  static const int task_queue_size = 5000;
  // const int shulf_window_size = 10007;
  // const bool shuffle = false;

  int verbose;
  bool print_help;
  bool disable_feaid_mapping;

  int batch_size;

  // train data format
  char fea_seperator;
  char fea_kv_seperator;
  char fea_multivalue_seperator;
  char fea_id_mapping_dict_seperator;

  // params for feature_configs
  const string fea_type_dense = "dense_features";
  const string fea_type_sparse = "sparse_features";
  const string fea_type_varlen_sparse = "varlen_sparse_features";

  ///////////////////////////////////////////////////////////
  // FTRL solver
  std::string solver;

  ///////////////////////////////////////////////////////////
  // adam params
  struct AdamParam {
    real_t lr;
    int bias_correct; // 默认false
    real_t beta1;
    real_t beta2;
    real_t weight_decay_w; // 设置weight_decay，则为AdamW。对于adam，宜用weight_decay，不宜用l2正则
    real_t weight_decay_V; // 设置weight_decay，则为AdamW。对于adam，宜用weight_decay，不宜用l2正则
    int amsgrad; // 取值0或1 TODO 暂时未实现
    static constexpr real_t eps = 1e-8;
    static constexpr real_t tolerance = 1e-5;
    static constexpr bool resetPolicy = true;
    static constexpr bool exactObjective = false;
  } adam;

  ///////////////////////////////////////////////////////////
  // SGDM params (SGD with Momentum)
  struct SgdmParam {
    real_t lr;
    real_t beta1;
    real_t l1_reg_w;
    real_t l2_reg_w;
    real_t l1_reg_V;
    real_t l2_reg_V;
  } sgdm;

  ///////////////////////////////////////////////////////////
  // FTRL params
  struct FtrlParam {
    real_t init_stdev;
    real_t w_alpha;
    real_t w_beta;
    real_t v_alpha;
    real_t v_beta;
    real_t l1_reg_w;
    real_t l2_reg_w;
    real_t l1_reg_V;
    real_t l2_reg_V;
  } ftrl;

private:
  char parse_seperator_chars(const char* param) const {
    if (0 == strcmp(param, "blank"))
      return ' ';
    else if (0 == strcmp(param, "tab"))
      return '\t';
    else if (0 == strcmp(param, "equal"))
      return '=';

    return param[0];
  }
  ArgsParser arg_parser;
};
