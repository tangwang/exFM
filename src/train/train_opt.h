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
  std::string mapping_dict_path;
  std::string model_path;
  std::string model_format;
  std::string init_model_path;
  std::string model_number_type;
  int epoch;

  enum BatchGradReduceType {
    BatchGradReduceType_AvgByBatchSize,
    BatchGradReduceType_Sum,
    BatchGradReduceType_AvgByOccurrences,
    BatchGradReduceType_AvgByOccurrencesSqrt
  };

  static constexpr BatchGradReduceType batch_grad_reduce_type = BatchGradReduceType_AvgByBatchSize;

  enum DataFormart {
    DataFormart_CSV,
    DataFormart_libSVM
  };
  DataFormart data_formart = DataFormart_libSVM;
  vector<string> csv_columns;


  int threads_num;
  int time_interval_of_validation;
  static constexpr long n_sample_per_output = 1000000;
  static constexpr int task_queue_size = 5000;
  // constexpr int shulf_window_size = 10007;
  // constexpr bool shuffle = false;

  int verbose;
  bool print_help;

  int batch_size;

  // train data format
  char feat_seperator;
  static constexpr char feat_kv_seperator = '=';
  char feat_value_list_seperator;
  char feat_id_dict_seperator;

  // params for feature_configs
  const string  feat_type_dense = "dense_features";
  const string  feat_type_sparse = "sparse_features";
  const string  feat_type_varlen_sparse = "varlen_sparse_features";

  // param initial
  real_t init_stdev;

  // solver
  std::string solver;

  // adam params
  struct AdamParam {
    real_t lr;
    real_t beta1;
    real_t beta2;
    real_t weight_decay_w; // 设置weight_decay，则为AdamW。对于adam，宜用weight_decay，不宜用l2正则
    real_t weight_decay_V; // 设置weight_decay，则为AdamW。对于adam，宜用weight_decay，不宜用l2正则
  } adam;

  // adagrad params
  struct AdagradParam {
    real_t lr;
    real_t l2_norm_w;
    real_t l2_norm_V;
  } adagrad;

  // RMSProp / adadelta params，带有二阶动量滑动平均的adagrad
  struct RmspropParam {
    real_t lr;
    real_t l2_norm_w;
    real_t l2_norm_V;
    real_t beta2; //  二阶动量滑动平均
  } rmsprop;

  // SGDM params (SGD with Momentum)
  struct SgdmParam {
    real_t lr;
    real_t beta1;
    real_t l1_reg_w;
    real_t l2_reg_w;
    real_t l1_reg_V;
    real_t l2_reg_V;
  } sgdm;

  // FTRL params
  struct FtrlParam {
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
    if (0 == strcasecmp(param, "blank"))
      return ' ';
    else if (0 == strcasecmp(param, "tab"))
      return '\t';
    else if (0 == strcasecmp(param, "equal"))
      return '=';

    return param[0];
  }
  ArgsParser arg_parser;
};
