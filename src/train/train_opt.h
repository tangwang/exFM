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
  std::string initial_model_format;
  std::string model_number_type;
  int epoch;

  int factor_num;

  int threads_num;
  int time_interval_of_validation;
  const long n_sample_per_output = 500000;
  const int task_queue_size = 5000;

  int verbose;
  bool print_help;
  bool disable_feaid_mapping;

  const bool shuffle = false; // TODO  shuffle is not implement yet
  const size_t batchSize = 32;// TODO  batchSize is not implement yet

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
  // FTRL params
  struct FtrlParam {
    const real_t init_mean = 0.0;
    real_t init_stdev;
    real_t w_alpha;
    real_t w_beta;
    real_t l1_reg_w;
    real_t l2_reg_w;
    real_t v_alpha;
    real_t v_beta;
    real_t l1_reg_V;
    real_t l2_reg_V;
  } ftrl;

  ///////////////////////////////////////////////////////////
  // adam params
  struct AdamParam {
    real_t step_size;
    real_t beta1;
    real_t beta2;
    const real_t eps = 1e-8;
    const real_t tolerance = 1e-5;
    const bool resetPolicy = true;
    const bool exactObjective = false;
  } adam;

  ///////////////////////////////////////////////////////////
  // SGD params
  struct SgdParam {
    real_t step_size;
  } sgd;

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
