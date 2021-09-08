/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include <sstream>
#include "utils/base.h"
#include "utils/utils.h"

class TrainOption;
extern TrainOption train_opt;

class TrainOption {
 public:
  TrainOption() {}
  ~TrainOption() {}

  bool parse_args(int argc, char* argv[]);

 public:
  const char* train_path;
  const char* eval_path;
  const char* feature_config_path;
  const char* model_path;
  const char* model_format;
  const char* init_model_path;
  const char* initial_model_format;
  const char* model_number_type;

  double init_mean;
  double init_stdev;

  double w_alpha;
  double w_beta;
  double l1_reg_w;
  double l2_reg_w;

  double v_alpha;
  double v_beta;
  double l1_reg_V;
  double l2_reg_V;

  int factor_num;

  int threads_num;
  int time_interval_of_validation;
  const long n_sample_per_output = 500000;
  const int task_queue_size = 5000;

  bool print_help;
  bool disable_feaid_mapping;

  // train data format
  char fea_spliter;
  char fea_kv_spliter;
  char fea_multivalue_spliter;
  char fea_id_mapping_dict_seperator;

  // params for feature_configs
  const string fea_type_dense = "dense_features";
  const string fea_type_sparse = "sparse_features";
  const string fea_type_varlen_sparse = "varlen_sparse_features";

 private:
  template <typename value_type>
  bool parse_arg(const char* key, value_type& v, value_type default_value,
                 const char* helper, bool necessary = false) {
    help_message << "arg <" << key << "> default_value <" << default_value
                 << "> necessary <" << necessary << "> helper: " << helper
                 << std::endl;
    for (auto& arg : args) {
      if (arg.first == key) {
        v = utils::cast_type<const char*, value_type>(arg.second);
        return true;
      }
    }
    v = default_value;
    if (necessary) {
      help_message << "ERROR! Missing necessary arg:  <" << key << "> " << std::endl;
    }
    return !necessary;
  }

  char parse_spliter_chars(const char* param) const {
    if (0 == strcmp(param, "blank"))
      return ' ';
    else if (0 == strcmp(param, "tab"))
      return '\t';
    else if (0 == strcmp(param, "equal"))
      return '=';

    return param[0];
  }

  bool parse_option(const char* key, const char* helper) {
    help_message << "opt <" << key << "> helper: " << helper << std::endl;
    for (auto& opt : options) {
      if (0 == strcmp(opt, key)) {
        return true;
      }
    }
    return false;
  }

  const char* program_name;
  std::vector<std::pair<std::string, const char*>> args;
  std::vector<const char*> options;
  std::stringstream help_message;
};
