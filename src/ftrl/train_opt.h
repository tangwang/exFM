/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include <sstream>

#include "utils/base.h"
#include "utils/utils.h"

class TrainOption;
extern TrainOption train_opt;

struct Arg {
 public:
  Arg(std::string _k, std::string _v) : k(_k), v(_v), process_stat(0) {}
  std::string k;
  std::string v;
  int process_stat;  // 0 : not processed, unknown arg , 1 : processed ok, 2 :
                     // bad values
};

class TrainOption {
 public:
  TrainOption() {}
  ~TrainOption() {}

  bool parse_args(int argc, char* argv[]);

  const char* config_file_path = "config/train.conf";

 public:
  std::string train_path;
  std::string valid_path;
  std::string feature_config_path;
  std::string model_path;
  std::string model_format;
  std::string init_model_path;
  std::string initial_model_format;
  std::string model_number_type;

  const double init_mean = 0.0;
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

  int verbose;
  bool print_help;
  bool disable_feaid_mapping;

  // train data format
  char fea_seperator;
  char fea_kv_seperator;
  char fea_multivalue_seperator;
  char fea_id_mapping_dict_seperator;

  // params for feature_configs
  const string fea_type_dense = "dense_features";
  const string fea_type_sparse = "sparse_features";
  const string fea_type_varlen_sparse = "varlen_sparse_features";

 private:
  template <typename value_type>
  bool parse_arg(const char* key, value_type& v, value_type default_value,
                 const char* helper, bool necessary = false) {
    using namespace std;
    help_message << "arg " << setiosflags(ios::left) << console_color::blue
                 << setw(20) << key << console_color::reset << " default_value "
                 << console_color::blue << setw(5) << default_value
                 << console_color::reset << " necessary " << console_color::blue
                 << setw(1) << necessary << console_color::reset
                 << " helper: " << console_color::blue << helper
                 << console_color::reset << endl;
    for (auto& arg : args) {
      if (arg.k == key) {
        v = utils::cast_type<const char*, value_type>(arg.v.c_str());
        arg.process_stat = 1;
        return true;
      }
    }
    v = default_value;
    if (necessary) {
      help_message << console_color::red << "ERROR! Missing necessary arg:  <"
                   << key << "> " << console_color::reset << endl;
    }
    return !necessary;
  }

  char parse_seperator_chars(const char* param) const {
    if (0 == strcmp(param, "blank"))
      return ' ';
    else if (0 == strcmp(param, "tab"))
      return '\t';
    else if (0 == strcmp(param, "equal"))
      return '=';

    return param[0];
  }

  bool parse_option(const char* key, const char* helper) {
    using namespace std;
    help_message << "opt " << setiosflags(ios::left) << console_color::blue
                 << setw(20) << key << console_color::reset
                 << " helper: " << console_color::blue << helper
                 << console_color::reset << std::endl;
    for (auto& arg : args) {
      if (arg.k == key) {
        arg.process_stat = 1;
        if (!arg.v.empty()) {
          help_message
              << console_color::red << key
              << " is an option, the value your specified cannot be recognized"
              << console_color::reset << std::endl;
        }
        return true;
      }
    }
    return false;
  }

  vector<string> arg_lines;

  const char* program_name;
  std::vector<Arg> args;
  std::vector<std::string> options;
  std::stringstream help_message;
};
