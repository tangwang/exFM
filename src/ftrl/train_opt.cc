/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "ftrl/train_opt.h"

TrainOption train_opt;

bool TrainOption::parse_args(int argc, char *argv[]) {
  // load config file
  if (access(config_file_path, F_OK) == 0) {
    ifstream config_file(config_file_path);
    if (config_file.is_open()) {
      string temp_str;
      while (std::getline(config_file, temp_str)) {
        utils::trim(temp_str);
        if (!temp_str.empty()) arg_lines.push_back(temp_str);
      }
    }
  }

  program_name = argv[0];
  for (int i = 1; i < argc; ++i) {
    arg_lines.push_back(string(argv[i]));
  }

  vector<string> spliter_values;
  for (const string &arg_line : arg_lines) {
    const char *value = strchr(arg_line.c_str(), '=');
    if (value != NULL) {
      // is argument
      spliter_values.clear();
      utils::split_string(arg_line, '=', spliter_values);
      if (spliter_values.size() == 2) {
        args.push_back(Arg(spliter_values[0], spliter_values[1]));
      }
    } else {
      // is option
      args.push_back(Arg(arg_line, ""));
    }
  }

  bool ret = true;
  // 训练参数相关配置
  help_message << "\ntrainning args : \n\n";

  ret &= parse_arg("fea_cfg", feature_config_path, string(),
                   "feature_config_path", true);
  ret &= parse_arg("train", train_path, string(),
                   "trainning data path, use stdin(standard input) if not set");
  ret &= parse_arg("valid", valid_path, string(), "validation data path");
  ret &= parse_arg("time_interval_of_validation", time_interval_of_validation,
                   60, "how many seconds between two validition");
  ret &= parse_arg("dim", factor_num, 8, "feature vector dim(factor_num)");
  ret &= parse_arg("im", init_model_path, string(), "init model path");
  ret &= parse_arg("imf", initial_model_format, string("txt"),
                   "input model format");
  ret &= parse_arg("om", model_path, string(), "output model path");
  ret &= parse_arg("omf", model_format, string("txt"), "output model format");
  ret &= parse_arg("threads", threads_num, 11, "trainning threads_num");

  const char *temp_split_param = NULL;
  ret &= parse_arg("fea_sep", temp_split_param, "\t",
                   "specify one character (or str \"blank\",\"tab\") for "
                   "fea_seperator in line of sample. ");
  fea_seperator = parse_seperator_chars(temp_split_param);

  ret &= parse_arg(
      "fea_kv_sep", temp_split_param, ":",
      "specify one character (or str \"blank\",\"tab\", \"eqaul\"(for \"=\"))  "
      "for fea_kv_seperator in line of sample. ");
  fea_kv_seperator = parse_seperator_chars(temp_split_param);

  ret &= parse_arg("fea_values_sep", temp_split_param, ",",
                   "specify one character(or str \"blank\",\"tab\")  for "
                   "fea_multivalue_seperator in line of sample. ");
  fea_multivalue_seperator = parse_seperator_chars(temp_split_param);

  // ID映射词典相关配置
  help_message << "\n\nfeature id mapping settings\n\n";
  disable_feaid_mapping = parse_option(
      "disable_feaid_mapping",
      "ignore feature ID mapping dicts, using hash or original ID.");
  ret &= parse_arg("id_map_dict_sep", temp_split_param, " ",
                   "specify one character(or str \"blank\",\"tab\") for "
                   "fea_id_mapping_dict_seperator in line of sample. ");
  fea_id_mapping_dict_seperator = parse_seperator_chars(temp_split_param);

  help_message << "\n\ntrainning hyper-paramerters : \n\n";

  // FTRL hyper params
  ret &= parse_arg("init_stdev", init_stdev, 0.1,
                   "stdev for initialization of 2-way factors");
  ret &= parse_arg("w_alpha", w_alpha, 0.05, "FTRL hyper-param");
  ret &= parse_arg("w_beta", w_beta, 1.0, "FTRL hyper-param");
  ret &= parse_arg("l1_reg_w", l1_reg_w, 0.1, "FTRL hyper-param");
  ret &= parse_arg("l2_reg_w", l2_reg_w, 5.0, "FTRL hyper-param");
  ret &= parse_arg("v_alpha", v_alpha, 0.05, "FTRL hyper-param");
  ret &= parse_arg("v_beta", v_beta, 1.0, "FTRL hyper-param");
  ret &= parse_arg("l1_reg_V", l1_reg_V, 0.1, "FTRL hyper-param");
  ret &= parse_arg("l2_reg_V", l2_reg_V, 5.0, "FTRL hyper-param");
  ret &= parse_arg("verbose", verbose, 1,
                   "0: only trainning logs. 1: open feature config messages, 2 "
                   ": open debug messages");

  print_help = parse_option("h", "print help message");

  for (auto &arg : args) {
    if (arg.process_stat == 0) {
      std::cerr << console_color::red << "unknown args " << arg.k
                << console_color::reset << std::endl;

      ret = false;
    }
  }
  if (!ret || print_help) {
    std::cerr
        << "\n\nexFM -- Flexible(support various feature forms) and "
           "high-performance(training and online serving) FM implementation. \n"
           "usage : "
        << program_name << " arg1=value1 arg2=value2 ...  opt1 opt2 ... \n";
    std::cerr << help_message.str() << endl;
  }
  return ret;
}
