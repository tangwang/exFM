/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "ftrl/train_opt.h"

TrainOption train_opt;

bool TrainOption::parse_args(int argc, char *argv[]) {
  program_name = argv[0];
  char keyname_buff[64];
  for (int i = 1; i < argc; ++i) {
    const char *value = strchr(argv[i], '=');
    if (value != NULL) {
      // is argument
      snprintf(keyname_buff, std::min(sizeof(keyname_buff), (size_t)(value - argv[i] + 1)), argv[i]);
      printf("parsed_args : key<%s>, v<%s>\n", keyname_buff, value+1);
      args.push_back(make_pair(string(keyname_buff), value+1));
    } else {
      // is option
      options.push_back(argv[i]);
    }
  }

  bool ret = true;
  help_message << "\ntrainning args : \n\n";

  ret &= parse_arg("train", train_path, "",
                   "trainning data path, use stdin(standard input) if not set");
  ret &= parse_arg("eval", eval_path, "", "eval data path");
  ret &= parse_arg("time_interval_of_validation", time_interval_of_validation, 60, "how many seconds between two validition");
  ret &= parse_arg("cfg", feature_config_path, "", "feature_config_path", true);
  ret &= parse_arg("dim", factor_num, 8, "feature vector dim(factor_num)");
  ret &= parse_arg("im", init_model_path, "", "init model path");
  ret &= parse_arg("imf", initial_model_format, "txt", "input model format");
  ret &= parse_arg("om", model_path, "", "output model path");
  ret &= parse_arg("omf", model_format, "txt", "output model format");
  ret &= parse_arg("threads_num", threads_num, 11, "trainning threads_num");

  const char *temp_split_param = NULL;
  ret &= parse_arg(
      "fea_spliter", temp_split_param, "\t",
      "speciafy one character for fea_spliter in line of sample. "
      "because blank chars(blank or tab) cannot passed from cmdline args, so "
      "if you want specify blank char, set \"blank\" or \"tab\" is ok");
  fea_spliter = parse_spliter_chars(temp_split_param);

  ret &= parse_arg(
      "fea_kv_spliter", temp_split_param, ":",
      "speciafy one character for fea_kv_spliter in line of sample. "
      "because blank chars(blank or tab) and '=' cannot passed from cmdline "
      "args, so if you want specify blank char, set \"equal\" or \"blank\" or "
      "\"tab\" is ok");
  fea_kv_spliter = parse_spliter_chars(temp_split_param);

  ret &= parse_arg(
      "fea_multivalue_spliter", temp_split_param, ",",
      "speciafy one character for fea_multivalue_spliter in line of sample. "
      "because blank chars(blank or tab) cannot passed from cmdline args, so "
      "if you want specify blank char, set \"blank\" or \"tab\" is ok");
  fea_multivalue_spliter = parse_spliter_chars(temp_split_param);

  help_message << "\n\nfeature id mapping settings\n\n";

  disable_feaid_mapping = parse_option("disable_feaid_mapping", "ignore feature ID mapping dicts, using hash or original ID.");

  ret &= parse_arg(
      "fea_id_mapping_dict_seperator", temp_split_param, " ",
      "speciafy one character for fea_id_mapping_dict_seperator in line of sample. "
      "because blank chars(blank or tab) cannot passed from cmdline args, so "
      "if you want specify blank char, set \"blank\" or \"tab\" is ok");
  fea_id_mapping_dict_seperator = parse_spliter_chars(temp_split_param);


  help_message << "\n\ntrainning hyper-paramerters : \n\n";

  ret &= parse_arg("init_mean", init_mean, 0.0, "");
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

  print_help = parse_option("h", "print help message");

  if (!ret || print_help) {
    std::cerr
        << "\n\nexFM -- Flexible(support various feature forms) and "
           "high-performance(training and online serving) FM implementation. \n"
           "usage : "
        << program_name
        << " arg1=value1 arg2=value2 ...  opt1 opt2 ... \n";
    std::cerr << help_message.str() << endl;
  }
  return ret;
}
