/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "train/train_opt.h"

TrainOption train_opt;

bool TrainOption::parse_cfg_and_cmdlines(int argc, char *argv[]) {
  // load config file

  arg_parser.loadArgs(config_file_path);
  arg_parser.loadArgs(argc, argv);

  arg_parser.setVerbose(false);
  arg_parser.parse_arg("verbose", verbose, 1,
                  "0: only trainning logs. 1: open feature config messages, 2 "
                  ": open debug messages");
  arg_parser.setVerbose(verbose > 0);

  // 训练参数相关配置
  string feat_cfg;
  arg_parser.parse_arg("feat_cfg", feat_cfg, string(),
                   "feature config name(dir) under config dir", true);
  
  feature_config_path = string("config/") + feat_cfg + "/feature_config.json";
  mapping_dict_path = string("config/") + feat_cfg + "/feat_id_dict/";

  arg_parser.parse_arg("train", train_path, string(), 
                   "trainning data path, use stdin(standard input) if not set");
  arg_parser.parse_arg("valid", valid_path, string(), "validation data path");
  arg_parser.parse_arg("valid_interval", time_interval_of_validation,
                   60, "how many seconds between two validition");

  arg_parser.parse_arg("csv_columns", csv_columns, string(),
                   "set csv_columns when your data_format is csv, and your csv file had no header line", false);

  string str_data_formart;
  arg_parser.parse_arg(
      "data_formart", str_data_formart, string(),
      "train and validation formart, support libSVM / CSV",
      true);
  if (0 == strcasecmp(str_data_formart.c_str(), "libSVM")) {
    data_formart = DataFormart_libSVM;
  } else if (0 == strcasecmp(str_data_formart.c_str(), "CSV")) {
    data_formart = DataFormart_CSV;
  } else {
    cerr << "arg data_formart must be libSVM / CSV" << endl;
    return false;
  }

  arg_parser.parse_arg("epoch", epoch, 1, "train epochs");

  arg_parser.parse_arg("im", init_model_path, string(), "init model path");
  arg_parser.parse_arg("om", model_path, string(), "output model path");
  arg_parser.parse_arg("mf", model_format, string("txt"), "bin/txt . output model format");
  arg_parser.parse_arg("threads", threads_num, 11, "trainning threads_num");

  const char *temp_split_param = NULL;
  arg_parser.parse_arg("feat_sep", temp_split_param, "blank",
                   "specify one character (or str \"blank\",\"tab\") for "
                   "feat_seperator in line of sample. ");
  feat_seperator = parse_seperator_chars(temp_split_param);

  arg_parser.parse_arg("feat_values_sep", temp_split_param, ",",
                   "specify one character(or str \"blank\",\"tab\")  for "
                   "feat_value_list_seperator in line of sample. ");
  feat_value_list_seperator = parse_seperator_chars(temp_split_param);

  // ID映射词典相关配置
  arg_parser.parse_arg("id_map_dict_sep", temp_split_param, " ",
                   "specify one character(or str \"blank\",\"tab\") for k_v_seperator in line of feature mapping dict");
  feat_id_dict_seperator = parse_seperator_chars(temp_split_param);

  // params initallization
  arg_parser.parse_arg("init_stdev", init_stdev, 0.001,
                  "stdev for initialization of 2-way factors");

//////////////////////
// solver
  arg_parser.parse_arg("solver", solver, string("ftrl"), "solver", false);
  arg_parser.parse_arg("batch_size", batch_size, 1024, "solver", false);

  // SGDM hyper params
  arg_parser.parse_arg("sgdm.lr", sgdm.lr, 0.001, "SGDM learning rate");
  arg_parser.parse_arg("sgdm.beta1"   , sgdm.beta1,    0.9   , "SGD 一阶动量平滑常数");
  arg_parser.parse_arg("sgdm.l1w", sgdm.l1_reg_w, 0.05, "l1 regularization of w");
  arg_parser.parse_arg("sgdm.l2w", sgdm.l2_reg_w, 5.0,  "l2 regularization of w");
  arg_parser.parse_arg("sgdm.l1v", sgdm.l1_reg_V, 0.05, "l1 regularization of V");
  arg_parser.parse_arg("sgdm.l2v", sgdm.l2_reg_V, 5.0,  "l2 regularization of V");

  // adagrad hyper params
  arg_parser.parse_arg("adagrad.lr", adagrad.lr, 0.01, "Adam learning rate");
  arg_parser.parse_arg("adagrad.l2_norm_w", adagrad.l2_norm_w, 1e-5, "l2 norm for w");
  arg_parser.parse_arg("adagrad.l2_norm_V", adagrad.l2_norm_V, 1e-5, "l2 norm for embeddings");

  // rmsprop hyper params
  arg_parser.parse_arg("rmsprop.lr", rmsprop.lr, 0.01, "Adam learning rate");
  arg_parser.parse_arg("rmsprop.l2_norm_w", rmsprop.l2_norm_w, 1e-5, "l2 norm for w");
  arg_parser.parse_arg("rmsprop.l2_norm_V", rmsprop.l2_norm_V, 1e-5, "l2 norm for embeddings");
  arg_parser.parse_arg("rmsprop.beta2", rmsprop.beta2, 0.9, "二阶动量滑动平均");

  // Adam hyper params
  arg_parser.parse_arg("adam.lr", adam.lr, 0.001, "Adam learning rate");
  arg_parser.parse_arg("adam.beta1", adam.beta1, 0.9, "adam一阶动量平滑常数");
  arg_parser.parse_arg("adam.beta2", adam.beta2, 0.999, "adam二阶动量平滑常数");
  arg_parser.parse_arg("adam.weight_decay_w", adam.weight_decay_w, 0.1, "l2正则在adam中的实现。对于adam，宜用weight_decay，不宜用l2正则");
  arg_parser.parse_arg("adam.weight_decay_V", adam.weight_decay_V, 0.1, "l2正则在adam中的实现。对于adam，宜用weight_decay，不宜用l2正则");

  // FTRL hyper params
  arg_parser.parse_arg("ftrl.w_alpha", ftrl.w_alpha, 0.01, "FTRL hyper-param");
  arg_parser.parse_arg("ftrl.w_beta", ftrl.w_beta, 1.0, "FTRL hyper-param");
  arg_parser.parse_arg("ftrl.v_alpha", ftrl.v_alpha, 0.01, "FTRL hyper-param");
  arg_parser.parse_arg("ftrl.v_beta", ftrl.v_beta, 1.0, "FTRL hyper-param");
  arg_parser.parse_arg("ftrl.l1w", ftrl.l1_reg_w, 0.05, "l1 regularization of w");
  arg_parser.parse_arg("ftrl.l2w", ftrl.l2_reg_w, 5.0,  "l2 regularization of w");
  arg_parser.parse_arg("ftrl.l1v", ftrl.l1_reg_V, 0.05, "l1 regularization of V");
  arg_parser.parse_arg("ftrl.l2v", ftrl.l2_reg_V, 5.0,  "l2 regularization of V");

  print_help = arg_parser.parse_option("h", "print help message");

  arg_parser.process_other_args();

  if (!arg_parser.parse_status || print_help) {
    arg_parser.print_helper();
  }
  return arg_parser.parse_status;
}

