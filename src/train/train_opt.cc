/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "train/train_opt.h"

TrainOption train_opt;
bool TrainOption::parse_cfg_and_cmdlines(int argc, char *argv[]) {
  // load config file

  int print_debug_info = 1; // 调试开关

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
  
  // 所有的路径，如果解析后得到的路径是绝对路径，那么使用绝对路径开头而不是自己拼接相对路径
  if (feat_cfg[0] == '/') {
    if (print_debug_info) {
      std::cout << "Using absolute path for feature config: " << feat_cfg << std::endl;
    }
    feature_config_path = feat_cfg + "/feature_config.json";
    mapping_dict_path = feat_cfg + "/feat_id_dict/";
  } else {
    if (print_debug_info) {
      std::cout << "Using relative path for feature config: " << feat_cfg << std::endl;
    }
    feature_config_path = string("config/") + feat_cfg + "/feature_config.json";
    mapping_dict_path = string("config/") + feat_cfg + "/feat_id_dict/";
  }

  if (print_debug_info) {
    std::cout << "Feature config path: " << feature_config_path << std::endl;
    std::cout << "Mapping dict path: " << mapping_dict_path << std::endl;
  }

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

  if (print_debug_info) {
    std::cout << "Data format: " << str_data_formart << std::endl;
  }

  arg_parser.parse_arg("epoch", epoch, 1, "train epochs");

  arg_parser.parse_arg("im", init_model_path, string(), "init model path");
  arg_parser.parse_arg("om", model_path, string(), "output model path");
  arg_parser.parse_arg("mf", model_format, string("txt"), "bin/txt . output model format");
  arg_parser.parse_arg("threads", threads_num, 11, "trainning threads_num");

  if (print_debug_info) {
    std::cout << "Epoch: " << epoch << std::endl;
    std::cout << "Init model path: " << init_model_path << std::endl;
    std::cout << "Output model path: " << model_path << std::endl;
    std::cout << "Model format: " << model_format << std::endl;
    std::cout << "Threads num: " << threads_num << std::endl;
  }

  const char *temp_split_param = NULL;
  arg_parser.parse_arg("feat_sep", temp_split_param, "blank",
                   "specify one character (or str \"blank\",\"tab\") for "
                   "feat_seperator in line of sample. ");
  feat_seperator = parse_seperator_chars(temp_split_param);

  arg_parser.parse_arg("feat_kv_sep", temp_split_param, ":",
                   "specify one character (or str \"equal\") for "
                   "feat_kv_seperator in line of sample. ");
  feat_kv_seperator = parse_seperator_chars(temp_split_param);

  arg_parser.parse_arg("feat_values_sep", temp_split_param, ",",
                   "specify one character(or str \"blank\",\"tab\")  for "
                   "feat_value_list_seperator in line of sample. ");
  feat_value_list_seperator = parse_seperator_chars(temp_split_param);

  if (print_debug_info) {
    std::cout << "Feature separator: " << feat_seperator << std::endl;
    std::cout << "Feature key-value separator: " << feat_kv_seperator << std::endl;
    std::cout << "Feature value list separator: " << feat_value_list_seperator << std::endl;
  }

  // ID映射词典相关配置
  arg_parser.parse_arg("id_map_dict_sep", temp_split_param, " ",
                   "specify one character(or str \"blank\",\"tab\") for k_v_seperator in line of feature mapping dict");
  feat_id_dict_seperator = parse_seperator_chars(temp_split_param);

  if (print_debug_info) {
    std::cout << "Feature ID dict separator: " << feat_id_dict_seperator << std::endl;
  }

  // params initallization
  arg_parser.parse_arg("init_stdev", init_stdev, 0.001,
                  "stdev for initialization of 2-way factors");

  if (print_debug_info) {
    std::cout << "Init stdev: " << init_stdev << std::endl;
  }

//////////////////////
// solver
  arg_parser.parse_arg("solver", solver, string("ftrl"), "solver", false);
  arg_parser.parse_arg("batch_size", batch_size, 1024, "solver", false);

  if (print_debug_info) {
    std::cout << "Solver: " << solver << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
  }

  // SGDM hyper params
  arg_parser.parse_arg("sgdm.lr", sgdm.lr, 0.001, "SGDM learning rate");
  arg_parser.parse_arg("sgdm.beta1"   , sgdm.beta1,    0.9   , "SGD 一阶动量平滑常数");
  arg_parser.parse_arg("sgdm.l1w", sgdm.l1_reg_w, 0.05, "l1 regularization of w");
  arg_parser.parse_arg("sgdm.l2w", sgdm.l2_reg_w, 5.0,  "l2 regularization of w");
  arg_parser.parse_arg("sgdm.l1v", sgdm.l1_reg_V, 0.05, "l1 regularization of V");
  arg_parser.parse_arg("sgdm.l2v", sgdm.l2_reg_V, 5.0,  "l2 regularization of V");

  if (print_debug_info) {
    std::cout << "SGDM learning rate: " << sgdm.lr << std::endl;
    std::cout << "SGDM beta1: " << sgdm.beta1 << std::endl;
    std::cout << "SGDM l1 regularization of w: " << sgdm.l1_reg_w << std::endl;
    std::cout << "SGDM l2 regularization of w: " << sgdm.l2_reg_w << std::endl;
    std::cout << "SGDM l1 regularization of V: " << sgdm.l1_reg_V << std::endl;
    std::cout << "SGDM l2 regularization of V: " << sgdm.l2_reg_V << std::endl;
  }

  // adagrad hyper params
  arg_parser.parse_arg("adagrad.lr", adagrad.lr, 0.01, "Adam learning rate");
  arg_parser.parse_arg("adagrad.l2_norm_w", adagrad.l2_norm_w, 1e-5, "l2 norm for w");
  arg_parser.parse_arg("adagrad.l2_norm_V", adagrad.l2_norm_V, 1e-5, "l2 norm for embeddings");

  if (print_debug_info) {
    std::cout << "Adagrad learning rate: " << adagrad.lr << std::endl;
    std::cout << "Adagrad l2 norm for w: " << adagrad.l2_norm_w << std::endl;
    std::cout << "Adagrad l2 norm for embeddings: " << adagrad.l2_norm_V << std::endl;
  }

  // rmsprop hyper params
  arg_parser.parse_arg("rmsprop.lr", rmsprop.lr, 0.01, "Adam learning rate");
  arg_parser.parse_arg("rmsprop.l2_norm_w", rmsprop.l2_norm_w, 1e-5, "l2 norm for w");
  arg_parser.parse_arg("rmsprop.l2_norm_V", rmsprop.l2_norm_V, 1e-5, "l2 norm for embeddings");
  arg_parser.parse_arg("rmsprop.beta2", rmsprop.beta2, 0.9, "二阶动量滑动平均");

  if (print_debug_info) {
    std::cout << "RMSProp learning rate: " << rmsprop.lr << std::endl;
    std::cout << "RMSProp l2 norm for w: " << rmsprop.l2_norm_w << std::endl;
    std::cout << "RMSProp l2 norm for embeddings: " << rmsprop.l2_norm_V << std::endl;
    std::cout << "RMSProp beta2: " << rmsprop.beta2 << std::endl;
  }

  // Adam hyper params
  arg_parser.parse_arg("adam.lr", adam.lr, 0.001, "Adam learning rate");
  arg_parser.parse_arg("adam.beta1", adam.beta1, 0.9, "adam一阶动量平滑常数");
  arg_parser.parse_arg("adam.beta2", adam.beta2, 0.999, "adam二阶动量平滑常数");
  arg_parser.parse_arg("adam.weight_decay_w", adam.weight_decay_w, 0.1, "l2正则在adam中的实现。对于adam，宜用weight_decay，不宜用l2正则");
  arg_parser.parse_arg("adam.weight_decay_V", adam.weight_decay_V, 0.1, "l2正则在adam中的实现。对于adam，宜用weight_decay，不宜用l2正则");

  if (print_debug_info) {
    std::cout << "Adam learning rate: " << adam.lr << std::endl;
    std::cout << "Adam beta1: " << adam.beta1 << std::endl;
    std::cout << "Adam beta2: " << adam.beta2 << std::endl;
    std::cout << "Adam weight decay for w: " << adam.weight_decay_w << std::endl;
    std::cout << "Adam weight decay for V: " << adam.weight_decay_V << std::endl;
  }

  // FTRL hyper params
  arg_parser.parse_arg("ftrl.w_alpha", ftrl.w_alpha, 0.01, "FTRL hyper-param");
  arg_parser.parse_arg("ftrl.w_beta", ftrl.w_beta, 1.0, "FTRL hyper-param");
  arg_parser.parse_arg("ftrl.v_alpha", ftrl.v_alpha, 0.01, "FTRL hyper-param");
  arg_parser.parse_arg("ftrl.v_beta", ftrl.v_beta, 1.0, "FTRL hyper-param");
  arg_parser.parse_arg("ftrl.l1w", ftrl.l1_reg_w, 0.05, "l1 regularization of w");
  arg_parser.parse_arg("ftrl.l2w", ftrl.l2_reg_w, 5.0,  "l2 regularization of w");
  arg_parser.parse_arg("ftrl.l1v", ftrl.l1_reg_V, 0.05, "l1 regularization of V");
  arg_parser.parse_arg("ftrl.l2v", ftrl.l2_reg_V, 5.0,  "l2 regularization of V");

  if (print_debug_info) {
    std::cout << "FTRL w_alpha: " << ftrl.w_alpha << std::endl;
    std::cout << "FTRL w_beta: " << ftrl.w_beta << std::endl;
    std::cout << "FTRL v_alpha: " << ftrl.v_alpha << std::endl;
    std::cout << "FTRL v_beta: " << ftrl.v_beta << std::endl;
    std::cout << "FTRL l1 regularization of w: " << ftrl.l1_reg_w << std::endl;
    std::cout << "FTRL l2 regularization of w: " << ftrl.l2_reg_w << std::endl;
    std::cout << "FTRL l1 regularization of V: " << ftrl.l1_reg_V << std::endl;
    std::cout << "FTRL l2 regularization of V: " << ftrl.l2_reg_V << std::endl;
  }

  print_help = arg_parser.parse_option("h", "print help message");

  arg_parser.process_other_args();

  if (!arg_parser.parse_status || print_help) {
    arg_parser.print_helper();
  }
  return arg_parser.parse_status;
}
