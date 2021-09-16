/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "train/train_opt.h"

TrainOption train_opt;

bool TrainOption::parse_cfg_and_cmdlines(int argc, char *argv[]) {
  // load config file

  arg_parser.load_args(config_file_path);
  arg_parser.load_args(argc, argv);

  // 训练参数相关配置

  arg_parser.parse_arg("fea_cfg", feature_config_path, string(),
                   "feature_config_path", true);
  arg_parser.parse_arg("train", train_path, string(), 
                   "trainning data path, use stdin(standard input) if not set");
  arg_parser.parse_arg("valid", valid_path, string(), "validation data path");
  arg_parser.parse_arg("valid_interval", time_interval_of_validation,
                   60, "how many seconds between two validition");

  arg_parser.parse_arg("epoch", epoch, 1, "train epochs");


  arg_parser.parse_arg("dim", factor_num, 8, "feature vector dim(factor_num)");
  arg_parser.parse_arg("im", init_model_path, string(), "init model path");
  arg_parser.parse_arg("imf", initial_model_format, string("txt"),
                   "input model format");
  arg_parser.parse_arg("om", model_path, string(), "output model path");
  arg_parser.parse_arg("omf", model_format, string("txt"), "output model format");
  arg_parser.parse_arg("threads", threads_num, 11, "trainning threads_num");

  const char *temp_split_param = NULL;
  arg_parser.parse_arg("fea_sep", temp_split_param, "\t",
                   "specify one character (or str \"blank\",\"tab\") for "
                   "fea_seperator in line of sample. ");
  fea_seperator = parse_seperator_chars(temp_split_param);

  arg_parser.parse_arg(
      "fea_kv_sep", temp_split_param, ":",
      "specify one character (or str \"blank\",\"tab\", \"eqaul\"(for \"=\"))  "
      "for fea_kv_seperator in line of sample. ");
  fea_kv_seperator = parse_seperator_chars(temp_split_param);

  arg_parser.parse_arg("fea_values_sep", temp_split_param, ",",
                   "specify one character(or str \"blank\",\"tab\")  for "
                   "fea_multivalue_seperator in line of sample. ");
  fea_multivalue_seperator = parse_seperator_chars(temp_split_param);

  // ID映射词典相关配置
  disable_feaid_mapping = arg_parser.parse_option(
      "disable_feaid_mapping",
      "ignore feature ID mapping dicts, using hash or original ID.");
  arg_parser.parse_arg("id_map_dict_sep", temp_split_param, " ",
                   "specify one character(or str \"blank\",\"tab\") for k_v_seperator in line of feature mapping dict");
  fea_id_mapping_dict_seperator = parse_seperator_chars(temp_split_param);

//////////////////////
// solver
  arg_parser.parse_arg("solver", solver, string("ftrl"), "solver", false);

  // common hyper params
  arg_parser.parse_arg("l1w", l1_reg_w, 0.05, "l1 regularization of w for all learners");
  arg_parser.parse_arg("l2w", l2_reg_w, 5.0,  "l2 regularization of w for all learners");
  arg_parser.parse_arg("l1v", l1_reg_V, 0.05, "l1 regularization of V for all learners");
  arg_parser.parse_arg("l2v", l2_reg_V, 5.0,  "l2 regularization of V for all learners");

  // FTRL hyper params
  arg_parser.parse_arg("ftrl.init_stdev", ftrl.init_stdev, 0.001,
                  "stdev for initialization of 2-way factors");
  arg_parser.parse_arg("ftrl.w_alpha", ftrl.w_alpha, 0.01, "FTRL hyper-param");
  arg_parser.parse_arg("ftrl.w_beta", ftrl.w_beta, 1.0, "FTRL hyper-param");
  arg_parser.parse_arg("ftrl.v_alpha", ftrl.v_alpha, 0.01, "FTRL hyper-param");
  arg_parser.parse_arg("ftrl.v_beta", ftrl.v_beta, 1.0, "FTRL hyper-param");
 
  // SGDM hyper params
  arg_parser.parse_arg("sgdm.step_size", sgdm.step_size, 0.001, "SGDM learning rate");
  arg_parser.parse_arg("sgdm.beta1"   , sgdm.beta1,    0.9   , "SGD 一阶动量平滑常数");

  // Adam hyper params
  arg_parser.parse_arg("adam.step_size", adam.step_size, 0.001, "Adam learning rate");
  arg_parser.parse_arg("adam.beta1", adam.beta1, 0.9, "adam一阶动量平滑常数");
  arg_parser.parse_arg("adam.bias_correct", adam.bias_correct, 0, "bias_correct");
  arg_parser.parse_arg("adam.beta2", adam.beta2, 0.999, "adam二阶动量平滑常数");
  arg_parser.parse_arg("adam.weight_decay_w", adam.weight_decay_w, 0.0001, "l2正则在adam中的实现。对于adam，宜用weight_decay，不宜用l2正则");
  arg_parser.parse_arg("adam.weight_decay_V", adam.weight_decay_V, 0.0001, "l2正则在adam中的实现。对于adam，宜用weight_decay，不宜用l2正则");
  arg_parser.parse_arg("adam.amsgrad", adam.amsgrad, 0, "保留历史最大的v_t，记为v_{max}，每次计算都是用最大的v_{max}，否则是用当前v_t");


  arg_parser.parse_arg("verbose", verbose, 1,
                  "0: only trainning logs. 1: open feature config messages, 2 "
                  ": open debug messages");

  print_help = arg_parser.parse_option("h", "print help message");

  arg_parser.process_other_args();

  if (!arg_parser.parse_status || print_help) {
    arg_parser.print_helper();
  }
  return arg_parser.parse_status;
}
