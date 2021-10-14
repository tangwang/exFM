/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include <sstream>

#include "utils/base.h"
#include "utils/utils.h"


class ArgsParser {
 public:
  ArgsParser() : parse_status(true) {}
  ~ArgsParser() {}

  // 如果有必要参数未得到解析，或者有不认识的参数，将被设为false
  bool parse_status;

  // 从配置文件获取参数
  // # 号后面的内容都会被忽略
  // 配置文件中的内容有两种形式： 包含"="和不包含"="的。包含“=”的将被解析为argument，不包含“=”被解析为option
  bool loadArgs(const char* config_file_path) {
    // load config file
    if (access(config_file_path, F_OK) == 0) {
      ifstream config_file(config_file_path);
      if (config_file.is_open()) {
        string temp_str;
        while (std::getline(config_file, temp_str)) {
          utils::trim(temp_str);
          const char* comment_pos = strchr(temp_str.c_str(), '#');
          if (comment_pos != NULL) {
            temp_str.erase(comment_pos - temp_str.c_str());
          }
          if (!temp_str.empty()) arg_lines.push_back(temp_str);
        }
      }
      return true;
    }
    return false;
  }

  // 从命令行参数加载参数（带“=”的被解析为argument，不带“=”的被解析为option)，并覆盖之前已经存在的参数
  bool loadArgs(int argc, char* argv[]) {
    program_name = argv[0];
    for (int i = 1; i < argc; ++i) {
      arg_lines.push_back(string(argv[i]));
    }

    vector<string> splite_fields;
    for (const string& arg_line : arg_lines) {
      // is argument
      splite_fields.clear();
      utils::split_string(arg_line, '=', splite_fields);

      string arg_key =  splite_fields.size() > 0 ? splite_fields[0] : string();
      string arg_value =  splite_fields.size() > 1 ? splite_fields[1] : string();
      if (arg_key.empty()) continue;

      utils::trim(arg_key);
      utils::trim(arg_value);
      bool existed = false;
      for (auto& arg : args) {
        if (arg.k == arg_key) {
          arg.v = arg_value;
          existed = true;
        }
      }
      if (!existed) args.push_back(Arg(arg_key, arg_value));
    }
    return true;
  }

  // 解析一个参数。
  // @param v: 如果该参数被设置，则将被设置的值解析到v
  // @param default_value: 如果该参数未被设置，则v设为default_value
  // @param necessary: 该参数是否为必要参数。如果是必要参数，并且该参数得不到解析，则parse_status将被设置为false
  template <typename value_type>
  void parse_arg(const char* key, value_type& v, value_type default_value,
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
        arg.process_stat = 1;
        // v非空使用设定的值，否则，使用默认值
        if (!arg.v.empty()) {
          v = utils::cast_type<const char*, value_type>(arg.v.c_str());
          std::cout << "arg    " << setiosflags(ios::left)
                    << console_color::green << setw(20) << key
                    << console_color::reset << " is set to "
                    << console_color::green << v << console_color::reset
                    << endl;
          return;
        }
      }
    }
    v = default_value;
    if (necessary) {
      help_message << console_color::red << "ERROR! Missing necessary arg:  <"
                   << key << "> " << console_color::reset << endl;
    }
    if (necessary) parse_status = false;
  }

  // 解析一个option
  // @return：该option是否被设置
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
          help_message << console_color::red << key
                       << " is an option, the value your specified cannot be "
                          "recognized"
                       << console_color::reset << std::endl;
        }

        std::cout << "option " << setiosflags(ios::left) << console_color::green
                 << setw(20) << key << console_color::reset << " is setted "
                 << console_color::reset << endl;

        return true;
      }
    }
    return false;
  }

  // 处理被加载、但是未被解析的参数，如果有，会答应相应的错误信息，并且将parse_status将被设置为false
  void process_other_args() {
    for (const auto& arg : args) {
      if (arg.process_stat == 0) {
        std::cerr << console_color::red << "unknown args " << arg.k
                  << console_color::reset << std::endl;
        parse_status = false;
      }
    }
  }

  void print_helper() const {
    std::cerr
        << "\n\nexFM -- FM with some useful extensions. \n"
           "usage : "
        << program_name << " arg1=value1 arg2=value2 ...  opt1 opt2 ... " << endl;
    std::cerr << help_message.str() << endl;
  }

private:
  struct Arg {
  public:
    Arg(std::string _k, std::string _v) : k(_k), v(_v), process_stat(0) {}
    std::string k;
    std::string v;
    int process_stat;  // 0 : not processed, unknown arg , 1 : processed ok, 2 :
                      // bad values
  };

  vector<string> arg_lines;

  const char* program_name;
  std::vector<Arg> args;
  std::stringstream help_message;
};
