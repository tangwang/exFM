/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "feature/fea_manager.h"
#include "ftrl/ftrl_learner.h"
#include "ftrl/param_container.h"
#include "ftrl/train_opt.h"
#include "train/evalution.h"


int main(int argc, char *argv[]) {
  srand(time(NULL));

  if (!train_opt.parse_args(argc, argv)) {
    cerr << "parse args faild, exit" << endl;
    return -1;
  }

  CommonFeaConfig::static_init(&train_opt);
  FTRLParamUnit::static_init(&train_opt);

  FeaManager fea_manager;
  assert(access(train_opt.feature_config_path, F_OK) != -1);
  fea_manager.parse_fea_config(train_opt.feature_config_path);
  fea_manager.initModelParams(true);

  FTRLLearner * trainer = FTRLLearner::Create(fea_manager, train_opt);

  const static int MAX_LINE_BUFF = 10240;
  char line[MAX_LINE_BUFF];
  size_t line_num = 0;

  size_t n_sample_per_output = 10000;
  size_t time_interval_of_validation = 100000;
  Evalution train_eval;
  Evalution evaldata_eval;
  int processed_samples = 0;

  std::istream *input_stream = NULL;
  std::istream *input_file_stream = NULL;
  if (train_opt.train_path && *train_opt.train_path) {
    input_file_stream = new ifstream(train_opt.train_path, std::ios::in);
    input_stream = input_file_stream;
  } else {
    cin.sync_with_stdio(false);
    input_stream = &std::cin;
  }

  std::ifstream eval_stream;
  if (train_opt.eval_path && *train_opt.eval_path) {
    eval_stream.open(train_opt.eval_path);
    if (!eval_stream) {
      cerr << "eval file open filed " << endl;
      return -1;
    }
  }

  while (input_stream->getline(line, sizeof(line))) {
    line_num++;

    ++processed_samples;

    trainer->feedRawData(line);
    trainer->train(*train_context);
    train_eval.add(trainer->y, train_context->logit);

    if (train_eval.size() == n_sample_per_output) {
      train_eval.output("train");
    }
    if (eval_stream && (processed_samples % time_interval_of_validation == 0)) {
      eval_stream.clear();
      eval_stream.seekg(0);
      while (eval_stream.getline(line, sizeof(line))) {
        trainer->feedRawData(line);
        trainer->train(true);
        evaldata_eval.add(trainer->y, trainer->logit);
      }
      evaldata_eval.output("eval");
    }
  }

  if (input_file_stream != NULL) {
    delete input_file_stream;
  }
  if (eval_stream != NULL) {
    delete eval_stream;
  }
  FTRLLearner::Destroy(trainer);
  return 0;
}
