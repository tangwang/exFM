/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "feature/feat_manager.h"
#include "train/train_worker.h"
#include "solver/ftrl/ftrl_param.h"
#include "train/train_opt.h"
#include "train/evalution.h"


int main(int argc, char *argv[]) {
  srand(time(NULL));

  if (!train_opt.parse_args(argc, argv)) {
    cerr << "parse args faild, exit" << endl;
    return -1;
  }

  CommonFeatConfig::static_init(&train_opt);

  FeatManager feat_manager;
  assert(access(train_opt.feature_config_path, F_OK) != -1);
  feat_manager.loadByFeatureConfig(train_opt.feature_config_path);

  Solver * trainer = Solver::Create(feat_manager, train_opt);

  const int MAX_LINE_BUFF = 10240;
  char line[MAX_LINE_BUFF];
  size_t line_num = 0;

  const size_t time_interval_of_validation = 100000;
  Evalution train_eval;
  Evalution evaldata_eval;
  int processed_samples = 0;

  std::istream *input_stream = NULL;
  std::istream *input_file_stream = NULL;
  if (!train_opt.train_path.empty()) {
    input_file_stream = new ifstream(train_opt.train_path, std::ios::in);
    input_stream = input_file_stream;
  } else {
    cin.sync_with_stdio(false);
    input_stream = &std::cin;
  }

  std::ifstream valid_stream;
  if (!train_opt.valid_path.empty()) {
    valid_stream.open(train_opt.valid_path);
    if (!valid_stream) {
      cerr << "eval file open filed " << endl;
      return -1;
    }
  }

  while (input_stream->getline(line, sizeof(line))) {
    line_num++;

    ++processed_samples;

    trainer->feedSample(line);
    trainer->train(*train_context);
    train_eval.add(trainer->y, train_context->logit);

    if (train_eval.size() == train_opt.n_sample_per_output) {
      train_eval.output("train");
    }
    if (valid_stream && (processed_samples % time_interval_of_validation == 0)) {
      valid_stream.clear();
      valid_stream.seekg(0);
      while (valid_stream.getline(line, sizeof(line))) {
        trainer->feedSample(line);
        trainer->train(true);
        evaldata_eval.add(trainer->y, trainer->logit);
      }
      evaldata_eval.output("eval");
    }
  }

  if (input_file_stream != NULL) {
    delete input_file_stream;
  }
  if (valid_stream != NULL) {
    delete valid_stream;
  }
  Solver::Destroy(trainer);
  return 0;
}
