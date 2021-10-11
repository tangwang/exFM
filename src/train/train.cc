/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "feature/feat_manager.h"
#include "train/train_worker.h"
#include "train/shulffer.h"
#include "solver/ftrl/ftrl_solver.h"
#include "solver/solver_factory.h"
#include "solver/ftrl/ftrl_solver.h"
#include "solver/adam/adam_solver.h"
#include "solver/sgdm/sgdm_solver.h"
#include "solver/adagrad/adagrad_solver.h"
#include "solver/rmsprop/rmsprop_solver.h"

size_t train_dispatcher(vector<TrainWorker *> &workers,
                          std::istream *input_stream, size_t run_sample_num) {
  string input_buff;
  size_t i = run_sample_num;
  MiniShuffer shulffer((uint8)workers.size());
  shulffer.reset();
  for (; i != 0 && std::getline(*input_stream, input_buff); --i) {
    int max_retry_times = train_opt.threads_num;
    for (; max_retry_times != 0; --max_retry_times) {
      if (workers[shulffer.next()]->TryPush(input_buff))
        break;
    }
    if (max_retry_times == 0) {
      workers[shulffer.next()]->WaitAndPush(input_buff);
    }
  }
  return run_sample_num - i;
}

void train_dispatcher(vector<TrainWorker *> &workers,
                           std::istream *input_stream) {

  MiniShuffer shulffer((uint8)workers.size());
  shulffer.reset();

  string input_buff;
  while (std::getline(*input_stream, input_buff)) {
    int max_retry_times = train_opt.threads_num;
#if 1  // TODO check perfermance
    for (;
         max_retry_times != 0 &&
         !workers[shulffer.next()]->TryPush(input_buff);
         --max_retry_times)
      ;

    if (max_retry_times == 0) {
      workers[shulffer.next()]->WaitAndPush(input_buff);
    }
#else
    workers[shulffer.next()]->WaitAndPush(input_buff);
#endif
  }
}

int main(int argc, char *argv[]) {
  srand(time(NULL));

  if (!train_opt.parse_cfg_and_cmdlines(argc, argv)) {
    cerr << "parse args faild, exit" << endl;
    return -1; 
  }

  FeatManager feat_manager;
  assert(!train_opt.feature_config_path.empty());
  assert(access(train_opt.feature_config_path.c_str(), F_OK) != -1);
  if (!feat_manager.loadByFeatureConfig(train_opt.feature_config_path)) {
    cerr << "init feature manager faild, check config file " << train_opt.feature_config_path << ". exit" << endl;
    return -1;
  }

  vector<TrainWorker *> workers;

  for (int thread_id = 0; thread_id < train_opt.threads_num; thread_id++) {
    std::cout << "start train thread " << thread_id << "..." << endl;
    TrainWorker *p = new TrainWorker("train", thread_id);
    p->RegisteSolver(creatSolver(feat_manager));
    p->StartTrainLoop();
    workers.push_back(p);
  }

  TrainWorker *validator = NULL;

  std::istream *input_stream = NULL;
  std::istream *input_file_stream = NULL;
  if (!train_opt.train_path.empty()) {
    input_file_stream = new ifstream(train_opt.train_path);
    input_stream = input_file_stream;
    if (!(*input_file_stream)) {
      cerr << "train file open filed " << endl;
      delete input_file_stream;
      return -1;
    }
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

    validator = new TrainWorker("valid", 0);
    validator->RegisteSolver(creatSolver(feat_manager));
    validator->StartValidationLoop(valid_stream);
    std::cout << "start validation thread " << "..." << endl;
  }

  for (int i = 0; i < train_opt.epoch; i++) {
    cout << "start epoch " << i << endl;
    train_dispatcher(workers, input_stream);
    if (input_file_stream) {
      input_file_stream->clear();
      input_file_stream->seekg(0);
    } else {
      // 从stdin读取训练数据，不支持多个epoch
      break;
    }
  }

  // 分发完后，结束所有线程
  sleep(5);
  for (auto p : workers) {
    p->Stop();
    p->Join();
    delete p;
  }
  if (validator != NULL) {
    sleep(5);
    validator->Stop();
    validator->Join();
    delete validator;
  }
  if (input_file_stream != NULL) {
    delete input_file_stream;
  }

  feat_manager.dumpModel();
  return 0;
}
