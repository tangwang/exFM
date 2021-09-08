#include "feature/dense_fea.h"
#include "feature/fea_manager.h"
#include "feature/sparse_fea.h"
#include "feature/varlen_sparse_fea.h"
#include "ftrl/ftrl_learner.h"
#include "ftrl/param_container.h"
#include "utils/base.h"

int main(int argc, char *argv[]) {
  static_assert(sizeof(void *) == 8,
                "only 64-bit code generation is supported.");
  cin.sync_with_stdio(false);
  cout.sync_with_stdio(false);
  srand(time(NULL));
  TrainOption train_opt;
  try {
    train_opt.parse_option(argc, argv);
  } catch (const invalid_argument &e) {
    cerr << "invalid_argument:" << e.what() << endl;
    cerr << train_help() << endl;
    return 1;
  }

  FTRLParamUnit::static_init(train_opt);

  FeaManager fea_manager;
  fea_manager.parse_fea_config("./config/fea.config");
  fea_manager.initModelParams(true);

  FTRLLearner trainer(fea_manager, train_opt);

  const static int MAX_LINE_BUFF = 10240;
  char line[MAX_LINE_BUFF];
  size_t line_num = 0;

  while (true) {
    if (!cin.getline(line, sizeof(line))) {
      break;
    }
    line_num++;

    trainer.feedRawData(line);
  }
  return 0;
}
