#include "feature/dense_fea.h"
#include "feature/fea_manager.h"
#include "feature/sparse_fea.h"
#include "feature/varlen_sparse_fea.h"
#include "train/train_worker.h"
#include "ftrl/ftrl_param.h"
#include "utils/base.h"

void test_fea_manager() {
  FtrlParamUnit::static_init(&train_opt);

  FeaManager fea_manager;
  fea_manager.parse_fea_config("./config/fea.config");
  fea_manager.initModelParams();

  FTRLSolver trainer(fea_manager, train_opt);

  const static int MAX_LINE_BUFF = 10240;
  char line[MAX_LINE_BUFF];
  size_t line_num = 0;

  while (true) {
    if (!cin.getline(line, sizeof(line))) {
      break;
    }
    line_num++;

    trainer.feedSample(line);
  }
}

int main(int argc, char *argv[]) {
  test_fea_manager();
  return 0;
}
