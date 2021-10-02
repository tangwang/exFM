#include "feature/dense_feat.h"
#include "feature/feat_manager.h"
#include "feature/sparse_feat.h"
#include "feature/varlen_sparse_feat.h"
#include "train/train_worker.h"
#include "solver/ftrl/ftrl_param.h"
#include "utils/base.h"

void test_feat_manager() {

  FeatManager feat_manager;
  feat_manager.loadByFeatureConfig("./config/fea.config");

  Solver trainer(feat_manager, train_opt);

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
  test_feat_manager();
  return 0;
}
