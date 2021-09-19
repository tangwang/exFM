/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "feature/fea_manager.h"
#include "solver/solver_factory.h"
#include "solver/ftrl/ftrl_solver.h"
#include "solver/adam/adam_solver.h"
#include "solver/sgdm/sgdm_solver.h"

shared_ptr<ParamContainerInterface> creatParamContainer(feaid_t fea_num) {
  if (train_opt.solver == "ftrl") {
    return std::make_shared<ParamContainer<FtrlParamUnit>>(fea_num);
  } else if (train_opt.solver == "sgd") {
    return std::make_shared<ParamContainer<SgdmParamUnit>>(fea_num);
  } else if (train_opt.solver == "adam") {
    return std::make_shared<ParamContainer<AdamParamUnit>>(fea_num);
  } else {
    // 默认采用adam
    return std::make_shared<ParamContainer<AdamParamUnit>>(fea_num);
  }
}

BaseSolver * creatParamContainer(const FeaManager &fea_manager) {
  if (train_opt.solver == "ftrl") {
    return new FtrlSolver(fea_manager);
  } else if (train_opt.solver == "sgd" || train_opt.solver == "sgdm") {
    return new SgdmSolver(fea_manager);
  } else if (train_opt.solver == "adam" || train_opt.solver == "adamW" || train_opt.solver == "adamw") {
    return new AdamSolver(fea_manager);
  } else {
    return new AdamSolver(fea_manager);
  }
}

