/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/fea_manager.h"
#include "solver/solver_factory.h"
#include "solver/ftrl/ftrl_solver.h"
#include "solver/adam/adam_solver.h"
#include "solver/sgd/sgd_solver.h"

shared_ptr<ParamContainerInterface> creat_param_container(feaid_t fea_num) {
  if (train_opt.solver == "ftrl") {
    return std::make_shared<FtrlParamContainer>(fea_num);
  } else if (train_opt.solver == "sgd") {
    return std::make_shared<SgdParamContainer>(fea_num);
  } else if (train_opt.solver == "adam") {
    return std::make_shared<AdamParamContainer>(fea_num);
  } else {
    // 默认采用adam
    return std::make_shared<AdamParamContainer>(fea_num);
  }
}

BaseSolver * CreateSover(const FeaManager &fea_manager) {
  if (train_opt.solver == "ftrl") {
    return new FtrlSolver(fea_manager);
  } else if (train_opt.solver == "sgd") {
    return new SgdSolver(fea_manager);
  } else if (train_opt.solver == "adam") {
    return new AdamSolver(fea_manager);
  } else {
    return new AdamSolver(fea_manager);
  }
}

