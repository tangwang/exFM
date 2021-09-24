/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "feature/fea_manager.h"
#include "solver/solver_factory.h"
#include "solver/ftrl/ftrl_solver.h"
#include "solver/adam/adam_solver.h"
#include "solver/sgdm/sgdm_solver.h"
#include "solver/adagrad/adagrad_solver.h"
#include "solver/rmsprop/rmsprop_solver.h"

shared_ptr<ParamContainerInterface> creatParamContainer(feaid_t fea_num, feaid_t mutex_nums) {
  if (train_opt.solver == "ftrl") {
    return std::make_shared<ParamContainer<FtrlParamUnit>>(fea_num, mutex_nums);
  } else if (train_opt.solver == "sgd") {
    return std::make_shared<ParamContainer<SgdmParamUnit>>(fea_num, mutex_nums);
  } else if (train_opt.solver == "adagrad") {
    return std::make_shared<ParamContainer<AdagradParamUnit>>(fea_num, mutex_nums);
  } else if (train_opt.solver == "rmsprop") {
    return std::make_shared<ParamContainer<RmspropParamUnit>>(fea_num, mutex_nums);
  } else if (train_opt.solver == "adam") {
    return std::make_shared<ParamContainer<AdamParamUnit>>(fea_num, mutex_nums);
  } else {
    return std::make_shared<ParamContainer<AdamParamUnit>>(fea_num, mutex_nums);
  }
}

BaseSolver * creatSolver(const FeaManager &fea_manager) {
  if (train_opt.solver == "ftrl") {
    return new FtrlSolver(fea_manager);
  } else if (train_opt.solver == "sgdm") {
    return new SgdmSolver(fea_manager);
  } else if (train_opt.solver == "adagrad") {
    return new AdagradSolver(fea_manager);
  } else if (train_opt.solver == "rmsprop") {
    return new RmspropSolver(fea_manager);
  } else if (train_opt.solver == "adam") {
    return new AdamSolver(fea_manager);
  } else {
    return new AdamSolver(fea_manager);
  }
}

