/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "feature/feat_manager.h"
#include "solver/solver_factory.h"
#include "solver/ftrl/ftrl_solver.h"
#include "solver/adam/adam_solver.h"
#include "solver/sgdm/sgdm_solver.h"
#include "solver/adagrad/adagrad_solver.h"
#include "solver/rmsprop/rmsprop_solver.h"

shared_ptr<ParamContainerInterface> creatParamContainer(feaid_t fea_num, feaid_t mutex_nums) {
  if (0 == strcasecmp(train_opt.solver.c_str(), "ftrl")) {
    return std::make_shared<ParamContainer<FtrlParamUnit>>(fea_num, mutex_nums);
  } else if (0 == strcasecmp(train_opt.solver.c_str(), "sgd") || 0 == strcasecmp(train_opt.solver.c_str(), "sgdm")) {
    return std::make_shared<ParamContainer<SgdmParamUnit>>(fea_num, mutex_nums);
  } else if (0 == strcasecmp(train_opt.solver.c_str(), "adagrad")) {
    return std::make_shared<ParamContainer<AdagradParamUnit>>(fea_num, mutex_nums);
  } else if (0 == strcasecmp(train_opt.solver.c_str(), "rmsprop")) {
    return std::make_shared<ParamContainer<RmspropParamUnit>>(fea_num, mutex_nums);
  } else if (0 == strcasecmp(train_opt.solver.c_str(), "adam")) {
    return std::make_shared<ParamContainer<AdamParamUnit>>(fea_num, mutex_nums);
  } else {
    cerr << "unknown solver, use adam by default." << endl;
    return std::make_shared<ParamContainer<AdamParamUnit>>(fea_num, mutex_nums);
  }
}

BaseSolver * creatSolver(const FeatManager &feat_manager) {
  if (0 == strcasecmp(train_opt.solver.c_str(), "ftrl")) {
    return new FtrlSolver(feat_manager);
  } else if (0 == strcasecmp(train_opt.solver.c_str(), "sgd") || 0 == strcasecmp(train_opt.solver.c_str(), "sgdm")) {
    return new SgdmSolver(feat_manager);
  } else if (0 == strcasecmp(train_opt.solver.c_str(), "adagrad")) {
    return new AdagradSolver(feat_manager);
  } else if (0 == strcasecmp(train_opt.solver.c_str(), "rmsprop")) {
    return new RmspropSolver(feat_manager);
  } else if (0 == strcasecmp(train_opt.solver.c_str(), "adam")) {
    return new AdamSolver(feat_manager);
  } else {
    cerr << "unknown solver, use adam by default." << endl;
    return new AdamSolver(feat_manager);
  }
}

