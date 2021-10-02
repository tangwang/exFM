/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "solver/parammeter_container.h"
#include "solver/base_solver.h"


shared_ptr<ParamContainerInterface> creatParamContainer(feaid_t fea_num, feaid_t mutex_nums);

BaseSolver * creatSolver(const FeatManager &feat_manager);
