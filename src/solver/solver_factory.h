/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "solver/parammeter_container.h"
#include "solver/base_solver.h"


shared_ptr<ParamContainerInterface> creatParamContainer(feat_id_t  feat_num, feat_id_t mutex_nums);

BaseSolver * creatSolver(const FeatManager &feat_manager);
