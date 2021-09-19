/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "solver/parammeter_container.h"
#include "solver/base_solver.h"


shared_ptr<ParamContainerInterface> creatParamContainer(int fea_num);

BaseSolver * creatParamContainer(const FeaManager &fea_manager);
