/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "solver/parammeter_interface.h"
#include "solver/base_solver.h"



shared_ptr<ParamContainerInterface> creat_param_container(int fea_num);

BaseSolver * CreateSover(const FeaManager &fea_manager);
