#include "feature/common_fea.h"
#include "solver/solver_factory.h"

CommonFeaContext::CommonFeaContext()
{
    forward_param_container = creatParamContainer(1);
    backward_param_container = creatParamContainer(1);
}
