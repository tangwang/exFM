#include "feature/common_feat.h"
#include "solver/solver_factory.h"

CommonFeatContext::CommonFeatContext()
{
    forward_param_container = creatParamContainer(1, 1);
    backward_param_container = creatParamContainer(1, 1);
}
