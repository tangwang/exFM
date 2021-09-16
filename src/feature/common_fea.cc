#include "feature/common_fea.h"
#include "solver/solver_factory.h"

CommonFeaContext::CommonFeaContext()
{
    forward_param_container = creat_param_container(1);
    backward_param_container = creat_param_container(1);
    local_buff_container = creat_param_container(1);
}
