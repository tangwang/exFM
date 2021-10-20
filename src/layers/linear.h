/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "utils/base.h"
#include "utils/utils.h"


template <int dim_in, int dim_out>
class Linear
{
public:
    linear() {}
    ~linear() {}

    void lock() {
        mutex.lock();
    }
    void unlock() {
        mutex.unlock();
    }

    void forward(real_t* in_data, real_t* out_data) const {
      for (int row = 0; row < dim_in; row++) {
        out_data[row] = 0.0;
        for (int col = 0; col < dim_in; col++) {
          out_data[row] += in_data[col] * mat[row][col];
        }
      }
    }

    void backward(real_t * grad) {

    }


private:
    ParamMutex_t mutex;

    real_t mat[dim_in][dim_out];
};

