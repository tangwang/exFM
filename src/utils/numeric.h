/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once

#include <random>  // for sampling from distributions
#include <string>
#include <vector>

#include "utils/base.h"
#include "utils/str_utils.h"

namespace utils {

/**
 * \brief returns the sum of a a vector
 */
template <typename T>
T sum(T const *data, int len) {
  T res = 0;
  for (int i = 0; i < len; ++i) res += data[i];
  return res;
}

/**
 * \brief return l1 norm of a vector
 */
template <typename T>
T norm1(T const *data, int len) {
  T norm = 0;
  for (int i = 0; i < len; ++i) norm += fabs(data[i]);
  return norm;
}

/**
 * \brief return l2 norm of a vector
 */
template <typename T>
double norm2(T const *data, int len) {
  double norm = 0;
  for (int i = 0; i < len; ++i) norm += data[i] * data[i];
  return norm;
}

template <typename T>
double norm2(const T &data) {
  return norm2(data.data(), data.size());
}

template <typename value_type>
void CalcMeanAndStdev(const vector<value_type> &vec, value_type &mean,
                      value_type &stdev) {
  if (vec.empty()) {
    mean = 0.0;
    stdev = 0.0;
  } else if (vec.size() == 1) {
    mean = vec[0];
    stdev = 0.0;
  } else {
    value_type sum = std::accumulate(std::begin(vec), std::end(vec), 0.0);
    mean = sum / vec.size();
    value_type accum = 0.0;
    std::for_each(std::begin(vec), std::end(vec), [&](const value_type d) {
      accum += (d - mean) * (d - mean);
    });
    stdev = sqrt(accum / (vec.size() - 1));
  }
}

#if 1
inline double uniform()
{
    return rand()/((double)RAND_MAX + 1.0);
}


inline double gaussian()
{
    double u,v, x, y, Q;
    do
    {
        do 
        {
            u = uniform();
        } while (u == 0.0); 

        v = 1.7156 * (uniform() - 0.5);
        x = u - 0.449871;
        y = fabs(v) + 0.386595;
        Q = x * x + y * (0.19600 * y - 0.25472 * x);
    } while (Q >= 0.27597 && (Q > 0.27846 || v * v > -4.0 * u * u * log(u)));
    return v / u;
}


inline double gaussian(double mean, double stdev)
{
    if(0.0 == stdev)
    {
        return mean;
    }
    else
    {
        return mean + stdev * gaussian();
    }
}

#else
inline real_t uniform() { return rand() / static_cast<real_t>(RAND_MAX); }

inline real_t gaussian(real_t mean, real_t stdev) {
  std::default_random_engine generator;
  std::normal_distribution<real_t> distribution(mean, stdev);
  return distribution(generator);
}

inline real_t gaussian() { return gaussian(0.0, 1.0); }
#endif


template <typename value_type>
void print2dArray(const vector<vector<value_type>> &arrays) {
  int row = 0;
  for (auto &array : arrays) {
    std::cout << "row_" << row++ << " ";
    for (auto &v : array) {
      std::cout << v << " ";
    }
    std::cout << "\n";
  }
}

template <typename value_type>
void printArray(const vector<value_type> &array) {
  for (auto &v : array) {
    std::cout << v << " ";
  }
  std::cout << "\n";
}

}  // namespace utils
