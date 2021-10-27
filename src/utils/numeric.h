/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once

#include <random>  // for sampling from distributions
#include <string>
#include <vector>
#include <cmath>
#include <ostream>
#include <iterator>

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
T norm2(T const *data, int len) {
  T norm = 0;
  for (int i = 0; i < len; ++i) norm += data[i] * data[i];
  return norm;
}

template <typename T>
T norm2(const T &data) {
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

// @return : if (a > 0) return b; else return -b;
inline double sign_a_multiply_b(double a, double b) {
  #if 1
  if (a > 0) return b; else return -b;
  #else
  unsigned long *p = (unsigned long *)&a;
  unsigned long bits_of_sign_a_multiply_b =
      (((*p) & ((unsigned long)1<<63)) ^ (*((unsigned long *)&b)));
  return *((double *)(&bits_of_sign_a_multiply_b));
  #endif
}

// @return : if (a > 0) return b; else return -b;
inline float sign_a_multiply_b(float a, float b) {
  #if 1
  if (a > 0) return b; else return -b;
  #else
  unsigned int *p = (unsigned int *)&a;
  unsigned int bits_of_sign_a_multiply_b =
      (((*p) & ((unsigned long)1<<31)) ^ (*((unsigned int *)&b)));
  return *((float *)(&bits_of_sign_a_multiply_b));
  #endif
}

inline real_t uniform() { return rand() / static_cast<real_t>(RAND_MAX); }

inline real_t gaussian(real_t mean, real_t stdev) {
  static std::default_random_engine generator;
  static std::normal_distribution<real_t> distribution(mean, stdev);
  real_t ret = distribution(generator);
  while (ret == 0.0) //TODO 是否要保证不为0.
  {
    ret = distribution(generator);
  }
  return ret;
}

inline real_t gaussian() { return gaussian(0.0, 1.0); }

// template <typename value_type>
// void print2dArray(const vector<vector<value_type>> &arrays) {
//   int row = 0;
//   for (auto &array : arrays) {
//     std::cout << "row_" << row++ << " ";
//     for (auto &v : array) {
//       std::cout << v << " ";
//     }
//     std::cout << endl;
//   }
// }

// template <typename value_type>
// void printArray(const vector<value_type> &array) {
//   for (auto &v : array) {
//     std::cout << v << " ";
//   }
//   std::cout << endl;
// }

}  // namespace utils


namespace std {

template <typename value_type>
ostream &operator<<(ostream &os, const vector<value_type> &array) {
  os << "[";
  std::copy(array.begin(), array.end(), ostream_iterator<value_type>(os, ", "));
  os << "]";
  return os;
}

template <typename value_type>
ostream &operator<<(ostream &os, const vector<vector<value_type>> &arrays) {
  os << "[";
  if (!arrays.empty()) {
    size_t len = arrays.size();
    os << arrays[0] << endl;
    for (size_t i = 1; i < len; i++) {
      os << ", " << endl << arrays[i];
    }
  }
  os << "]";
  return os;
}

}
