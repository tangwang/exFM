/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once

#include <random>  // for sampling from distributions
#include <string>
#include <vector>

#include "utils/base.h"
#include "utils/str_utils.h"
#include "utils/numeric.h"
#include "utils/console_color.h"

namespace utils {

template <typename first_type, typename second_type>
inline bool judgeByPairFirst(const std::pair<first_type, second_type> &a,
                             const std::pair<first_type, second_type> &b) {
  return a.first < b.first;
}

template <typename first_type, typename second_type>
inline bool judgeByPairSecond(const std::pair<first_type, second_type> &a,
                              const std::pair<first_type, second_type> &b) {
  return a.second < b.second;
}

template <typename first_type, typename second_type>
inline bool judgeByPairFirstGreater(const std::pair<first_type, second_type> &a,
                             const std::pair<first_type, second_type> &b) {
  return a.first > b.first;
}

template <typename first_type, typename second_type>
inline bool judgeByPairSecondGreater(const std::pair<first_type, second_type> &a,
                              const std::pair<first_type, second_type> &b) {
  return a.second > b.second;
}


}  // namespace utils
