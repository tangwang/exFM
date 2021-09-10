/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once

#include <string>
#include <vector>

#include "utils/base.h"

namespace utils {

template <typename src_type, typename dest_type>
dest_type cast_type(src_type s) {
  return s;
}

template <typename src_type, typename dest_type>
dest_type cast_ref_type(const src_type &s) {
  return s;
}

// str -> int
template <>
inline int cast_ref_type<std::string, int>(const std::string &s) {
  return atoi(s.c_str());
}

template <>
inline int cast_type<const char *, int>(const char *s) {
  return atoi(s);
}

// str -> unsigned int
template <>
inline unsigned int cast_ref_type<std::string, unsigned int>(
    const std::string &s) {
  return atoi(s.c_str());
}

template <>
inline unsigned int cast_type<const char *, unsigned int>(const char *s) {
  return atoi(s);
}

// str -> char
template <>
inline char cast_type<const char *, char>(const char *s) {
  return s[0];
}

// str -> long
template <>
inline long cast_ref_type<std::string, long>(const std::string &s) {
  return atol(s.c_str());
}

template <>
inline long cast_type<const char *, long>(const char *s) {
  return atol(s);
}

// str -> unsigned long
template <>
inline unsigned long cast_ref_type<std::string, unsigned long>(
    const std::string &s) {
  return atol(s.c_str());
}

template <>
inline unsigned long cast_type<const char *, unsigned long>(const char *s) {
  return atol(s);
}

// str -> float
template <>
inline float cast_ref_type<std::string, float>(const std::string &s) {
  return atof(s.c_str());
}

template <>
inline float cast_type<const char *, float>(const char *s) {
  return atof(s);
}

// str -> double
template <>
inline double cast_ref_type<std::string, double>(const std::string &s) {
  return atof(s.c_str());
}

template <>
inline double cast_type<const char *, double>(const char *s) {
  return atof(s);
}
// end of str type casts

inline std::string &trim(std::string &s) {
  s.erase(0, s.find_first_not_of(" "));
  s.erase(s.find_last_not_of(" ") + 1);
  return s;
}

template <typename value_type>
void split_string(const string &line, char delimiter, vector<value_type> &r) {
  int begin = 0;
  for (int i = 0; i < line.size(); ++i) {
    if (line[i] == delimiter) {
      r.push_back(cast_type<string, value_type>(line.substr(begin, i - begin)));
      begin = i + 1;
    }
  }
  if (begin < line.size()) {
    r.push_back(
        cast_type<string, value_type>(line.substr(begin, line.size() - begin)));
  }
}

template <class Iter>
Iter split_string(const std::string &s, const std::string &delim, Iter out) {
  size_t a = 0, b = s.find(delim);
  for (; b != std::string::npos; a = b + delim.length(), b = s.find(delim, a)) {
    *out++ = std::move(s.substr(a, b - a));
  }
  *out++ = std::move(s.substr(a, s.length() - a));
  return out;
}

template <typename value_type>
void split_string(const char *beg, char delimiter, char end_char,
                  vector<value_type> &vec) {
  const char *feaid_beg = beg;
  const char *p = beg;
  for (; *p && *p != end_char; p++) {
    if (*p == delimiter) {
      if (feaid_beg < p) {
        vec.push_back(cast_type<const char *, value_type>(feaid_beg));
      }
      feaid_beg = p + 1;
    }
  }
  if (feaid_beg < p) {
    vec.push_back(cast_type<const char *, value_type>(feaid_beg));
  }
}

}  // namespace utils
