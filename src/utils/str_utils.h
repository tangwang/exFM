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

// str -> str
template <>
inline std::string cast_ref_type<std::string, std::string>(const std::string &s) {
  return s;
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
  size_t begin = 0;
  for (size_t i = 0; i < line.size(); ++i) {
    if (line[i] == delimiter) {
      r.emplace_back(cast_ref_type<string, value_type>(line.substr(begin, i - begin)));
      begin = i + 1;
    }
  }
  if (begin < line.size()) {
    r.emplace_back(
        cast_ref_type<string, value_type>(line.substr(begin, line.size() - begin)));
  }
}

template <>
inline void split_string(const string &line, char delimiter, vector<string> &r) {
  size_t begin = 0;
  for (size_t i = 0; i < line.size(); ++i) {
    if (line[i] == delimiter) {
      r.emplace_back(line.substr(begin, i - begin));
      begin = i + 1;
    }
  }
  if (begin < line.size()) {
    r.emplace_back(line.substr(begin, line.size() - begin));
  } else {
    r.emplace_back("");
  }
}

template <typename value_type>
value_type * split_string(const string &line, char delimiter, value_type * r) {
  size_t begin = 0;
  for (size_t i = 0; i < line.size(); ++i) {
    if (line[i] == delimiter) {
      *r = cast_ref_type<string, value_type>(line.substr(begin, i - begin));
      r++;
      begin = i + 1;
    }
  }
  if (begin < line.size()) {
    *r = cast_ref_type<string, value_type>(line.substr(begin, line.size() - begin));
    r++;
  }
  return r;
}

// template <class Iter>
// Iter split_string(const std::string &s, const std::string &delim, Iter out) {
//   size_t a = 0, b = s.find(delim);
//   for (; b != std::string::npos; a = b + delim.length(), b = s.find(delim, a)) {
//     *out++ = std::move(s.substr(a, b - a));
//   }
//   *out++ = std::move(s.substr(a, s.length() - a));
//   return out;
// }

template <typename value_type>
void split_string(const char *beg, char delimiter, char end_char,
                  vector<value_type> &vec) {
  const char *begin = beg;
  const char *p = beg;
  for (; *p && *p != end_char; p++) {
    if (*p == delimiter) {
      if (begin < p) {
        vec.emplace_back(cast_type<const char *, value_type>(begin));
      }
      begin = p + 1;
    }
  }
  if (begin < p) {
    vec.emplace_back(cast_type<const char *, value_type>(begin));
  }
}

template <>
inline void split_string(const char *beg, char delimiter, char end_char,
                  vector<string> &vec) {
  const char *begin = beg;
  const char *p = beg;
  for (; *p && *p != end_char; p++) {
    if (*p == delimiter) {
      if (begin < p) {
        vec.emplace_back(string(begin, p - begin));
      }
      begin = p + 1;
    }
  }
  if (begin < p) {
    vec.emplace_back(string(begin, p - begin));
  }
}

inline void split_string(const char *beg, char delimiter,
                  vector<string> &vec) {
  const char *begin = beg;
  const char *p = beg;
  for (; *p; p++) {
    if (*p == delimiter) {
      if (begin < p) {
        vec.emplace_back(string(begin, p - begin));
      }
      begin = p + 1;
    }
  }
  if (begin < p) {
    vec.emplace_back(string(begin, p - begin));
  }
}

inline bool startswith(const std::string &s, char pfx) {
  return !s.empty() && s[0] == pfx;
}

inline bool startswith(const std::string &s, const char *pfx, size_t len) {
  return s.size() >= len && (std::memcmp(s.data(), pfx, len) == 0);
}

inline bool startswith(const std::string &s, const char *pfx) {
  return startswith(s, pfx, std::strlen(pfx));
}

inline bool startswith(const std::string &s, const std::string &pfx) {
  return startswith(s, pfx.data(), pfx.size());
}

inline bool endswith(const std::string &s, char sfx) {
  return !s.empty() && s[s.size() - 1] == sfx;
}

inline bool endswith(const std::string &s, const char *sfx, size_t len) {
  return s.size() >= len &&
         (std::memcmp(s.data() + s.size() - len, sfx, len) == 0);
}

inline bool endswith(const std::string &s, const char *sfx) {
  return endswith(s, sfx, std::strlen(sfx));
}

inline bool endswith(const std::string &s, const std::string &sfx) {
  return endswith(s, sfx.data(), sfx.size());
}

inline std::string::size_type common_prefix_length(const std::string &a,
                                                   const std::string &b) {
  std::string::size_type minlen = std::min(a.size(), b.size());
  std::string::size_type common;
  for (common = 0; common < minlen; ++common) {
    if (a[common] != b[common]) break;
  }
  return common;
}

}  // namespace utils
