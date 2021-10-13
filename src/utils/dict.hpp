/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once

#define SUPPORT_BOOST 0

#include <iostream>
#include <fstream>
#include <string>
#include <iostream>
#include <fstream>
#include <cstring>
#include <unordered_map>
#include <set>
#include <vector>
#include <algorithm>
#include <tr1/functional>
#include <unordered_map>
#include "cedar/cedar.h"
#if SUPPORT_BOOST
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <boost/regex.hpp>
#else
#include "utils/str_utils.h"
#endif

/*

examples: 
    Dict<int, std::string> my_dict_1;
    my_dict_1.create("path_to_xxxxx", "\t", "#");


    Set<std::string>  my_set_1;
    my_set_1.create("path_to_xxxxx", "#");

    Dict<std::string, std::vector<int> > name_to_ids;
    Dict<int, std::string> id_to_name;
    id_to_name.create("path_to_xxxxx", "\t", "#");
    
    id_to_name.get(k1);


define your deserializer of dict field struct:

template<>
class CommonFieldParser<MyFieldStruct> {
public:
    typedef MyFieldStruct field_type;
    bool operator()(const std::string & str, field_type & v) {
        v.a = xxx;
        v.b = xxx;
        ... ...
    }
*/

namespace utils {

using std::vector;
using std::string;

/*
 * 通用的字段解析器
 * 如果该字段解析器不能满足你的需求，请对该解析器进行特化，或者在声明你的Dict时指定你自己实现的字段解析器。
 */
template <typename Field_t>
class CommonFieldParser {
public:
    typedef Field_t field_type;

    bool operator()(const std::string & str, Field_t & v) {
        try {
#if SUPPORT_BOOST
            v = boost::lexical_cast<Field_t>(str);
#else
            v = utils::cast_ref_type<std::string, Field_t>(str);
#endif
        } catch (...) {
            return false;
        }
        return true;
    }
};


template <typename Key_t, typename Value_t,
          class KeyParser_t = CommonFieldParser<Key_t>,
          class ValueParser_t = CommonFieldParser<Value_t> >
class Dict {
     typedef std::unordered_map<Key_t, Value_t> InnerDict_t;
public:
    typedef typename InnerDict_t::const_iterator const_iterator;
    typedef Key_t key_type;
    typedef Value_t value_type;

public:
    Dict(Value_t null_value = Value_t())
    : _null_value(null_value)
    { }

    bool create(const std::string & path, char separator = '\t') {
        _path = path;
        _separator = separator;
        return _load();
    }

    void setNullValue(Value_t null_value) {
        _null_value = null_value;
    }

    virtual ~Dict() { }

    bool empty() const {
        return _inner_dict.empty();
    }

    const Value_t & get(const Key_t & key) const {
        const_iterator iter = _inner_dict.find(key);
        return iter != _inner_dict.end() ? iter->second : _null_value;
    }

    Value_t & get_mutable(const Key_t & key) {
        return _inner_dict[key];
    }

    const Value_t & operator[](const Key_t & key) const {
        return get(key);
    }

    void set(const Key_t & key, const Value_t & v) {
        _inner_dict[key] = v;
    }

    size_t size() const {
        return _inner_dict.size();
    }

    // read-only iterator
    const_iterator begin() const {
        return _inner_dict.begin();
    }

    const_iterator end() const {
        return _inner_dict.end();
    }

private:
    bool _load() {
        using namespace std;
        bool ret = true;
        ifstream ifs;
        ifs.open(_path.c_str());
        if (!ifs) {
            return false;
        }
        string line;
        vector<string> line_segs;
        while (ifs) {
            getline(ifs, line);

            if (line.empty()) {
                continue;
            }
            line_segs.clear();
            split_string(line, _separator, line_segs);
            // boost::split(line_segs, line, boost::is_any_of(_separator.c_str()),
            //         boost::token_compress_on);
            if (line_segs.size() != 2) {
                ret = false;
                continue;
            }

            Key_t k;
            Value_t v(_null_value);
            if (!key_parser(line_segs[0], k) || !value_parser(line_segs[1], v)) {
                ret = false;
                continue;
            }

            _inner_dict.insert(std::make_pair(k, v));
        }
        ifs.close();
        return ret;
    }

    static void split_string(string& line, char delimiter, vector<string> & r)
    {
        size_t begin = 0;
        for(size_t i = 0; i < line.size(); ++i)
        {
            if(line[i] == delimiter)
            {
                r.push_back(line.substr(begin, i - begin));
                begin = i + 1;
            }
        }
        if(begin < line.size())
        {
            r.push_back(line.substr(begin, line.size() - begin));
        }
    }

private:
    KeyParser_t key_parser;
    ValueParser_t value_parser;

    std::string _path;
    char _separator;
    Value_t _null_value;
    InnerDict_t _inner_dict;

};

template <typename Key_t,
          class KeyParser_t = CommonFieldParser<Key_t> >
class Set {
    typedef std::set<Key_t> InnerDict_t;
public:
    typedef typename InnerDict_t::const_iterator const_iterator;
    typedef Key_t key_type;

public:
    Set() { }

    bool create(const std::string & path) {
        _path = path;
        return _load();
    }

    virtual ~Set() { }

    bool empty() const {
        return _inner_dict.empty();
    }
    
    bool contains(const Key_t & key) const {
        return _inner_dict.end() != _inner_dict.find(key);
    }

    void insert(const Key_t & key) {
        _inner_dict.insert(key);
    }

    size_t size() const {
        return _inner_dict.size();
    }

    // read-only iterator
    const_iterator begin() const {
        return _inner_dict.begin();
    }

    const_iterator end() const {
        return _inner_dict.end();
    }

private:
    bool _load() {
        using namespace std;
        bool ret = true;
        ifstream ifs;
        ifs.open(_path.c_str());
        if (!ifs) {
            return false;
        }
        string line;
        vector<string> line_segs;
        while (ifs) {
            getline(ifs, line);
            if (line.empty()) {
                continue;
            }

            Key_t k;
            if (!key_parser(line, k)) {
                ret = false;
                continue;
            }
            _inner_dict.insert(k);
        }
        ifs.close();
        return ret;
    }

private:
    KeyParser_t key_parser;

    std::string _path;
    InnerDict_t _inner_dict;

};

template <typename Element_t,
          class ElementParser_t = CommonFieldParser<Element_t> >
class ArrayDict {
    typedef std::vector<Element_t> InnerDict_t;
public:
    typedef typename InnerDict_t::const_iterator const_iterator;
    typedef Element_t element_type;

public:
    ArrayDict() { }

    bool create(std::string path) {
        _path = path;
        return _load();
    }

    virtual ~ArrayDict() { }

    const Element_t & at(size_t idx) const {
        return _inner_dict[idx];
    }
    const Element_t & operator[](size_t idx) const {
        return at(idx);
    }

    size_t size() const {
        return _inner_dict.size();
    }

    // read-only iterator
    const_iterator begin() const {
        return _inner_dict.begin();
    }

    const_iterator end() const {
        return _inner_dict.end();
    }

private:
    bool _load() {
        using namespace std;
        bool ret = true;
        ifstream ifs;
        ifs.open(_path.c_str());
        if (!ifs) {
            return false;
        }
        string line;
        vector<string> line_segs;
        while (ifs) {
            getline(ifs, line);
            if (line.empty()) {
                continue;
            }

            Element_t k;
            if (!element_parser(line, k)) {
                ret = false;
                continue;
            }
            _inner_dict.push_back(k);
        }
        ifs.close();
        return ret;
    }

private:
    ElementParser_t element_parser;

    std::string _path;
    InnerDict_t _inner_dict;

};

class TrieDict {
public:
    TrieDict() { }

    bool create(std::string path) {
        _path = path;
        return _load();
    }

    size_t size() const {
        return trie_.num_keys();
    }

    void update(const std::string &str) {
        trie_.update(str.c_str(), str.length(), 1);
    }

    bool isMatch(const std::string& pattern) const {
        if (trie_.exactMatchSearch<int>(pattern.c_str()) > 0) {
            return true;
        }
        return false;
    }
private:
    bool _load() {
        using namespace std;
        bool ret = true;
        ifstream ifs;
        ifs.open(_path.c_str());
        if (!ifs) {
            return false;
        }
        string line;
        while (ifs) {
            getline(ifs, line);
            if (line.empty()) {
                continue;
            }

            update(line);
        }
        ifs.close();
        return ret;
    }

private:
    cedar::da<int> trie_;

    std::string _path;
};


} /* namespace utils */

