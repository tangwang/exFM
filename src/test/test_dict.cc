#include "utils/dict.hpp"

using namespace std;
using utils::Dict;
using utils::Set;

// // 字段解析器特化，参考： 
// template<>
// class CommonFieldParser<MyFieldStruct> {
// public:
//     typedef MyFieldStruct field_type;
//     bool operator()(const std::string & str, field_type & v) {
//         using namespace std;
//         bool ret = false;
//         vector<string> field_segs;
//         boost::split(field_segs, str, boost::is_any_of(","),boost::token_compress_on);

//         try {
//             v.cat_new = boost::lexical_cast<int>(field_segs[0]);
//         } catch (...) {
//             return ret;
//         }
// }


int main()
{
    Dict<string, int> my_dict_1;
    my_dict_1.create("test_dict1.dict", ' ');

    Set<string>  my_set_1;
    my_set_1.create("test_dict3.dict");

    Dict<string, vector<int> > name_to_ids;
    Dict<int, string> id_to_name;
    id_to_name.create("test_dict2.dict", ' ');

    #define DICT_FOREACH(iter, container) \
    for (typeof((container).begin()) iter=(container).begin(); \
    iter != (container).end(); \
    ++iter)

    for (auto it = id_to_name.begin(); it != id_to_name.end(); ++it) {
        name_to_ids.get_mutable(it->second).push_back(it->first);
    }

    cout << "my_dict_1" <<endl;
    for (auto it = my_dict_1.begin(); it != my_dict_1.end(); ++it) {
        cout << it->first << " ";
        cout << it->second <<endl;
    }

    cout << "my_set_1" <<endl;
    for (auto it = my_set_1.begin(); it != my_set_1.end(); ++it) {
        cout << *it <<endl;
    }
}
