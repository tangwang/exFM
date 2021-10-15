#include <fstream>  // ifstream, ofstream
#include <iomanip>  // std::setw()
#include <iostream>

#include "nlohmann/json.hpp"  // https://github.com/nlohmann/json/tree/develop/single_include/nlohmann/json.hpp

using json = nlohmann::json;
using std::string;
using std::vector;

struct Player {
  string name;
  int credits;
  int ranking;
};

void to_json(nlohmann::json& j, const Player& p) {
  j = json{{"name", p.name}, {"credits", p.credits}, {"ranking", p.ranking}};
}

void from_json(const nlohmann::json& j, Player& p) {
  j.at("name").get_to(p.name);
  j.at("credits").get_to(p.credits);
  j.at("ranking").get_to(p.ranking);
}

void test_parse_str() {
  std::string s = R"(
        {
            "name": "Judd Trump",
            "credits": 1754500,
            "ranking": 1
        }
        )";
  auto j = json::parse(s);
  std::string s = j.dump();
}

void test_parse_struct() {
  auto j = R"([
            {
                "name": "Judd Trump",
                "credits": 1754500,
                "ranking": 1
            },
            {
                "name": "Neil Robertson",
                "credits": 1040500,
                "ranking": 2
            },
            {
                "name": "Ronnie O'Sullivan",
                "credits": 954500,
                "ranking": 3
            }    
            ])"_json;

  std::vector<Player> players = j.get<std::vector<Player>>();
  std::cout << "name:    " << players[2].name << std::endl;
  std::cout << "credits: " << players[2].credits << std::endl;
  std::cout << "ranking: " << players[2].ranking << std::endl;
}

void test_dump() {
  std::ifstream fin("feat_config.json");  // 注意此处是相对路径
  json j;
  fin >> j;
  fin.close();

  // 写入文件
  std::ofstream fout(
      "test_feat_config.json");  // 注意 object.json 和 config.json
                                // 内容一致，但是顺序不同
  fout << std::setw(4) << j << std::endl;
  fout.close();

  // 注意：
  // JSON标准将对象定义为“零个或多个名称 / 值对的无序集合”。
  // 如果希望保留插入顺序，可以使用tsl::ordered_map(integration)或nlohmann::fifo_map(integration)等容器专门化对象类型。
}

int main() {
  test_parse_str();
  test_parse_struct();
  test_dump();
}
