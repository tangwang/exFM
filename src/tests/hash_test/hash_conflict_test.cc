#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>

#include "MurmurHash3.h"
#include "hash.h"

using namespace std;

int main(int argc, char** argv) {
  std::unordered_map<unsigned int, long> m;
  uint32_t hash_value;
  int hash_type = atoi(argv[1]);
  cout << " hash_type is : " << hash_type << endl;

  uint64_t hash_key;
  uint64_t insert_beg = 1000000;
  uint64_t insert_num = 1000000;
  for (uint64_t i = insert_beg; i < insert_num + insert_num; i++) {
    //hash_key = rand();
    hash_key = i;

    switch (hash_type) {
      case 1 : hash_value = MurmurHash64A(&hash_key, 8, 0);                                    break;
      case 2 : hash_value = MurmurHash64A(&hash_key, 8, 59985934289349208671);                 break;
      case 3 : hash_value = MurmurHash3_x86_32(&hash_key, 8, 59985934289349208671);            break;
      case 4 : hash_value = MurmurHash3_x86_32(&hash_key, 8, 0);                               break;
      case 5 : { uint32_t temp = hash_key; hash_value = MurmurHash3_x86_32(&hash_key, 4, 0); }        break;
      case 6 : hash_value = twang_32from64(hash_key);  /*Redis对于Key是整数类型时用的hash */   break;
      case 7 : hash_value = hsieh_hash32_buf(&hash_key, 8);  /* Paul Hsieh's hash */           break;
      case 8 : hash_value = jenkins_rev_unmix32(hash_key);                                     break;
      case 9 : hash_value = jenkins_rev_mix32(hash_key);                                       break;
      default : cout << "unknown hash_type, exit " << endl; return -1;
    }

    //hash_value = hash_value % (int(insert_num * 2));
    hash_value = hash_value % (2000003); // 取余是否素数，没什么差别（2000003 与2000000，冲突个数基本一致）
    /*
    插入100w个整数（100w到200w），如果对hash后值：
    取余2000000：
213329
213535
212658
212854
213499
212798
212947
213371
212612
    取余2000003（素数）：
213219
212340
213221
213072
212837
213178
213285
212648
212983
取余400w：
115123
115360
115237
114771
115136
115341
114996
114383
115169

取余800w：
60238
59998
59653
60072
60226
59881
60005
60290
59408

取余1600w：
30828
30530
30525
30738
30712
30534
30544
30869
30354

不取余：
11652
11531
1
1
1
11577
1047
1
1

    插入1000w个整数（1000w到2000w），如果对hash后值：
    取余2000w：
2130131
2130581
2123331
2122608
2123812
2129402
2125883
2122053
2124569

    取余2000w：
1152817
1152552
1142516
1142821
1143568
1151416
1143996
1141901
1144403


    1）取余是否素数，没什么差别（2000003 与2000000，冲突个数基本一致）
    2）各种hash方式，如果对hash后的数取余（限定hash桶数），则各种hash方法冲突概率很相近。 
       如果不取余，则冲突概率差异明显：
         Murmurhash3 hash到32位冲突概率极低（100w次hash无冲突）
        Murmurhash3 hash到64位然后取低32位：冲突较高（1.16%）
        Thomas Wang hash（redis用的） hash到64位 100w次无冲突；没有到32位的hash方法，hash到64位然后取低32位 和Murmurhash3 hash到64位然后取低32位冲突概率一致（1.16%）。
        Paul Hsieh's hash hash到32位，有冲突，冲突概率0.105%
        
    3）hash桶数（200w）为插入个数（100w）的2倍时，冲突概率为21.3%左右
       hash桶数（2000w）为插入个数（1000w）的2倍时，冲突概率为21.3%左右
       hash桶数（400w）为插入个数（100w）的4倍时，冲突概率为11.5%左右
       hash桶数（4000w）为插入个数（1000w）的4倍时，冲突概率为11.5%左右
       hash桶数（800w）为插入个数（100w）的4倍时，冲突概率为6%左右
       hash桶数（1600w）为插入个数（100w）的16倍时，冲突概率为3.5%左右
    */
    auto iter = m.find(hash_value);
    if (iter != m.end()) {
      cout << " conflict : " << hash_key << " v: " << iter->second << endl;
    } else {
      m[hash_value] = hash_key;
    }
  }

  return 0;
}
