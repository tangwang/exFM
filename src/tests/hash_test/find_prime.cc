#include <iostream>

using namespace std;

#define MAX 2010000
char isPrime[MAX + 10];  //用char类型节省内存，int占4字节，char占1字节，效果都一样，因为这里用1，0代表素数与否
int main() {
  for (int i = 2; i <= MAX; ++i)  //先将所有元素都认为是素数
    isPrime[i] = 1;
  for (int i = 2; i <= MAX; ++i) {
    if (isPrime[i])  //只用标记素数的倍数
      for (int j = i * 2; j <= MAX; j += i) isPrime[j] = 0;  //将其标记为非素数
  }
  for (int i = 2; i <= MAX; ++i)  //依次输出素数
    if (isPrime[i]) cout << i << " ";
  return 0;
}
