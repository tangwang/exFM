#pragma once

/*设置输出前景色*/
#define PRINT_FONT_BLA()  printf("\033[30m") //黑色
#define PRINT_FONT_RED()  printf("\033[31m") //红色
#define PRINT_FONT_GRE()  printf("\033[32m") //绿色
#define PRINT_FONT_YEL()  printf("\033[33m") //黄色
#define PRINT_FONT_BLU()  printf("\033[34m") //蓝色
#define PRINT_FONT_PUR()  printf("\033[35m") //紫色
#define PRINT_FONT_CYA()  printf("\033[36m") //青色
#define PRINT_FONT_WHI()  printf("\033[37m") //白色
/*设置输出背景色*/
#define PRINT_BACK_BLA()  printf("\033[40m") //黑色
#define PRINT_BACK_RED()  printf("\033[41m") //红色
#define PRINT_BACK_GRE()  printf("\033[42m") //绿色
#define PRINT_BACK_YEL()  printf("\033[43m") //黄色
#define PRINT_BACK_BLU()  printf("\033[44m") //蓝色
#define PRINT_BACK_PUR()  printf("\033[45m") //紫色
#define PRINT_BACK_CYA()  printf("\033[46m") //青色
#define PRINT_BACK_WHI()  printf("\033[47m") //白色
/*输出属性设置*/
#define PRINT_ATTR_REC()  printf("\033[0m")  //重新设置属性到缺省设置 
#define PRINT_ATTR_BOL()  printf("\033[1m")  //设置粗体 
#define PRINT_ATTR_LIG()  printf("\033[2m")  //设置一半亮度(模拟彩色显示器的颜色) 
#define PRINT_ATTR_LIN()  printf("\033[4m")  //设置下划线(模拟彩色显示器的颜色) 
#define PRINT_ATTR_GLI()  printf("\033[5m")  //设置闪烁 
#define PRINT_ATTR_REV()  printf("\033[7m")  //设置反向图象 
#define PRINT_ATTR_THI()  printf("\033[22m") //设置一般密度 
#define PRINT_ATTR_ULIN()  printf("\033[24m")//关闭下划线 
#define PRINT_ATTR_UGLI()  printf("\033[25m")//关闭闪烁 
#define PRINT_ATTR_UREV()  printf("\033[27m")//关闭反向图象

