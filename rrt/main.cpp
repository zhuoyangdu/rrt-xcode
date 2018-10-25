//
//  main.cpp
//  rrt
//
//  Created by zhuoyang on 2018/10/24.
//  Copyright © 2018年 zhuoyang. All rights reserved.
//

#include <iostream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "planner.h"

int main(int argc, const char * argv[]) {
    planning::Planner planner;
    planner.Run();
    return 0;
}
