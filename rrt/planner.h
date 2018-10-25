//
//  planner.hpp
//  rrt
//
//  Created by zhuoyang on 2018/10/24.
//  Copyright © 2018年 zhuoyang. All rights reserved.
//

#ifndef SRC_PLANNING_SRC_PLANNING_NODE_H_
#define SRC_PLANNING_SRC_PLANNING_NODE_H_

#include <fcntl.h>
#include <iostream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "planning_conf.pb.h"
#include "common/vehicle_state.h"
#include "rrt/heuristic_rrt.h"
#include "common/environment.h"

namespace planning {
    class Planner {
    public:
        Planner();
        
        void Run();
        
    private:
        void ParamConfig();
        
        void InitEnv();
        
        void RegisterPlanner();
        
        planning::PlanningConf planning_conf_;
        std::unique_ptr<HeuristicRRT> rrt_planner_;
        
        VehicleState vehicle_state_;
        Environment* env_;
    };
}  // namespace planning

#endif  // SRC_PLANNING_SRC_PLANNING_NODE_H_
