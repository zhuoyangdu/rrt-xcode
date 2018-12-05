//
//  planner.cpp
//  rrt
//
//  Created by zhuoyang on 2018/10/24.
//  Copyright © 2018年 zhuoyang. All rights reserved.
//

#include "planner.h"
#include <fstream>
#include <stdio.h>
#include <unistd.h>

using namespace std;

namespace planning {
    Planner::Planner(){
        Timer t1;
        ParamConfig();
        InitEnv();
        RegisterPlanner();
        std::cout << "environment init time:" << t1.duration() << endl;

    }
    
    
    void Planner::ParamConfig() {
        // Get configuration file path.
        std::string planning_path = "/Users/zhuoyang/workspace/rrt";
        std::string path = planning_path + "/conf/planning_conf.pb.txt";
        
        // Parse the text file into protobuf.
        using google::protobuf::TextFormat;
        using google::protobuf::io::FileInputStream;
        using google::protobuf::io::ZeroCopyInputStream;
        std::cout << path.c_str() << std::endl;
        int file_descriptor = open(path.c_str(), O_RDONLY);
        if (file_descriptor < 0) {
            std::cout << "[PlanningNode] Invalid file descriptor." << std::endl;
            return;
        }
        ZeroCopyInputStream *input = new FileInputStream(file_descriptor);
        if (!TextFormat::Parse(input, &planning_conf_)) {
            std::cout << "[PlanningNode] Failed to parse file." << std::endl;
        }
        delete input;
        close(file_descriptor);
        
        // Print configuration file.
        std::string print_conf;
        TextFormat::PrintToString(planning_conf_, &print_conf);
        std::cout << "[PlanningNode] Planning config: \n" << print_conf.c_str() << std::endl;
        

    }
    
    void Planner::InitEnv() {
        std::string map_path = "/Users/zhuoyang/workspace/rrt"
            + planning_conf_.map_path();
        cv::Mat cv_image = cv::imread(map_path, CV_8U);
        env_ = new Environment(cv_image, planning_conf_);

        vehicle_state_.x = planning_conf_.fake_state().x();
        vehicle_state_.y = planning_conf_.fake_state().y();
        vehicle_state_.theta = planning_conf_.fake_state().theta();
    }
    
    void Planner::Run() {
        auto status = rrt_planner_->MultiThreadSolve(vehicle_state_, env_);
    }
    
    void Planner::RegisterPlanner() {
        rrt_planner_.reset(new HeuristicRRT(planning_conf_));
    }
    
}  // namespace planning

