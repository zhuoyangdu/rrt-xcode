//
//  heuristic_rrt.hpp
//  rrt
//
//  Created by zhuoyang on 2018/10/24.
//  Copyright © 2018年 zhuoyang. All rights reserved.
//

// Copyright [2018] <Zhuoyang Du>

#ifndef SRC_PLANNING_SRC_RRT_HEURISTIC_RRT_H_
#define SRC_PLANNING_SRC_RRT_HEURISTIC_RRT_H_

#include "planning_conf.pb.h"

#include <iostream>
#include <vector>
#include <memory>

#include "../common/planning_status.h"
#include "../common/environment.h"
#include "../common/image_proc.h"
#include "../common/vehicle_state.h"

#include "node.h"
#include "probablistic_map.h"
#include "gnat.h"

using namespace std;

namespace planning {

class HeuristicRRT {
public:
    HeuristicRRT() = default;
    
    explicit HeuristicRRT(const PlanningConf& planning_conf);
    
    PlanningStatus Solve(const VehicleState& vehicle_state,
                         Environment* environment);
    
private:
    struct Compare {
        Compare(Node sample) {this->sample = sample;}
        bool operator() (Node& a, Node& b) {
            int dist1 = (a.col() - sample.col()) * (a.col() - sample.col())
            + (a.row() - sample.row()) * (a.row() - sample.row());
            int dist2 = (b.col() - sample.col()) * (b.col() - sample.col())
            + (b.row() - sample.row()) * (b.row() - sample.row());
            return dist1 > dist2;
        }
        Node sample;
    };
    
    bool GetNearestNode(const Node& sample,
                        GNAT& gnat,
                        const std::vector<Node> tree,
                        Node* nearest_node);
    
    bool CheckCollision(const Node& a, const Node& b, const Environment& env);
    
    bool Steer(const Node& sample, const Node& nearest, Node* new_node);
    
    bool CheckTarget(const Node& node, const Node& goal);
    
    vector<Node> GetPath(const std::vector<Node>& tree, const Node& new_node);
    
    double PathLength(const std::vector<Node>& path);
    
    std::vector<Node> PostProcessing(const std::vector<Node>& path,
                                     const Environment* env);
    
    void Record(const std::vector<Node>& tree,
                const std::vector<Node>& spline_path,
                const std::vector<Node>& path);
    
    Node UniformSample(const Environment* environment);

    bool is_init_ = false;
    PlanningConf planning_conf_;
    RRTConf rrt_conf_;
    bool show_image_ = false;
    double shortest_path_length_ = 0;
    double shortest_spath_length_ = 0;
    std::vector<Node> min_path;
};

}  // namespace planning

#endif  // SRC_PLANNING_SRC_RRT_HEURISTIC_RRT_H_
