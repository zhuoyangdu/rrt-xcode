//
//  environment.hpp
//  rrt
//
//  Created by zhuoyang on 2018/10/24.
//  Copyright © 2018年 zhuoyang. All rights reserved.
//

// Copyright [2018] <Zhuoyang Du>

#ifndef SRC_PLANNING_SRC_COMMON_ENVIRONMENT_H_
#define SRC_PLANNING_SRC_COMMON_ENVIRONMENT_H_

#include "planning_conf.pb.h"

#include <iostream>
#include <vector>
#include <utility>

#include "image_proc.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace planning {
    class Environment {
    public:
        // The default position of the map center is (0,0,0).
        Environment() = default;
        
        Environment(const cv::Mat& image,
                    const PlanningConf& planning_conf);
        
        ~Environment() = default;
        
        void GetPixelCoord(double x, double y,
                           double* row, double* col);
        
        void GetWorldCoord(double row, double col,
                           double* x, double* y);
        
        bool CheckCollisionByPixelCoord(double row, double col) const;
        
        bool CheckCollisionByPixelCoord(const cv::Point& point) const;
        
        bool CheckCollisionByWorldCoord(double x, double y);
        
        bool CollisionCheckByEdge(const Node& a, const Node& b);
        
        cv::Mat DynamicMap() const { return map_dynamic_; }
        
        cv::Mat AttractiveMap() {return attractive_map_; }
        
        cv::Mat TargetAttractiveMap() {return goal_prob_map_;}
        
        cv::Mat VoronoiAttractiveMap() {return voronoi_prob_map_;}
        
        cv::Mat RepulsiveX() const { return repulsive_filed_x_; }
        
        cv::Mat RepulsiveY() const { return repulsive_filed_y_; }
        
        Node Goal() {
            return Node(int(pixel_goal_.x), int(pixel_goal_.y));
        }
        
        
    private:
        void  InitParams();
        
        void GenerateAttractiveProbMap();
        
        // Params.
        PlanningConf planning_conf_;
        int resolutionX_ = 512;
        int resolutionY_ = 512;
        std::pair<double, double> rangeX_;
        std::pair<double, double> rangeY_;
        
        bool is_init_ = false;
        cv::Mat map_static_;
        cv::Mat map_dynamic_;
        cv::Mat dilate_map_;
        cv::Point2d goal_;
        cv::Point2d pixel_goal_;
        
        cv::Mat attractive_map_;
        cv::Mat goal_prob_map_;
        cv::Mat voronoi_prob_map_;
        cv::Mat repulsive_filed_x_;
        cv::Mat repulsive_filed_y_;
    };
    
}  // namespace planning

#endif  // SRC_PLANNING_SRC_COMMON_ENVIRONMENT_H_
