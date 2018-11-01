//
//  environment.cpp
//  rrt
//
//  Created by zhuoyang on 2018/10/24.
//  Copyright © 2018年 zhuoyang. All rights reserved.
//

// Copyright [2018] <Zhuoyang Du>

#include "environment.h"

namespace planning {
    
    Environment::Environment(const cv::Mat& image,
                             const PlanningConf& planning_conf)
    : planning_conf_(planning_conf) {
        InitParams();
        map_static_ = image;
        map_dynamic_ = map_static_;
        GenerateAttractiveProbMap();
        
        ImageProc::GetObstacleRepulsiveField(map_static_,
                                             &repulsive_filed_x_,
                                             &repulsive_filed_y_);
        
        Mat element = cv::getStructuringElement(MORPH_RECT, Size(10, 10));
        erode(map_static_, dilate_map_, element);
        
    }
    
    void Environment::InitParams() {
        resolutionX_   = planning_conf_.vrep_conf().resolutionx();
        resolutionY_   = planning_conf_.vrep_conf().resolutiony();
        rangeX_.first  = planning_conf_.vrep_conf().minx();
        rangeX_.second = planning_conf_.vrep_conf().maxx();
        rangeY_.first  = planning_conf_.vrep_conf().miny();
        rangeY_.second = planning_conf_.vrep_conf().maxy();
        
        pixel_goal_.x = planning_conf_.goal().row();
        pixel_goal_.y = planning_conf_.goal().col();
        is_init_ = true;
    }
    
    void Environment::GetPixelCoord(double x, double y,
                                    double *row, double *col) {
        *row = (rangeY_.second - y)
        / (rangeY_.second - rangeY_.first) * resolutionY_;
        *col = (x - rangeX_.first) * resolutionX_
        / (rangeX_.second - rangeX_.first);
    }
    
    void Environment::GetWorldCoord(double row, double col,
                                    double *x, double *y) {
        *y = rangeY_.second - row / resolutionY_ * (rangeY_.second - rangeY_.first);
        *x = rangeX_.first + col / resolutionX_ * (rangeX_.second - rangeX_.first);
    }
    
    void Environment::GenerateAttractiveProbMap() {
        ImageProc::GetAttractiveProbMap(
                                        map_static_, pixel_goal_,
                                        planning_conf_.rrt_conf().k_voronoi(),
                                        planning_conf_.rrt_conf().k_goal(),
                                        &goal_prob_map_,
                                        &voronoi_prob_map_,
                                        &attractive_map_);
        std::cout << "[Environment] Generated attractive prob map" << std::endl;
    }
    
    bool Environment::CheckCollisionByPixelCoord(double row, double col) const {
        int value = static_cast<int>(dilate_map_.at<uchar>(static_cast<int>(row),
                                                           static_cast<int>(col)));
        if (value == 0) {
            // Is collided.
            return true;
        } else {
            return false;
        }
    }
    
    bool Environment::CheckCollisionByPixelCoord(const cv::Point& point) const {
        return CheckCollisionByPixelCoord(point.x, point.y);
    }
    
    bool Environment::CheckCollisionByWorldCoord(double x, double y) {
        double row, col;
        GetPixelCoord(x, y, &row, &col);
        return CheckCollisionByPixelCoord(static_cast<int>(row),
                                          static_cast<int>(col));
    }
    
    
    
}  // namespace planning
