//
//  image_proc.hpp
//  rrt
//
//  Created by zhuoyang on 2018/10/24.
//  Copyright © 2018年 zhuoyang. All rights reserved.
//

#ifndef SRC_PLANNING_SRC_COMMON_IMAGE_PROC_H_
#define SRC_PLANNING_SRC_COMMON_IMAGE_PROC_H_

#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <set>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../rrt/node.h"

using namespace cv;

namespace planning {
    class ImageProc {
    public:
        ImageProc() = delete;
        
        static std::vector<Point> GetVertex(const cv::Mat& image);
        
        static cv::Mat GetVoronoiProbMap(const cv::Mat& image);
        
        static cv::Mat GetTargetAttractiveMap(
                                              const cv::Mat& image, const cv::Point& goal);
        
        static cv::Mat GetAttractiveProbMap(
                                            const cv::Mat& image, const cv::Point& goal,
                                            double k_voronoi, double k_goal);
        
        static void GetAttractiveProbMap(
                                         const cv::Mat& image, const cv::Point& goal,
                                         double k_voronoi, double k_goal,
                                         cv::Mat* goal_prob_map,
                                         cv::Mat* voronoi_prob_map,
                                         cv::Mat* attractive_prob_map);
        
        static void GetObstacleRepulsiveField(const cv::Mat& image,
                                              cv::Mat* repulsive_filed_x,
                                              cv::Mat* repulsive_filed_y);
        
        static void PlotPoint(const cv::Mat& image,
                              const Point& point,
                              const cv::Scalar& scalar,
                              double thickness);
        
        static void PlotPoint(const cv::Mat& image,
                              const Node& node,
                              const cv::Scalar& scalar,
                              double thickness);
        
        static void PlotLine(const cv::Mat& image,
                             const Node& a,
                             const Node& b,
                             const cv::Scalar& scalar,
                             double thickness);
        
        static void PlotPath(const cv::Mat& image,
                             const std::vector<Node> path,
                             const cv::Scalar& scalar,
                             double thickness);
        
        static void PlotPath(const cv::Mat& image,
                             const std::vector<double> x,
                             const std::vector<double> y,
                             const cv::Scalar& scalar,
                             double thickness);
    };
}  // namespace planning
#endif  // SRC_PLANNING_SRC_COMMON_IMAGE_PROC_H_
