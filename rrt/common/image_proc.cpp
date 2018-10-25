//
//  image_proc.cpp
//  rrt
//
//  Created by zhuoyang on 2018/10/24.
//  Copyright © 2018年 zhuoyang. All rights reserved.
//

#include "image_proc.h"

using namespace std;
using namespace cv;

namespace planning {
    void ImageProc::PlotPoint(const cv::Mat& image,
                              const Point& point,
                              const cv::Scalar& scalar,
                              double thickness) {
        circle(image, point, thickness, scalar, CV_FILLED, CV_AA, 0);
    }
    
    void ImageProc::PlotPoint(const cv::Mat& image,
                              const Node& node,
                              const cv::Scalar& scalar,
                              double thickness) {
        PlotPoint(image, cv::Point(node.col(), node.row()), scalar, thickness);
    }
    
    void ImageProc::PlotLine(const cv::Mat& image,
                             const Node& a,
                             const Node& b,
                             const cv::Scalar& scalar,
                             double thickness) {
        line(image, cv::Point(a.col(), a.row()),
             cv::Point(b.col(), b.row()), scalar, thickness);
    }
    
    void ImageProc::PlotPath(const cv::Mat& image,
                             const std::vector<Node> path,
                             const cv::Scalar& scalar,
                             double thickness) {
        for (int i = 0; i < path.size()-1; ++i) {
            PlotLine(image, path[i], path[i+1], scalar, thickness);
        }
    }
    
    void ImageProc::PlotPath(const cv::Mat& image,
                             const std::vector<double> x,
                             const std::vector<double> y,
                             const cv::Scalar& scalar,
                             double thickness) {
        for (int i = 0; i < x.size()-1; ++i) {
            line(image, cv::Point(y[i], x[i]), cv::Point(y[i+1], x[i+1]), scalar, thickness);
        }
    }
    
    std::vector<Point> ImageProc::GetVertex(const cv::Mat &image) {
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        cv::findContours(image, contours, hierarchy,
                         RETR_TREE, CV_CHAIN_APPROX_NONE);
        
        vector<Point> vertex;
        for (vector<Point> contour : contours) {
            for (int i = 0; i < contour.size(); i = i+2) {
                vertex.push_back(contour[i]);
            }
        }
        return vertex;
    }
    
    cv::Mat ImageProc::GetVoronoiProbMap(const cv::Mat& image) {
        // Get the vertex point of the image.
        cv::Size size = image.size();
        cv::Rect rect(0, 0, size.width, size.height);
        cv::Subdiv2D subdiv(rect);
        std::vector<cv::Point> vertex = GetVertex(image);
        for (cv::Point point : vertex) {
            subdiv.insert(point);
        }
        
        // Get Voronoi map of image.
        vector<vector<Point2f> > facets;
        vector<Point2f> centers;
        subdiv.getVoronoiFacetList(vector<int>(), facets, centers);
        Mat img_vor(image.rows, image.cols, CV_8UC1, Scalar(255));
        vector<Point2f> circle_center;
        for (vector<Point2f> p1 : facets) {
            for (Point2f p2 : p1) {
                if (p2.x > 511 || p2.x < 0 || p2.y > 511 || p2.y < 0) {
                    continue;
                }
                int value = static_cast<int>(image.at<uchar>(p2));
                if (value > 0) {
                    circle(img_vor, p2, 3, Scalar(0), CV_FILLED, CV_AA, 0);
                } else {
                    continue;
                }
                circle_center.push_back(p2);
            }
        }
        
        int kern_dim = 50;
        cv::Mat_<double> kern(kern_dim*2+1, kern_dim*2+1);
        for (int i = 0; i < kern_dim*2+1; ++i) {
            for (int j = 0; j < kern_dim*2+1; ++j) {
                int den = (i - kern_dim) * (i - kern_dim)
                + (j - kern_dim) * (j - kern_dim);
                double k = (den == 0) ? 1.0 : 1.0 / sqrt(static_cast<double>(den));
                kern.at<double>(i, j) = k;
            }
        }
        double k_min, k_max;
        cv::minMaxIdx(kern, &k_min, &k_max);
        cout << "kmin:" << k_min << ", kmax:" << k_max << endl;
        
        cv::Mat filter_image(image.rows, image.cols, CV_8U);
        filter2D((255-img_vor) / 255 , filter_image, CV_8U, kern);
        
        cv::Mat voronoi_prob(image.rows, image.cols, CV_8U);
        double min, max;
        cv::minMaxIdx(filter_image, &min, &max);
        for (int i = 0; i < voronoi_prob.rows; ++i) {
            for (int j = 0; j < voronoi_prob.cols; ++j) {
                voronoi_prob.at<uchar>(i, j) =
                static_cast<int>(filter_image.at<uchar>(i, j) * 255 / max);
            }
        }
        cv::minMaxIdx(voronoi_prob, &min, &max);
        std::cout << "GetVoronoiProbMap : min:" << min
        << ", max:" << max << std::endl;
        
        return voronoi_prob;
    }
    
    cv::Mat ImageProc::GetTargetAttractiveMap(
                                              const cv::Mat& image, const cv::Point& goal) {
        cv::Mat goal_prob(image.rows, image.cols, CV_8UC1);
        for (int i = 0; i < goal_prob.rows; ++i) {
            for (int j = 0; j < goal_prob.cols; ++j) {
                goal_prob.at<uchar>(i, j) =
                int(255.0 / (sqrt(0.5 * ((goal.x - i) * (goal.x - i)
                                         + (goal.y - j) * (goal.y - j))) + 1));
            }
        }
        double min, max;
        cv::minMaxIdx(goal_prob, &min, &max);
        std::cout << "GetTargetAttractiveMap: min:"
        << min << ", max:" << max << std::endl;
        return  goal_prob;
    }
    
    void ImageProc::GetAttractiveProbMap(
                                         const cv::Mat& image, const cv::Point& goal,
                                         double k_voronoi, double k_goal,
                                         cv::Mat* goal_prob_map,
                                         cv::Mat* voronoi_prob_map,
                                         cv::Mat* attractive_prob_map) {
        *voronoi_prob_map
        = planning::ImageProc::GetVoronoiProbMap(image);
        *goal_prob_map
        = planning::ImageProc::GetTargetAttractiveMap(image, goal);
        
        cv::Mat_<double> attractive_prob_double(cv::Size(image.rows, image.cols));
        double max_prob = 0.0;
        for (int i = 0; i < attractive_prob_double.rows; ++i) {
            for (int j = 0; j < attractive_prob_double.cols; ++j) {
                double value = k_voronoi * voronoi_prob_map->at<uchar>(i, j) +
                k_goal * goal_prob_map->at<uchar>(i, j);
                attractive_prob_double.at<double>(i, j) = value;
                max_prob = (max_prob > value) ? max_prob : value;
            }
        }
        cv::Mat attractive_prob(image.rows, image.cols, CV_8U);
        for (int i = 0; i < attractive_prob_double.rows; ++i) {
            for (int j = 0; j < attractive_prob_double.cols; ++j) {
                attractive_prob.at<uchar>(i, j) =
                int(attractive_prob_double.at<double>(i, j) / max_prob * 255);
            }
        }
        double min, max;
        cv::minMaxIdx(attractive_prob, &min, &max);
        std::cout << "GetAttractiveProbMap: min:"
        << min << ", max:" << max << std::endl;
        
        // Remove collision.
        for (int i = 0; i < attractive_prob.rows; ++i) {
            for (int j = 0; j < attractive_prob.cols; ++j) {
                if (static_cast<int>(image.at<uchar>(i, j)) <= 0) {
                    attractive_prob.at<uchar>(i, j) = 0;
                }
            }
        }
        *attractive_prob_map = attractive_prob;
    }
    
    void ImageProc::GetObstacleRepulsiveField(const cv::Mat& image,
                                              cv::Mat* repulsive_filed_x,
                                              cv::Mat* repulsive_filed_y) {
        std::vector<cv::Point> vertex = GetVertex(image);
        cv::Mat img_cont = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
        for (cv::Point p : vertex) {
            PlotPoint(img_cont, p, Scalar(255), 2);
        }
        // imshow("contours", img_cont);
        
        int kern_dim = 50;
        cv::Mat kern_x = cv::Mat::zeros(kern_dim*2+1, kern_dim*2+1, CV_64F);
        cv::Mat kern_y = cv::Mat::zeros(kern_dim*2+1, kern_dim*2+1, CV_64F);
        cv::Mat kern = cv::Mat::zeros(kern_dim*2+1, kern_dim*2+1, CV_64F);
        for (int i = 0; i < kern_dim*2+1; ++i) {
            for (int j = 0; j < kern_dim*2+1; ++j) {
                if (i == kern_dim && j == kern_dim) {
                    kern_x.at<double>(i,j) = 0.0;
                    kern_y.at<double>(i,j) = 0.0;
                } else {
                    double num = pow(pow(kern_dim-i, 2) + pow(kern_dim-j, 2), 2);
                    kern_x.at<double>(i,j) = (kern_dim - i) / num;
                    kern_y.at<double>(i,j) = (kern_dim - j) / num;
                }
            }
        }
        
        filter2D(img_cont, *repulsive_filed_x, CV_64F, kern_x);
        filter2D(img_cont, *repulsive_filed_y, CV_64F, kern_y);
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                if (image.at<uchar>(i,j) == 0) {
                    repulsive_filed_x->at<double>(i,j) = 255.0;
                    repulsive_filed_y->at<double>(i,j) = 255.0;
                }
                if (img_cont.at<uchar>(i,j) == 255) {
                    repulsive_filed_x->at<double>(i,j) = 255.0;
                    repulsive_filed_y->at<double>(i,j) = 255.0;
                }
            }
        }
        double min, max;
        cv::minMaxIdx(*repulsive_filed_x , &min, &max);
        std::cout << "repulsive_filed_x: min:"
        << min << ", max:" << max << std::endl;
        
        cv::minMaxIdx(*repulsive_filed_y , &min, &max);
        std::cout << "repulsive_filed_y: min:"
        << min << ", max:" << max << std::endl;
        
    }

}  // namespace planning

