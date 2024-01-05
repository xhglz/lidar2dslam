//
// Created by xiang on 2022/3/15.
//
#include "g2o_types.h"
#include "icp_2d.h"
#include "utils/math_utils.h"

#include <glog/logging.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/impl/kdtree.hpp>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

namespace sad {

bool Icp2d::AlignGaussNewton(SE2& init_pose) {
    int iterations = 10;
    double cost = 0, lastCost = 0;
    SE2 current_pose = init_pose;
    const float max_dis2 = 0.01;    // 最近邻时的最远距离（平方）
    const int min_effect_pts = 20;  // 最小有效点数

    for (int iter = 0; iter < iterations; ++iter) {
        Mat3d H = Mat3d::Zero();
        Vec3d b = Vec3d::Zero();
        cost = 0;

        int effective_num = 0;  // 有效点数

        // 遍历source
        for (size_t i = 0; i < source_scan_->ranges.size(); ++i) {
            float r = source_scan_->ranges[i];
            if (r < source_scan_->range_min || r > source_scan_->range_max) {
                continue;
            }

            float angle = source_scan_->angle_min + i * source_scan_->angle_increment;
            float theta = current_pose.so2().log();
            Vec2d pw = current_pose * Vec2d(r * std::cos(angle), r * std::sin(angle));
            Point2d pt;
            pt.x = pw.x();
            pt.y = pw.y();

            // 最近邻
            std::vector<int> nn_idx;
            std::vector<float> dis;
            kdtree_.nearestKSearch(pt, 1, nn_idx, dis);

            if (nn_idx.size() > 0 && dis[0] < max_dis2) {
                effective_num++;
                Mat32d J;
                J << 1, 0, 0, 1, -r * std::sin(angle + theta), r * std::cos(angle + theta);
                H += J * J.transpose();

                Vec2d e(pt.x - target_cloud_->points[nn_idx[0]].x, pt.y - target_cloud_->points[nn_idx[0]].y);
                b += -J * e;

                cost += e.dot(e);
            }
        }

        if (effective_num < min_effect_pts) {
            return false;
        }

        // solve for dx
        Vec3d dx = H.ldlt().solve(b);
        if (isnan(dx[0])) {
            break;
        }

        cost /= effective_num;
        if (iter > 0 && cost >= lastCost) {
            break;
        }

        LOG(INFO) << "iter " << iter << " cost = " << cost << ", effect num: " << effective_num;

        current_pose.translation() += dx.head<2>();
        current_pose.so2() = current_pose.so2() * SO2::exp(dx[2]);
        lastCost = cost;
    }

    init_pose = current_pose;
    LOG(INFO) << "estimated pose: " << current_pose.translation().transpose()
              << ", theta: " << current_pose.so2().log();

    return true;
}

bool Icp2d::AlignGaussNewtonPoint2Plane(SE2& init_pose) {
    int iterations = 10;
    double cost = 0, lastCost = 0;
    SE2 current_pose = init_pose;
    const float max_dis = 0.3;      // 最近邻时的最远距离
    const int min_effect_pts = 20;  // 最小有效点数

    for (int iter = 0; iter < iterations; ++iter) {
        Mat3d H = Mat3d::Zero();
        Vec3d b = Vec3d::Zero();
        cost = 0;

        int effective_num = 0;  // 有效点数

        // 遍历source
        for (size_t i = 0; i < source_scan_->ranges.size(); ++i) {
            float r = source_scan_->ranges[i];
            if (r < source_scan_->range_min || r > source_scan_->range_max) {
                continue;
            }

            float angle = source_scan_->angle_min + i * source_scan_->angle_increment;
            float theta = current_pose.so2().log();
            Vec2d pw = current_pose * Vec2d(r * std::cos(angle), r * std::sin(angle));
            Point2d pt;
            pt.x = pw.x();
            pt.y = pw.y();

            // 查找5个最近邻
            std::vector<int> nn_idx;
            std::vector<float> dis;
            kdtree_.nearestKSearch(pt, 5, nn_idx, dis);

            std::vector<Vec2d> effective_pts;  // 有效点
            for (int j = 0; j < nn_idx.size(); ++j) {
                if (dis[j] < max_dis) {
                    effective_pts.emplace_back(
                        Vec2d(target_cloud_->points[nn_idx[j]].x, target_cloud_->points[nn_idx[j]].y));
                }
            }

            if (effective_pts.size() < 3) {
                continue;
            }

            // 拟合直线，组装J、H和误差
            Vec3d line_coeffs;
            if (math::FitLine2D(effective_pts, line_coeffs)) {
                effective_num++;
                Vec3d J;
                J << line_coeffs[0], line_coeffs[1],
                    -line_coeffs[0] * r * std::sin(angle + theta) + line_coeffs[1] * r * std::cos(angle + theta);
                H += J * J.transpose();

                double e = line_coeffs[0] * pw[0] + line_coeffs[1] * pw[1] + line_coeffs[2];
                b += -J * e;

                cost += e * e;
            }
        }

        if (effective_num < min_effect_pts) {
            return false;
        }

        // solve for dx
        Vec3d dx = H.ldlt().solve(b);
        if (isnan(dx[0])) {
            break;
        }

        cost /= effective_num;
        if (iter > 0 && cost >= lastCost) {
            break;
        }

        LOG(INFO) << "iter " << iter << " cost = " << cost << ", effect num: " << effective_num;

        current_pose.translation() += dx.head<2>();
        current_pose.so2() = current_pose.so2() * SO2::exp(dx[2]);
        lastCost = cost;
    }

    init_pose = current_pose;
    LOG(INFO) << "estimated pose: " << current_pose.translation().transpose()
              << ", theta: " << current_pose.so2().log();

    return true;
}

void Icp2d::BuildTargetKdTree() {
    if (target_scan_ == nullptr) {
        LOG(ERROR) << "target is not set";
        return;
    }

    target_cloud_.reset(new Cloud2d);
    for (size_t i = 0; i < target_scan_->ranges.size(); ++i) {
        if (target_scan_->ranges[i] < target_scan_->range_min || target_scan_->ranges[i] > target_scan_->range_max) {
            continue;
        }

        double real_angle = target_scan_->angle_min + i * target_scan_->angle_increment;

        Point2d p;
        p.x = target_scan_->ranges[i] * std::cos(real_angle);
        p.y = target_scan_->ranges[i] * std::sin(real_angle);
        target_cloud_->points.push_back(p);
    }

    target_cloud_->width = target_cloud_->points.size();
    target_cloud_->is_dense = false;
    kdtree_.setInputCloud(target_cloud_);
}

// G2O point to point 
bool Icp2d::AlignG2OPoint2Point(SE2& init_pose) {
    // 每个误差项优化变量维度为3(x,y,theta)，误差值维度为1
    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;
    // 创建一个线性求解器LinearSolver
    using LinearSolverType = g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType>;
    // 创建总求解器solver
    auto* solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    // 稀疏优化器
    g2o::SparseOptimizer optimizer;
    // 设置求解器
    optimizer.setAlgorithm(solver); 

    // 定义图的顶点
    auto* v = new VertexSE2();
    v->setId(0);
    v->setEstimate(init_pose);
    optimizer.addVertex(v); 
    
    // 不考虑太远的scan，不准
    const double range_th = 15.0;  
    const double rk_delta = 0.8;    
    int effective_num = 0;  // 有效点数
    const int min_effect_pts = 20;  // 最小有效点数
    for (size_t i = 0; i < source_scan_->ranges.size(); ++i) {
        float r = source_scan_->ranges[i];
        if (r < source_scan_->range_min || r > source_scan_->range_max || r > range_th) {
            continue;
        }

        // 每个点的角度
        float angle = source_scan_->angle_min + i * source_scan_->angle_increment;
        if (angle < source_scan_->angle_min + 30 * M_PI / 180.0 || angle > source_scan_->angle_max - 30 * M_PI / 180.0) {
            continue;
        }

        auto e = new EdgeSE2ICPPoint2Point(kdtree_, target_cloud_, r, angle);
        e->setVertex(0, v);
        if (!e->isPointValid()) {
            delete e;
            continue;
        }

        effective_num++;
        // 信息矩阵 & 核函数 & 阈值 & 约束边
        e->setInformation(Eigen::Matrix<double, 2, 2>::Identity());
        auto rk = new g2o::RobustKernelHuber;
        rk->setDelta(rk_delta);
        e->setRobustKernel(rk);
        optimizer.addEdge(e);
    }

    if (effective_num < min_effect_pts) {
        return false;
    }

    // 调试信息
    optimizer.setVerbose(false);      
    optimizer.initializeOptimization();
    optimizer.optimize(10);      
    
    init_pose = v->estimate();
    LOG(INFO) << "estimated pose: " << init_pose.translation().transpose()
              << ", theta: " << init_pose.so2().log();
    return true;
}

// G2O point to line 
bool Icp2d::AlignG2OPoint2Line(SE2& init_pose) {
// 每个误差项优化变量维度为3(x,y,theta)，误差值维度为1
    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;
    // 创建一个线性求解器LinearSolver
    using LinearSolverType = g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType>;
    // 创建总求解器solver
    auto* solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    // 稀疏优化器
    g2o::SparseOptimizer optimizer;
    // 设置求解器
    optimizer.setAlgorithm(solver); 

    // 定义图的顶点
    auto* v = new VertexSE2();
    v->setId(0);
    v->setEstimate(init_pose);
    optimizer.addVertex(v); 
    
    // 不考虑太远的scan，不准
    const double range_th = 15.0;  
    const double rk_delta = 0.8;    
    int effective_num = 0;  // 有效点数
    const int min_effect_pts = 20;  // 最小有效点数
    for (size_t i = 0; i < source_scan_->ranges.size(); ++i) {
        float r = source_scan_->ranges[i];
        if (r < source_scan_->range_min || r > source_scan_->range_max || r > range_th) {
            continue;
        }

        // 每个点的角度
        float angle = source_scan_->angle_min + i * source_scan_->angle_increment;
        if (angle < source_scan_->angle_min + 30 * M_PI / 180.0 || angle > source_scan_->angle_max - 30 * M_PI / 180.0) {
            continue;
        }

        auto e = new EdgeSE2ICPPoint2Line(kdtree_, target_cloud_, r, angle);
        e->setVertex(0, v);
        if (!e->isLineValid()) {
            delete e;
            continue;
        }

        effective_num++;
        // 信息矩阵 & 核函数 & 阈值 & 约束边
        e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        auto rk = new g2o::RobustKernelHuber;
        rk->setDelta(rk_delta);
        e->setRobustKernel(rk);
        optimizer.addEdge(e);
    }

    if (effective_num < min_effect_pts) {
        return false;
    }

    // 调试信息
    optimizer.setVerbose(false);      
    optimizer.initializeOptimization();
    optimizer.optimize(10);      

    init_pose = v->estimate();
    LOG(INFO) << "estimated pose: " << init_pose.translation().transpose()
              << ", theta: " << init_pose.so2().log();
    return true;
}
}  // namespace sad