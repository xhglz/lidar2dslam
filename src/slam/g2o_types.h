//
// Created by xiang on 2022/3/22.
//

#ifndef SLAM_IN_AUTO_DRIVING_G2O_TYPES_H
#define SLAM_IN_AUTO_DRIVING_G2O_TYPES_H

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/impl/kdtree.hpp>

#include <glog/logging.h>
#include <opencv2/core.hpp>

#include "types/eigen_types.h"
#include "utils/math_utils.h"

namespace sad {

class VertexSE2 : public g2o::BaseVertex<3, SE2> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void setToOriginImpl() override { _estimate = SE2(); }
    void oplusImpl(const double* update) override {
        _estimate.translation()[0] += update[0];
        _estimate.translation()[1] += update[1];
        _estimate.so2() = _estimate.so2() * SO2::exp(update[2]);
    }

    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }
};

class EdgeSE2LikelihoodFiled : public g2o::BaseUnaryEdge<1, double, VertexSE2> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeSE2LikelihoodFiled(const cv::Mat& field_image, double range, double angle, float resolution = 10.0)
        : field_image_(field_image), range_(range), angle_(angle), resolution_(resolution) {}

    /// 判定此条边是否在field image外面
    bool IsOutSide() {
        VertexSE2* v = (VertexSE2*)_vertices[0];
        SE2 pose = v->estimate();
        Vec2d pw = pose * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));
        Vec2i pf = (pw * resolution_ + Vec2d(field_image_.rows / 2, field_image_.cols / 2)).cast<int>();  // 图像坐标

        if (pf[0] >= image_boarder_ && pf[0] < field_image_.cols - image_boarder_ && pf[1] >= image_boarder_ &&
            pf[1] < field_image_.rows - image_boarder_) {
            return false;
        } else {
            return true;
        }
    }

    void computeError() override {
        VertexSE2* v = (VertexSE2*)_vertices[0];
        SE2 pose = v->estimate();
        Vec2d pw = pose * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));
        Vec2d pf =
            pw * resolution_ + Vec2d(field_image_.rows / 2, field_image_.cols / 2) - Vec2d(0.5, 0.5);  // 图像坐标

        if (pf[0] >= image_boarder_ && pf[0] < field_image_.cols - image_boarder_ && pf[1] >= image_boarder_ &&
            pf[1] < field_image_.rows - image_boarder_) {
            _error[0] = math::GetPixelValue<float>(field_image_, pf[0], pf[1]);
        } else {
            _error[0] = 0;
            setLevel(1);
        }
    }

    void linearizeOplus() override {
        VertexSE2* v = (VertexSE2*)_vertices[0];
        SE2 pose = v->estimate();
        float theta = pose.so2().log();
        Vec2d pw = pose * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));
        Vec2d pf =
            pw * resolution_ + Vec2d(field_image_.rows / 2, field_image_.cols / 2) - Vec2d(0.5, 0.5);  // 图像坐标

        if (pf[0] >= image_boarder_ && pf[0] < field_image_.cols - image_boarder_ && pf[1] >= image_boarder_ &&
            pf[1] < field_image_.rows - image_boarder_) {
            // 图像梯度
            float dx = 0.5 * (math::GetPixelValue<float>(field_image_, pf[0] + 1, pf[1]) -
                              math::GetPixelValue<float>(field_image_, pf[0] - 1, pf[1]));
            float dy = 0.5 * (math::GetPixelValue<float>(field_image_, pf[0], pf[1] + 1) -
                              math::GetPixelValue<float>(field_image_, pf[0], pf[1] - 1));

            _jacobianOplusXi << resolution_ * dx, resolution_ * dy,
                -resolution_ * dx * range_ * std::sin(angle_ + theta) +
                    resolution_ * dy * range_ * std::cos(angle_ + theta);
        } else {
            _jacobianOplusXi.setZero();
            setLevel(1);
        }
    }

    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }

   private:
    const cv::Mat& field_image_;
    double range_ = 0;
    double angle_ = 0;
    float resolution_ = 10.0;
    inline static const int image_boarder_ = 10;
};

/**
 * SE2 pose graph使用
 * error = v1.inv * v2 * meas.inv
 */
class EdgeSE2 : public g2o::BaseBinaryEdge<3, SE2, VertexSE2, VertexSE2> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeSE2() {}

    void computeError() override {
        VertexSE2* v1 = (VertexSE2*)_vertices[0];
        VertexSE2* v2 = (VertexSE2*)_vertices[1];
        _error = (v1->estimate().inverse() * v2->estimate() * measurement().inverse()).log();
    }

    // TODO jacobian

    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }

   private:
};

// BaseUnaryEdge，BaseBinaryEdge，BaseMultiEdge 分别表示一元边，两元边，多元边
class EdgeSE2ICPPoint2Point : public g2o::BaseUnaryEdge<2, Vec2d, VertexSE2> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Point2d = pcl::PointXY;
    using Cloud2d = pcl::PointCloud<Point2d>;
    EdgeSE2ICPPoint2Point(const pcl::search::KdTree<Point2d>& kdtree, 
        const Cloud2d::Ptr target_cloud, float range, float angle) : 
        kdtree_(kdtree),  target_cloud_(target_cloud), range_(range), angle_(angle) {}

    bool isPointValid() { 
        VertexSE2* v = (VertexSE2*)_vertices[0];
        SE2 pose = v->estimate();
        theta_ = pose.so2().log(); 
        // source point
        pw_ = pose * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));
        
        Point2d pt;
        pt.x = pw_.x();
        pt.y = pw_.y();

        kdtree_.nearestKSearch(pt, 1, nn_idx_, dis_);
        if (nn_idx_.size() > 0 && dis_[0] < max_dis) {
            // target point
            qw_ = Vec2d(target_cloud_->points[nn_idx_[0]].x, target_cloud_->points[nn_idx_[0]].y);   
            isPoint = true;
        }
        else  {
            isPoint = false;
        }
        return isPoint;
    }
    
    void computeError() override {
        if (isPointValid()) 
            _error =  pw_ - qw_; 
        else {
            _error = Vec2d(0, 0);
            // 当前边的等级设为1，不参与优化
            setLevel(1);
        }
    }

    void linearizeOplus() override {
        if (isPoint) {
            _jacobianOplusXi <<  1, 0, 0, 1,
                            -range_ * std::sin(angle_ + theta_), 
                            range_ * std::cos(angle_ + theta_);    
        } else {
            _jacobianOplusXi.setZero();
            setLevel(1);
        }                   
    }

    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }

private:
    float range_ = 0;  // 距离
    float angle_ = 0;  // 角度
    float theta_ = 0;
    Vec2d pw_, qw_;
    const pcl::search::KdTree<Point2d> kdtree_; 
    const Cloud2d::Ptr target_cloud_;
    std::vector<int> nn_idx_;    // 最近邻的索引
    std::vector<float> dis_;     // 最近邻的距离
    bool isPoint = false;
    const float max_dis = 0.01;
};

class EdgeSE2ICPPoint2Line : public g2o::BaseUnaryEdge<1, Vec2d, VertexSE2> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Point2d = pcl::PointXY;
    using Cloud2d = pcl::PointCloud<Point2d>;
    EdgeSE2ICPPoint2Line(const pcl::search::KdTree<Point2d>& kdtree, 
        const Cloud2d::Ptr target_cloud, float range, float angle) : 
        kdtree_(kdtree),  target_cloud_(target_cloud), range_(range), angle_(angle) {}

    bool isLineValid() { 
        VertexSE2* v = (VertexSE2*)_vertices[0];
        theta_ = v->estimate().so2().log(); 
        pw_ = v->estimate() * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));

        Point2d pt;
        pt.x = pw_.x();
        pt.y = pw_.y();

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
            return false;
        }

        if (math::FitLine2D(effective_pts, line_coeffs)) {
            isLine = true;
        } else {
            isLine = false;
        }
        return isLine;
    }
    
    void computeError() override {
        if (isLineValid()) 
            _error[0] = line_coeffs[0] * pw_[0] + line_coeffs[1] * pw_[1] + line_coeffs[2];
        else {
            _error[0] = 0;
            setLevel(1);
        }
    }

    void linearizeOplus() override {
        if (isLine) {
            _jacobianOplusXi << line_coeffs[0], line_coeffs[1], 
                - line_coeffs[0] * range_ * std::sin(angle_ + theta_) + line_coeffs[1] * range_ * std::cos(angle_ + theta_);
        } else {
            _jacobianOplusXi.setZero();
            setLevel(1);
        }                   
    }

    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }

private:
    float range_ = 0;  // 距离
    float angle_ = 0;  // 角度
    float theta_ = 0;
    Vec2d pw_, qw_;
    const pcl::search::KdTree<Point2d> kdtree_; 
    const Cloud2d::Ptr target_cloud_;
    Vec3d line_coeffs;
    bool isLine = false;
    const float max_dis = 0.3; 
};
    
}  // namespace sad

#endif  // SLAM_IN_AUTO_DRIVING_G2O_TYPES_H
