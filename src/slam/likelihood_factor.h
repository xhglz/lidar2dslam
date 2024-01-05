#ifndef LIKELIHOOD_FACTOR_H
#define LIKELIHOOD_FACTOR_H

#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include "utils/math_utils.h"

namespace sad {

// 1 resudual 维度    3 优化变量维度，单独一块
class LikelihoodFactor : public ceres::SizedCostFunction<1, 3> {

public:
    explicit LikelihoodFactor(cv::Mat field, double range, double angle)
        : field_(std::move(field))
        , range_(std::move(range))
        , angle_(std::move(angle)) {
    }
    // 自定义残差 雅可比
    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
        Sophus::SE2d pose;
        pose.translation() = Eigen::Vector2d(parameters[0][0], parameters[0][1]);       // x, y
        pose.so2() = SO2::exp(parameters[0][2]);        
        float theta = pose.so2().log();                                                 // theta
        Vec2d pw = pose * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));
        Vec2d pf = pw * resolution_ + Vec2d(field_.rows / 2, field_.cols / 2) - Vec2d(0.5, 0.5);  // 图像坐标

        Eigen::Map<Eigen::Matrix<double, 1, 1>> error(residuals);
        if (pf[0] >= image_boarder_ && pf[0] < field_.cols - image_boarder_ && pf[1] >= image_boarder_ &&
            pf[1] < field_.rows - image_boarder_) {
            error[0] = math::GetPixelValue<float>(field_, pf[0], pf[1]);
        }

        if (jacobians) {
            if (jacobians[0]) {
                // 残差维度，优化变量维度
                Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
                jacobian_pose.setZero();
                // 图像梯度
                float dx = 0.5 * (math::GetPixelValue<float>(field_, pf[0] + 1, pf[1]) -
                                math::GetPixelValue<float>(field_, pf[0] - 1, pf[1]));
                float dy = 0.5 * (math::GetPixelValue<float>(field_, pf[0], pf[1] + 1) -
                                math::GetPixelValue<float>(field_, pf[0], pf[1] - 1));

                jacobian_pose << resolution_ * dx, resolution_ * dy,
                        -resolution_ * dx * range_ * std::sin(angle_ + theta) +
                        resolution_ * dy * range_ * std::cos(angle_ + theta);
            }
        }
        return true;
    }

private:
    cv::Mat field_; // 场函数
    double range_ = 0;
    double angle_ = 0;
    float resolution_ = 10.0;
    inline static const int image_boarder_ = 10;
};

}
#endif 
