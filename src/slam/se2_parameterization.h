#ifndef SE2_PARAMETERIZATION_H
#define SE2_PARAMETERIZATION_H

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <sophus/se2.hpp>

// x, y, theta
class SE2Parameterization : public ceres::LocalParameterization {
public:
    bool Plus(const double *x, const double *delta, double *x_plus_delta) const override {
        // 内存映射
        // 从R,t构造SE(3)
        // 进行 exp() 操作时，平移在前，旋转在后
        Eigen::Map<const Eigen::Vector3d> _x(x);
        Eigen::Map<const Eigen::Vector3d> _delta(delta);
        Eigen::Map<Eigen::Vector3d> T_plus_delta(x_plus_delta);
        Sophus::SO2d _theta = Sophus::SO2d::exp(_x[2]) * Sophus::SO2d::exp(_delta[2]);
        T_plus_delta[0] = _x[0] + _delta[0];
        T_plus_delta[1] = _x[1] + _delta[1];
        T_plus_delta[2] = _theta.log();
        return true;
    }

    // 自动 AutoDiffCostFunction()，或者 计算雅可比矩阵，实现 costfunction
    // jacobian is a row-major GlobalSize() x LocalSize() matrix.
    // local_matrix = global_matrix * jacobian
    //
    // global_matrix is a num_rows x GlobalSize  row major matrix.
    // global_matrix是一个num_rows x GlobalSize行主矩阵 
    // local_matrix is a num_rows x LocalSize row major matrix.
    // local_matrix是一个num_rows x LocalSize行主矩阵
    // jacobian(x) is the matrix returned by ComputeJacobian at x.
    // jacobian是LocalParameterization::ComputeJacobian()在x处返回的矩阵
    // This is only used by GradientProblem. For most normal uses, it is
    // okay to use the default implementation.
    bool ComputeJacobian(const double *x, double *jacobian) const override {
        // Eigen 存储顺序，行顺序
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> j(jacobian);
        j.setIdentity();
        return true;
    }

    // 参数块 x 所在的环境空间的维度
    int GlobalSize() const override {
        return 3;
    }

    // delta 所在的切线空间的维度
    int LocalSize() const override {
        return 3;
    }
};

#endif
