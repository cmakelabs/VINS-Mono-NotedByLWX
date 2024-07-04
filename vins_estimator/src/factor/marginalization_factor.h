#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    // 构造函数需要，cost function（约束），loss function：残差的计算方式，相关联的参数块，待边缘化的参数块的索引
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks; //优化变量参数块，细说就是跟需要被marg变量相关的参数块，全部存储到parameter_blocks数组（Vector）
    std::vector<int> drop_set; //待边缘化的优化变量id，即待边缘化的参数块在parameter_blocks中的序号

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals; //残差 IMU:15X1 视觉2X1

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    void preMarginalize();
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors; //所有观测项
    // 为了进行边缘化，先要计算边缘化状态量，以及跟边缘化状态量有关联的状态量的残差和线性化，进而构造包含边缘化信息的Hessian矩阵，其维数为m + n
    // m为将要被marg掉的维数，n为剩余状态量的维数
    // 然后将所有需要被marg的维数移到矩阵的左上角，再进行舒尔补。
    // m为要边缘化状态量的维数，n为剩余状态量的总维数
    int m, n; // m为要边缘化的变量个数，n为要保留下来的变量个数
    // 地址->global size
    std::unordered_map<long, int> parameter_block_size; //global size //<优化变量内存地址,global size>
    int sum_block_size;
    // 地址->参数排列的顺序idx
    std::unordered_map<long, int> parameter_block_idx; //local size //<待边缘化的优化变量内存地址,在parameter_block_size中的id>
    // 地址->参数块实际内容的地址
    std::unordered_map<long, double *> parameter_block_data; //<优化变量内存地址,数据>

    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;

};

// 由于边缘化的costfuntion不是固定大小的，因此只能继承最基本的类
class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
