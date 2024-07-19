
// created by wxliu on 2024-7-8

#pragma once

#include <Eigen/Core>

#include <sym/rot3.h>

#include "../../factor/marginalization_factor.h"
#include<vector>
#include<string>
#include<map>

/**
 * @brief manually create marginalization factor.
 * 
 */


/*
using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
*/

/**
 * 
 * typedef Matrix< double, Dynamic, Dynamic >  Eigen::MatrixXd 
 * 
 * MatX H_;
 * VecX b_;
 * H_.setZero(opt_size_, opt_size_);
 * 
 * b_.setZero(opt_size_);
 * 
 */

namespace sym {

// 优化的状态量用大写字符， 常量用小写字符
/*
enum Var : char {
    POSE = 'p',        // Pose3d
    VELOCITY = 'v',    // Vector3d
    ACCEL_BIAS = 'A',  // Vector3d
    GYRO_BIAS = 'G',   // Vector3d
    GRAVITY = 'g',     // Vector3d
    EPSILON = 'e'      // Scalar
};
*/
// 可以指定枚举成员的类型(这里指定为char类型)，通过在enum后加冒号再加数据类型来指明数据类型
enum Var : char {
  VIEW = 'v',                  // Pose3d
  CALIBRATION = 'c',           // Vector4d
  POSE_PRIOR_T = 'T',          // Pose3d
  POSE_PRIOR_SQRT_INFO = 's',  // Matrix6d
  LANDMARK = 'l',              // Scalar
  LANDMARK_PRIOR = 'P',        // Scalar
  LANDMARK_PRIOR_SIGMA = 'S',  // Scalar
  MATCH_SOURCE_COORDS = 'm',   // Vector2d
  MATCH_TARGET_COORDS = 'M',   // Vector2d
  MATCH_WEIGHT = 'W',          // Scalar
  GNC_MU = 'u',                // Scalar
  GNC_SCALE = 'C',             // Scalar
  EPSILON = 'e',               // Scalar
};

constexpr size_t RESIDUAL_DIM1 = 75; // marg old frame
constexpr size_t RESIDUAL_DIM2 = 69; // marg new frame

enum EDataType
{
    EPOSE,
    EVBABG,
    EEXTRINSIC
};

enum EParameterType
{
    POSE0,
    POSE1,
    POSE2,
    POSE3,
    POSE4,
    POSE5,
    POSE6,
    POSE7,
    POSE8,
    POSE9,
    EX_POSE,
    VBABG0
};

// template <typename Scalar>
template <typename ScalarType>
struct TPose
{
    using Scalar = ScalarType;
    // using Self = MyStruct<Scalar>;

    Eigen::Matrix<Scalar, 3, 1> P;
    sym::Rot3<Scalar> Q;
};

template <typename Scalar>
struct TVBaBg
{
    // using Scalar = ScalarType;
    // using Self = MyStruct<Scalar>;

    Eigen::Matrix<Scalar, 3, 1> V;
    Eigen::Matrix<Scalar, 3, 1> Ba;
    Eigen::Matrix<Scalar, 3, 1> Bg;
};

MarginalizationInfo * g_pMarginalizationInfo; // marginalization result
// std::vector<std::string>* g_pVecKeys; // parameter order
std::vector<EParameterType>* g_pVecKeys; // parameter order
vector<double *> last_marg_para_blocks;

template <typename Scalar, int RESIDUAL_DIM>
// template <typename Scalar>
void MargFactor(const Eigen::Matrix<Scalar, 3, 1>& P0, const sym::Rot3<Scalar>& Q0, // pose 0
                const Eigen::Matrix<Scalar, 3, 1>& P1, const sym::Rot3<Scalar>& Q1, // pose 1
                const Eigen::Matrix<Scalar, 3, 1>& P2, const sym::Rot3<Scalar>& Q2,
                const Eigen::Matrix<Scalar, 3, 1>& P3, const sym::Rot3<Scalar>& Q3,
                const Eigen::Matrix<Scalar, 3, 1>& P4, const sym::Rot3<Scalar>& Q4,
                const Eigen::Matrix<Scalar, 3, 1>& P5, const sym::Rot3<Scalar>& Q5,
                const Eigen::Matrix<Scalar, 3, 1>& P6, const sym::Rot3<Scalar>& Q6,
                const Eigen::Matrix<Scalar, 3, 1>& P7, const sym::Rot3<Scalar>& Q7,
                const Eigen::Matrix<Scalar, 3, 1>& P8, const sym::Rot3<Scalar>& Q8,
                const Eigen::Matrix<Scalar, 3, 1>& P9, const sym::Rot3<Scalar>& Q9, // pose 9           
                const Eigen::Matrix<Scalar, 3, 1>& ex_P, const sym::Rot3<Scalar>& ex_Q, // extrinsic parameters
                const Eigen::Matrix<Scalar, 3, 1>& V0, // v ba bg of window 0
                const Eigen::Matrix<Scalar, 3, 1>& Ba0,
                const Eigen::Matrix<Scalar, 3, 1>& Bg0,
                // const MarginalizationInfo * MarginalizationInfo, // marginalization result
                // Eigen::Matrix<Scalar, 75, 1>* const res = nullptr,
                // Eigen::Matrix<Scalar, 75, 75>* const jacobian = nullptr,
                // Eigen::Matrix<Scalar, 75, 75>* const hessian = nullptr,
                // Eigen::Matrix<Scalar, 75, 1>* const rhs = nullptr
                Eigen::Matrix<Scalar, RESIDUAL_DIM, 1>* const res = nullptr,
                Eigen::Matrix<Scalar, RESIDUAL_DIM, RESIDUAL_DIM>* const jacobian = nullptr,
                Eigen::Matrix<Scalar, RESIDUAL_DIM, RESIDUAL_DIM>* const hessian = nullptr,
                Eigen::Matrix<Scalar, RESIDUAL_DIM, 1>* const rhs = nullptr
               )
{
    assert((*g_pVecKeys).size() == g_pMarginalizationInfo->keep_block_size.size());

    int n = g_pMarginalizationInfo->n; // 上一次边缘化保留的残差块的local size的和,也就是残差维数
    int m = g_pMarginalizationInfo->m; // 上次边缘化的被margin的残差块总和
    Eigen::VectorXd dx(n); // 用来存储残差
    // 遍历所有的剩下的有约束的残差块
    for (int i = 0; i < static_cast<int>(g_pMarginalizationInfo->keep_block_size.size()); i++)
    {
        int size = g_pMarginalizationInfo->keep_block_size[i];
        int idx = g_pMarginalizationInfo->keep_block_idx[i] - m; // idx起点统一到0

        TPose<Scalar> pose;
        TVBaBg<Scalar> vbabg;
        EDataType type = EPOSE;
        auto key = (*g_pVecKeys).at(i);
        /*std::string key = (*g_pVecKeys).at(i);
        if(key.find("pose") !=  std::string::npos)
        {
            type = EPOSE;
        }
        else
        {
            type = EVBABG;
        }*/

        switch (key)
        {
        case POSE0:
            {
                pose.P = P0;
                pose.Q = Q0;
            }        
            break;

        case POSE1:
            {
                pose.P = P1;
                pose.Q = Q1;
            }        
            break;

        case POSE2:
            {
                pose.P = P2;
                pose.Q = Q2;
            }        
            break;        
        
        case POSE3:
            {
                pose.P = P3;
                pose.Q = Q3;
            }        
            break;

        case POSE4:
            {
                pose.P = P4;
                pose.Q = Q4;
            }        
            break;

        case POSE5:
            {
                pose.P = P5;
                pose.Q = Q5;
            }        
            break;  

        case POSE6:
            {
                pose.P = P6;
                pose.Q = Q6;
            }        
            break;

        case POSE7:
            {
                pose.P = P7;
                pose.Q = Q7;
            }        
            break;

        case POSE8:
            {
                pose.P = P8;
                pose.Q = Q8;
            }        
            break;  

        case POSE9:
            {
                pose.P = P9;
                pose.Q = Q9;
            }        
            break;

        case EX_POSE:
            {
                pose.P = ex_P;
                pose.Q = ex_Q;
            }        
            break;

        case VBABG0:
            {
                vbabg.V = V0;
                vbabg.Ba = Ba0;
                vbabg.Bg = Bg0;

                type = EVBABG;
            }        
            break;  

        default:
            break;
        }

        Scalar parameter[9] = { 0 };
        switch (type)
        {
        case EPOSE:
            {
                parameter[0] = pose.P.x();
                parameter[1] = pose.P.y();
                parameter[2] = pose.P.z();
                const Eigen::Matrix<double, 4, 1>& rotation = pose.Q.Data();
                parameter[3] = rotation(0);
                parameter[4] = rotation(1);
                parameter[5] = rotation[2];
                parameter[6] = rotation(3, 0);

            }
            break;
        
        case EVBABG:
            {
                parameter[0] = vbabg.V.x();
                parameter[1] = vbabg.V.y();
                parameter[2] = vbabg.V.z();

                parameter[3] = vbabg.Ba.x();
                parameter[4] = vbabg.Ba.y();
                parameter[5] = vbabg.Ba.z();

                parameter[6] = vbabg.Bg.x();
                parameter[7] = vbabg.Bg.y();
                parameter[8] = vbabg.Bg.z();
            }
            break;

        default:
            // error
            break;
        }


        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameter, size); // 当前参数块的值
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(g_pMarginalizationInfo->keep_block_data[i], size); // 当时参数块的值
        if (size != 7)
            dx.segment(idx, size) = x - x0; // 不需要local param的直接做差
        else // 代表位姿的param
        {
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>(); // 位移直接做差
            // 旋转就是李代数做差
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            // 确保实部大于0
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }
    // 更新残差　边缘化后的先验误差 e = e0 + J * dx
    // 个人理解：根据FEJ．雅克比保持不变，但是残差随着优化会变化，因此下面不更新雅克比　只更新残差
    if(res != nullptr)
    {
        Eigen::Matrix<Scalar, RESIDUAL_DIM, 1>& _res = (*res);
        _res = g_pMarginalizationInfo->linearized_residuals + g_pMarginalizationInfo->linearized_jacobians * dx; // marg之后反解出来的残差，作为边缘化残差
    }
    
    if (jacobian != nullptr)
    {
        Eigen::Matrix<Scalar, RESIDUAL_DIM, RESIDUAL_DIM>& _jacobian = (*jacobian);
        // _jacobian.setZero();
        _jacobian = g_pMarginalizationInfo->linearized_jacobians;
    }

    if (hessian != nullptr) {
        Eigen::Matrix<Scalar, RESIDUAL_DIM, RESIDUAL_DIM>& _hessian = (*hessian);
        // _hessian.setZero();
        _hessian = g_pMarginalizationInfo->linearized_jacobians.transpose() * g_pMarginalizationInfo->linearized_jacobians;
    }

    if (rhs != nullptr) {
        Eigen::Matrix<Scalar, RESIDUAL_DIM, 1>& _rhs = (*rhs);
        _rhs = g_pMarginalizationInfo->linearized_jacobians.transpose() * (g_pMarginalizationInfo->linearized_residuals + g_pMarginalizationInfo->linearized_jacobians * dx);
    }

/*    //
    std::map<std::string, TPose> pose_map;
    // std::map<std::string, TVBaBg> vbabg_map;

    TPose pose;
    pose.P = P0;
    pose.Q = Q0;
    pose_map["pose0"] = pose;

    pose.P = P1;
    pose.Q = Q1;
    pose_map["pose1"] = pose;

    pose.P = P2;
    pose.Q = Q2;
    pose_map["pose2"] = pose;

    pose.P = P3;
    pose.Q = Q3;
    pose_map["pose3"] = pose;

    pose.P = P4;
    pose.Q = Q4;
    pose_map["pose4"] = pose;

    pose.P = P5;
    pose.Q = Q5;
    pose_map["pose5"] = pose;

    pose.P = P6;
    pose.Q = Q6;
    pose_map["pose6"] = pose;

    pose.P = P7;
    pose.Q = Q7;
    pose_map["pose7"] = pose;

    pose.P = P8;
    pose.Q = Q8;
    pose_map["pose8"] = pose;

    pose.P = P9;
    pose.Q = Q9;
    pose_map["pose9"] = pose;

    pose.P = P9;
    pose.Q = Q9;
    pose_map["ex_pose"] = pose;


    TVBaBg vbabg;
*/    
}


template <typename Scalar>
void MargOldFactor(const Eigen::Matrix<Scalar, 3, 1>& P0, const sym::Rot3<Scalar>& Q0, // pose 0
                const Eigen::Matrix<Scalar, 3, 1>& P1, const sym::Rot3<Scalar>& Q1, // pose 1
                const Eigen::Matrix<Scalar, 3, 1>& P2, const sym::Rot3<Scalar>& Q2,
                const Eigen::Matrix<Scalar, 3, 1>& P3, const sym::Rot3<Scalar>& Q3,
                const Eigen::Matrix<Scalar, 3, 1>& P4, const sym::Rot3<Scalar>& Q4,
                const Eigen::Matrix<Scalar, 3, 1>& P5, const sym::Rot3<Scalar>& Q5,
                const Eigen::Matrix<Scalar, 3, 1>& P6, const sym::Rot3<Scalar>& Q6,
                const Eigen::Matrix<Scalar, 3, 1>& P7, const sym::Rot3<Scalar>& Q7,
                const Eigen::Matrix<Scalar, 3, 1>& P8, const sym::Rot3<Scalar>& Q8,
                const Eigen::Matrix<Scalar, 3, 1>& P9, const sym::Rot3<Scalar>& Q9, // pose 9           
                const Eigen::Matrix<Scalar, 3, 1>& ex_P, const sym::Rot3<Scalar>& ex_Q, // extrinsic parameters
                const Eigen::Matrix<Scalar, 3, 1>& V0, // v ba bg of window 0
                const Eigen::Matrix<Scalar, 3, 1>& Ba0,
                const Eigen::Matrix<Scalar, 3, 1>& Bg0,
                // const MarginalizationInfo * MarginalizationInfo, // marginalization result
                Eigen::Matrix<Scalar, 75, 1>* const res = nullptr,
                Eigen::Matrix<Scalar, 75, 75>* const jacobian = nullptr,
                Eigen::Matrix<Scalar, 75, 75>* const hessian = nullptr,
                Eigen::Matrix<Scalar, 75, 1>* const rhs = nullptr
               )
{
    assert((*g_pVecKeys).size() == g_pMarginalizationInfo->keep_block_size.size());

    int n = g_pMarginalizationInfo->n; // 上一次边缘化保留的残差块的local size的和,也就是残差维数
    int m = g_pMarginalizationInfo->m; // 上次边缘化的被margin的残差块总和

    Eigen::VectorXd dx(n); // 用来存储残差
    // 遍历所有的剩下的有约束的残差块
    for (int i = 0; i < static_cast<int>(g_pMarginalizationInfo->keep_block_size.size()); i++)
    {
        int size = g_pMarginalizationInfo->keep_block_size[i];
        int idx = g_pMarginalizationInfo->keep_block_idx[i] - m; // idx起点统一到0

        TPose<Scalar> pose;
        TVBaBg<Scalar> vbabg;
        EDataType type = EPOSE;
        auto key = (*g_pVecKeys).at(i);
        /*std::string key = (*g_pVecKeys).at(i);
        if(key.find("pose") !=  std::string::npos)
        {
            type = EPOSE;
        }
        else
        {
            type = EVBABG;
        }*/

        switch (key)
        {
        case POSE0:
            {
                pose.P = P0;
                pose.Q = Q0;
            }        
            break;

        case POSE1:
            {
                pose.P = P1;
                pose.Q = Q1;
            }        
            break;

        case POSE2:
            {
                pose.P = P2;
                pose.Q = Q2;
            }        
            break;        
        
        case POSE3:
            {
                pose.P = P3;
                pose.Q = Q3;
            }        
            break;

        case POSE4:
            {
                pose.P = P4;
                pose.Q = Q4;
            }        
            break;

        case POSE5:
            {
                pose.P = P5;
                pose.Q = Q5;
            }        
            break;  

        case POSE6:
            {
                pose.P = P6;
                pose.Q = Q6;
            }        
            break;

        case POSE7:
            {
                pose.P = P7;
                pose.Q = Q7;
            }        
            break;

        case POSE8:
            {
                pose.P = P8;
                pose.Q = Q8;
            }        
            break;  

        case POSE9:
            {
                pose.P = P9;
                pose.Q = Q9;
            }        
            break;

        case EX_POSE:
            {
                pose.P = ex_P;
                pose.Q = ex_Q;
            }        
            break;

        case VBABG0:
            {
                vbabg.V = V0;
                vbabg.Ba = Ba0;
                vbabg.Bg = Bg0;

                type = EVBABG;
            }        
            break;  

        default:
            break;
        }

        Scalar parameter[9] = { 0 };
        switch (type)
        {
        case EPOSE:
            {
                parameter[0] = pose.P.x();
                parameter[1] = pose.P.y();
                parameter[2] = pose.P.z();
                const Eigen::Matrix<double, 4, 1>& rotation = pose.Q.Data();
                parameter[3] = rotation(0);
                parameter[4] = rotation(1);
                parameter[5] = rotation[2];
                parameter[6] = rotation(3, 0);

            }
            break;
        
        case EVBABG:
            {
                parameter[0] = vbabg.V.x();
                parameter[1] = vbabg.V.y();
                parameter[2] = vbabg.V.z();

                parameter[3] = vbabg.Ba.x();
                parameter[4] = vbabg.Ba.y();
                parameter[5] = vbabg.Ba.z();

                parameter[6] = vbabg.Bg.x();
                parameter[7] = vbabg.Bg.y();
                parameter[8] = vbabg.Bg.z();
            }
            break;

        default:
            // error
            break;
        }


        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameter, size); // 当前参数块的值
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(g_pMarginalizationInfo->keep_block_data[i], size); // 当时参数块的值
        if (size != 7)
            dx.segment(idx, size) = x - x0; // 不需要local param的直接做差
        else // 代表位姿的param
        {
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>(); // 位移直接做差
            // 旋转就是李代数做差
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            // 确保实部大于0
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }

    Eigen::Matrix<Scalar, 75, 75> new_jacobian;
    Eigen::Matrix<Scalar, 75, 1> new_residual;
    Eigen::Matrix<Scalar, 75, 1> new_dx;
    // for(auto key : *g_pVecKeys)
    // std::cout << "get new one:\n";
    for (int i = 0; i < static_cast<int>(g_pMarginalizationInfo->keep_block_size.size()); i++)
    {
        auto key = (*g_pVecKeys).at(i);
        int order = static_cast<int>(key);
        int size = g_pMarginalizationInfo->keep_block_size[i];
        int local_size = g_pMarginalizationInfo->localSize(size);
        int idx = g_pMarginalizationInfo->keep_block_idx[i] - m;
        new_jacobian.middleCols(order * 6, local_size) = g_pMarginalizationInfo->linearized_jacobians.middleCols(idx, local_size);

        new_residual.segment(order * 6, local_size) =  g_pMarginalizationInfo->linearized_residuals.segment(idx, local_size);
        new_dx.segment(order * 6, local_size) = dx.segment(idx, local_size);
        // std::cout << "order=" << order << " idx=" << idx << " local_size=" << local_size << "" << " para_addr=" << reinterpret_cast<long>(last_marg_para_blocks[i]) << std::endl;
    }
/*    std::cout << "old residual=\n" << g_pMarginalizationInfo->linearized_residuals.transpose() << std::endl;
    std::cout << "new residual=\n" << new_residual.transpose() << std::endl;
*/

    // 更新残差　边缘化后的先验误差 e = e0 + J * dx
    // 个人理解：根据FEJ．雅克比保持不变，但是残差随着优化会变化，因此下面不更新雅克比　只更新残差
    if(res != nullptr)
    {
        Eigen::Matrix<Scalar, 75, 1>& _res = (*res);
        // _res = g_pMarginalizationInfo->linearized_residuals + g_pMarginalizationInfo->linearized_jacobians * dx; // marg之后反解出来的残差，作为边缘化残差

        _res = new_residual + new_jacobian * new_dx;

        // _res = g_pMarginalizationInfo->linearized_residuals + new_jacobian * dx;
    }
    
    if (jacobian != nullptr)
    {
        Eigen::Matrix<Scalar, 75, 75>& _jacobian = (*jacobian);
        // _jacobian.setZero();
        // _jacobian = g_pMarginalizationInfo->linearized_jacobians;

        _jacobian = new_jacobian;
    }

    if (hessian != nullptr) {
        Eigen::Matrix<Scalar, 75, 75>& _hessian = (*hessian);
        // _hessian.setZero();
        // _hessian = g_pMarginalizationInfo->linearized_jacobians.transpose() * g_pMarginalizationInfo->linearized_jacobians;
        _hessian = new_jacobian.transpose() * new_jacobian;
    }

    if (rhs != nullptr) {
        Eigen::Matrix<Scalar, 75, 1>& _rhs = (*rhs);
        // _rhs = g_pMarginalizationInfo->linearized_jacobians.transpose() * (g_pMarginalizationInfo->linearized_residuals + g_pMarginalizationInfo->linearized_jacobians * dx);
        _rhs = new_jacobian.transpose() * (new_residual + new_jacobian * new_dx);

        // _rhs = new_jacobian.transpose() * (g_pMarginalizationInfo->linearized_residuals + new_jacobian * dx);
    }
   
}


template <typename Scalar>
void MargNewFactor(const Eigen::Matrix<Scalar, 3, 1>& P0, const sym::Rot3<Scalar>& Q0, // pose 0
                const Eigen::Matrix<Scalar, 3, 1>& P1, const sym::Rot3<Scalar>& Q1, // pose 1
                const Eigen::Matrix<Scalar, 3, 1>& P2, const sym::Rot3<Scalar>& Q2,
                const Eigen::Matrix<Scalar, 3, 1>& P3, const sym::Rot3<Scalar>& Q3,
                const Eigen::Matrix<Scalar, 3, 1>& P4, const sym::Rot3<Scalar>& Q4,
                const Eigen::Matrix<Scalar, 3, 1>& P5, const sym::Rot3<Scalar>& Q5,
                const Eigen::Matrix<Scalar, 3, 1>& P6, const sym::Rot3<Scalar>& Q6,
                const Eigen::Matrix<Scalar, 3, 1>& P7, const sym::Rot3<Scalar>& Q7,
                const Eigen::Matrix<Scalar, 3, 1>& P8, const sym::Rot3<Scalar>& Q8,
                // const Eigen::Matrix<Scalar, 3, 1>& P9, const sym::Rot3<Scalar>& Q9, // pose 9           
                const Eigen::Matrix<Scalar, 3, 1>& ex_P, const sym::Rot3<Scalar>& ex_Q, // extrinsic parameters
                const Eigen::Matrix<Scalar, 3, 1>& V0, // v ba bg of window 0
                const Eigen::Matrix<Scalar, 3, 1>& Ba0,
                const Eigen::Matrix<Scalar, 3, 1>& Bg0,
                // const MarginalizationInfo * MarginalizationInfo, // marginalization result
                Eigen::Matrix<Scalar, 69, 1>* const res = nullptr,
                Eigen::Matrix<Scalar, 69, 69>* const jacobian = nullptr,
                Eigen::Matrix<Scalar, 69, 69>* const hessian = nullptr,
                Eigen::Matrix<Scalar, 69, 1>* const rhs = nullptr
               )
{
    assert((*g_pVecKeys).size() == g_pMarginalizationInfo->keep_block_size.size());

    int n = g_pMarginalizationInfo->n; // 上一次边缘化保留的残差块的local size的和,也就是残差维数
    int m = g_pMarginalizationInfo->m; // 上次边缘化的被margin的残差块总和
    Eigen::VectorXd dx(n); // 用来存储残差
    // 遍历所有的剩下的有约束的残差块
    for (int i = 0; i < static_cast<int>(g_pMarginalizationInfo->keep_block_size.size()); i++)
    {
        int size = g_pMarginalizationInfo->keep_block_size[i];
        int idx = g_pMarginalizationInfo->keep_block_idx[i] - m; // idx起点统一到0

        TPose<Scalar> pose;
        TVBaBg<Scalar> vbabg;
        EDataType type = EPOSE;
        auto key = (*g_pVecKeys).at(i);
        /*std::string key = (*g_pVecKeys).at(i);
        if(key.find("pose") !=  std::string::npos)
        {
            type = EPOSE;
        }
        else
        {
            type = EVBABG;
        }*/

        switch (key)
        {
        case POSE0:
            {
                pose.P = P0;
                pose.Q = Q0;
            }        
            break;

        case POSE1:
            {
                pose.P = P1;
                pose.Q = Q1;
            }        
            break;

        case POSE2:
            {
                pose.P = P2;
                pose.Q = Q2;
            }        
            break;        
        
        case POSE3:
            {
                pose.P = P3;
                pose.Q = Q3;
            }        
            break;

        case POSE4:
            {
                pose.P = P4;
                pose.Q = Q4;
            }        
            break;

        case POSE5:
            {
                pose.P = P5;
                pose.Q = Q5;
            }        
            break;  

        case POSE6:
            {
                pose.P = P6;
                pose.Q = Q6;
            }        
            break;

        case POSE7:
            {
                pose.P = P7;
                pose.Q = Q7;
            }        
            break;

        case POSE8:
            {
                pose.P = P8;
                pose.Q = Q8;
            }        
            break;  

        // case POSE9:
        //     {
        //         pose.P = P9;
        //         pose.Q = Q9;
        //     }        
        //     break;

        case EX_POSE:
            {
                pose.P = ex_P;
                pose.Q = ex_Q;
            }        
            break;

        case VBABG0:
            {
                vbabg.V = V0;
                vbabg.Ba = Ba0;
                vbabg.Bg = Bg0;

                type = EVBABG;
            }        
            break;  

        default:
            break;
        }

        Scalar parameter[9] = { 0 };
        switch (type)
        {
        case EPOSE:
            {
                parameter[0] = pose.P.x();
                parameter[1] = pose.P.y();
                parameter[2] = pose.P.z();
                const Eigen::Matrix<double, 4, 1>& rotation = pose.Q.Data();
                parameter[3] = rotation(0);
                parameter[4] = rotation(1);
                parameter[5] = rotation[2];
                parameter[6] = rotation(3, 0);

            }
            break;
        
        case EVBABG:
            {
                parameter[0] = vbabg.V.x();
                parameter[1] = vbabg.V.y();
                parameter[2] = vbabg.V.z();

                parameter[3] = vbabg.Ba.x();
                parameter[4] = vbabg.Ba.y();
                parameter[5] = vbabg.Ba.z();

                parameter[6] = vbabg.Bg.x();
                parameter[7] = vbabg.Bg.y();
                parameter[8] = vbabg.Bg.z();
            }
            break;

        default:
            // error
            break;
        }


        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameter, size); // 当前参数块的值
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(g_pMarginalizationInfo->keep_block_data[i], size); // 当时参数块的值
        if (size != 7)
            dx.segment(idx, size) = x - x0; // 不需要local param的直接做差
        else // 代表位姿的param
        {
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>(); // 位移直接做差
            // 旋转就是李代数做差
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            // 确保实部大于0
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }

    Eigen::Matrix<Scalar, 69, 69> new_jacobian;
    Eigen::Matrix<Scalar, 69, 1> new_residual;
    Eigen::Matrix<Scalar, 69, 1> new_dx;
    // for(auto key : *g_pVecKeys)
    for (int i = 0; i < static_cast<int>(g_pMarginalizationInfo->keep_block_size.size()); i++)
    {
        auto key = (*g_pVecKeys).at(i);
        int order = static_cast<int>(key);
        int size = g_pMarginalizationInfo->keep_block_size[i];
        int local_size = g_pMarginalizationInfo->localSize(size);
        int idx = g_pMarginalizationInfo->keep_block_idx[i] - m;

        if(order > 8)
        {
            order -= 1;
            new_jacobian.middleCols(order * 6, local_size) = g_pMarginalizationInfo->linearized_jacobians.middleCols(idx, local_size);
            new_residual.segment(order * 6, local_size) =  g_pMarginalizationInfo->linearized_residuals.segment(idx, local_size);
            new_dx.segment(order * 6, local_size) = dx.segment(idx, local_size);
        }
        
    }/**/

    // 更新残差　边缘化后的先验误差 e = e0 + J * dx
    // 个人理解：根据FEJ．雅克比保持不变，但是残差随着优化会变化，因此下面不更新雅克比　只更新残差
    if(res != nullptr)
    {
        Eigen::Matrix<Scalar, 69, 1>& _res = (*res);
        // _res = g_pMarginalizationInfo->linearized_residuals + g_pMarginalizationInfo->linearized_jacobians * dx; // marg之后反解出来的残差，作为边缘化残差
        _res = new_residual + new_jacobian * new_dx;
        // _res = g_pMarginalizationInfo->linearized_residuals + new_jacobian * dx;
    }
    
    if (jacobian != nullptr)
    {
        Eigen::Matrix<Scalar, 69, 69>& _jacobian = (*jacobian);
        // _jacobian.setZero();
        // _jacobian = g_pMarginalizationInfo->linearized_jacobians;
        _jacobian = new_jacobian;

    }

    if (hessian != nullptr) {
        Eigen::Matrix<Scalar, 69, 69>& _hessian = (*hessian);
        // _hessian.setZero();
        // _hessian = g_pMarginalizationInfo->linearized_jacobians.transpose() * g_pMarginalizationInfo->linearized_jacobians;
        _hessian = new_jacobian.transpose() * new_jacobian;
    }

    if (rhs != nullptr) {
        Eigen::Matrix<Scalar, 69, 1>& _rhs = (*rhs);
        // _rhs = g_pMarginalizationInfo->linearized_jacobians.transpose() * (g_pMarginalizationInfo->linearized_residuals + g_pMarginalizationInfo->linearized_jacobians * dx);
        _rhs = new_jacobian.transpose() * (new_residual + new_jacobian * new_dx);
        // _rhs = new_jacobian.transpose() * (g_pMarginalizationInfo->linearized_residuals + new_jacobian * dx);
    }
    
}

} // namespace sym