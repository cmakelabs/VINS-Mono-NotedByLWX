#pragma once
#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../utility/utility.h"
#include "../parameters.h"
#include "integration_base.h"

#include <ceres/ceres.h>

class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9>
{
  public:
    IMUFactor() = delete;
    IMUFactor(IntegrationBase* _pre_integration):pre_integration(_pre_integration)
    {
    }
    virtual bool Evaluate1(double const *const *parameters, double *residuals, double **jacobians) const
    {

        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
        Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

        Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Quaterniond Qj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

        Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
        Eigen::Vector3d Baj(parameters[3][3], parameters[3][4], parameters[3][5]);
        Eigen::Vector3d Bgj(parameters[3][6], parameters[3][7], parameters[3][8]);

        // auto start = std::chrono::steady_clock::now();

//Eigen::Matrix<double, 15, 15> Fd;
//Eigen::Matrix<double, 15, 12> Gd;

//Eigen::Vector3d pPj = Pi + Vi * sum_t - 0.5 * g * sum_t * sum_t + corrected_delta_p;
//Eigen::Quaterniond pQj = Qi * delta_q;
//Eigen::Vector3d pVj = Vi - g * sum_t + corrected_delta_v;
//Eigen::Vector3d pBaj = Bai;
//Eigen::Vector3d pBgj = Bgi;

//Vi + Qi * delta_v - g * sum_dt = Vj;
//Qi * delta_q = Qj;

//delta_p = Qi.inverse() * (0.5 * g * sum_dt * sum_dt + Pj - Pi);
//delta_v = Qi.inverse() * (g * sum_dt + Vj - Vi);
//delta_q = Qi.inverse() * Qj;

#if 0
        if ((Bai - pre_integration->linearized_ba).norm() > 0.10 ||
            (Bgi - pre_integration->linearized_bg).norm() > 0.01)
        {
            pre_integration->repropagate(Bai, Bgi);
        }
#endif

        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
        residual = pre_integration->evaluate(Pi, Qi, Vi, Bai, Bgi,
                                            Pj, Qj, Vj, Baj, Bgj);

        Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration->covariance.inverse()).matrixL().transpose();
        //sqrt_info.setIdentity();
        residual = sqrt_info * residual;

        if (jacobians)
        {
            double sum_dt = pre_integration->sum_dt;
            Eigen::Matrix3d dp_dba = pre_integration->jacobian.template block<3, 3>(O_P, O_BA);
            Eigen::Matrix3d dp_dbg = pre_integration->jacobian.template block<3, 3>(O_P, O_BG);

            Eigen::Matrix3d dq_dbg = pre_integration->jacobian.template block<3, 3>(O_R, O_BG);

            Eigen::Matrix3d dv_dba = pre_integration->jacobian.template block<3, 3>(O_V, O_BA);
            Eigen::Matrix3d dv_dbg = pre_integration->jacobian.template block<3, 3>(O_V, O_BG);

            if (pre_integration->jacobian.maxCoeff() > 1e8 || pre_integration->jacobian.minCoeff() < -1e8)
            {
                ROS_WARN("numerical unstable in preintegration");
                //std::cout << pre_integration->jacobian << std::endl;
///                ROS_BREAK();
            }

            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();

                jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
                jacobian_pose_i.block<3, 3>(O_P, O_R) = Utility::skewSymmetric(Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));

#if 0
            jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Qj.inverse() * Qi).toRotationMatrix();
#else
                Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Utility::Qleft(Qj.inverse() * Qi) * Utility::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();
#endif

                jacobian_pose_i.block<3, 3>(O_V, O_R) = Utility::skewSymmetric(Qi.inverse() * (G * sum_dt + Vj - Vi));

                jacobian_pose_i = sqrt_info * jacobian_pose_i;

                if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in preintegration");
                    //std::cout << sqrt_info << std::endl;
                    //ROS_BREAK();
                }
            }
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);
                jacobian_speedbias_i.setZero();
                jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
                jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
                jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;

#if 0
            jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -dq_dbg;
#else
                //Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                //jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi * corrected_delta_q).bottomRightCorner<3, 3>() * dq_dbg;
                jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi * pre_integration->delta_q).bottomRightCorner<3, 3>() * dq_dbg;
#endif

                jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix();
                jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
                jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;

                jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix3d::Identity();

                jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix3d::Identity();

                jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;

                //ROS_ASSERT(fabs(jacobian_speedbias_i.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_speedbias_i.minCoeff()) < 1e8);
            }
            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]);
                jacobian_pose_j.setZero();

                jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();

#if 0
            jacobian_pose_j.block<3, 3>(O_R, O_R) = Eigen::Matrix3d::Identity();
#else
                Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                jacobian_pose_j.block<3, 3>(O_R, O_R) = Utility::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
#endif

                jacobian_pose_j = sqrt_info * jacobian_pose_j;

                //ROS_ASSERT(fabs(jacobian_pose_j.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_pose_j.minCoeff()) < 1e8);
            }
            if (jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[3]);
                jacobian_speedbias_j.setZero();

                jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix();

                jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity();

                jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();

                jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;

                //ROS_ASSERT(fabs(jacobian_speedbias_j.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_speedbias_j.minCoeff()) < 1e8);
            }
        }

        // auto end = std::chrono::steady_clock::now();
        // std::chrono::duration<double> diff = end - start;
        // std::cout << std::fixed << "sym::imu_factor: " << diff.count() << " s\n";
        // std::cout << std::fixed << "sym::imu_factor: " << std::chrono::duration_cast<std::chrono::milliseconds>((end - start).count()) << " ms\n";
        // auto diff = end - start;
        // std::cout << "imu_factor: " << std::chrono::duration <double, std::milli>(diff).count() << " ms" << std::endl;

        return true;
    }

    // 实现了用符号表达式来计算residual和jacobian
    using Scalar = double;
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
        Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

        Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Quaterniond Qj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

        Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
        Eigen::Vector3d Baj(parameters[3][3], parameters[3][4], parameters[3][5]);
        Eigen::Vector3d Bgj(parameters[3][6], parameters[3][7], parameters[3][8]);

        // auto start = std::chrono::steady_clock::now();

        // from preintegration
        const Eigen::Matrix3d dp_dba = pre_integration->jacobian.block<3, 3>(O_P, O_BA);
        const Eigen::Matrix3d dp_dbg = pre_integration->jacobian.block<3, 3>(O_P, O_BG);

        const Eigen::Matrix3d dq_dbg = pre_integration->jacobian.block<3, 3>(O_R, O_BG);

        const Eigen::Matrix3d dv_dba = pre_integration->jacobian.block<3, 3>(O_V, O_BA);
        const Eigen::Matrix3d dv_dbg = pre_integration->jacobian.block<3, 3>(O_V, O_BG);

        // const Eigen::Vector3d dba = Bai - pre_integration->linearized_ba;
        // const Eigen::Vector3d dbg = Bgi - pre_integration->linearized_bg;

        const Eigen::Vector3d linearized_ba = pre_integration->linearized_ba;
        const Eigen::Vector3d linearized_bg = pre_integration->linearized_bg;

        const Eigen::Quaterniond delta_q = pre_integration->delta_q;
        const Eigen::Vector3d delta_v = pre_integration->delta_v;
        const Eigen::Vector3d delta_p = pre_integration->delta_p;

        const Scalar sum_dt = pre_integration->sum_dt;
        // const Eigen::Matrix<Scalar, 3, 1>& G = pre_integration->G
        Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration->covariance.inverse()).matrixL().transpose();
        // the end.

        // Input arrays
        const Eigen::Matrix<Scalar, 4, 1>& _Qi = Qi.coeffs();
        const Eigen::Matrix<Scalar, 4, 1>& _Qj = Qj.coeffs();
        const Eigen::Matrix<Scalar, 4, 1>& _delta_q = delta_q.coeffs();

        ////////////////////////////////////////
        // Intermediate terms (457)
        const Scalar _tmp0 = std::pow(_Qi[2], Scalar(2));
        const Scalar _tmp1 = -2 * _tmp0;
        const Scalar _tmp2 = std::pow(_Qi[1], Scalar(2));
        const Scalar _tmp3 = -2 * _tmp2;
        const Scalar _tmp4 = _tmp1 + _tmp3 + 1;
        const Scalar _tmp5 = std::pow(sum_dt, Scalar(2));
        const Scalar _tmp6 = Scalar(0.5) * _tmp5;
        const Scalar _tmp7 = G(0, 0) * _tmp6 - Pi(0, 0) + Pj(0, 0) - Vi(0, 0) * sum_dt;
        const Scalar _tmp8 = Bgi(0, 0) - linearized_bg(0, 0);
        const Scalar _tmp9 = Bgi(1, 0) - linearized_bg(1, 0);
        const Scalar _tmp10 = Bgi(2, 0) - linearized_bg(2, 0);
        const Scalar _tmp11 = Bai(1, 0) - linearized_ba(1, 0);
        const Scalar _tmp12 = Bai(0, 0) - linearized_ba(0, 0);
        const Scalar _tmp13 = Bai(2, 0) - linearized_ba(2, 0);
        const Scalar _tmp14 = 2 * _Qi[1];
        const Scalar _tmp15 = _Qi[3] * _tmp14;
        const Scalar _tmp16 = -_tmp15;
        const Scalar _tmp17 = 2 * _Qi[0];
        const Scalar _tmp18 = _Qi[2] * _tmp17;
        const Scalar _tmp19 = _tmp16 + _tmp18;
        const Scalar _tmp20 = G(2, 0) * _tmp6 - Pi(2, 0) + Pj(2, 0) - Vi(2, 0) * sum_dt;
        const Scalar _tmp21 = _Qi[1] * _tmp17;
        const Scalar _tmp22 = 2 * _Qi[2] * _Qi[3];
        const Scalar _tmp23 = _tmp21 + _tmp22;
        const Scalar _tmp24 = G(1, 0) * _tmp6 - Pi(1, 0) + Pj(1, 0) - Vi(1, 0) * sum_dt;
        const Scalar _tmp25 = _tmp19 * _tmp20 + _tmp23 * _tmp24;
        const Scalar _tmp26 = -_tmp10 * dp_dbg(0, 2) - _tmp11 * dp_dba(0, 1) - _tmp12 * dp_dba(0, 0) -
                                _tmp13 * dp_dba(0, 2) + _tmp25 + _tmp4 * _tmp7 - _tmp8 * dp_dbg(0, 0) -
                                _tmp9 * dp_dbg(0, 1) - delta_p(0, 0);
        const Scalar _tmp27 = std::pow(_Qi[0], Scalar(2));
        const Scalar _tmp28 = 1 - 2 * _tmp27;
        const Scalar _tmp29 = _tmp1 + _tmp28;
        const Scalar _tmp30 = _Qi[2] * _tmp14;
        const Scalar _tmp31 = _Qi[3] * _tmp17;
        const Scalar _tmp32 = _tmp30 + _tmp31;
        const Scalar _tmp33 = -_tmp22;
        const Scalar _tmp34 = _tmp21 + _tmp33;
        const Scalar _tmp35 = _tmp20 * _tmp32 + _tmp34 * _tmp7;
        const Scalar _tmp36 = -_tmp10 * dp_dbg(1, 2) - _tmp11 * dp_dba(1, 1) - _tmp12 * dp_dba(1, 0) -
                                _tmp13 * dp_dba(1, 2) + _tmp24 * _tmp29 + _tmp35 - _tmp8 * dp_dbg(1, 0) -
                                _tmp9 * dp_dbg(1, 1) - delta_p(1, 0);
        const Scalar _tmp37 = _tmp28 + _tmp3;
        const Scalar _tmp38 = _tmp15 + _tmp18;
        const Scalar _tmp39 = -_tmp31;
        const Scalar _tmp40 = _tmp30 + _tmp39;
        const Scalar _tmp41 = _tmp24 * _tmp40 + _tmp38 * _tmp7;
        const Scalar _tmp42 = -_tmp10 * dp_dbg(2, 2) - _tmp11 * dp_dba(2, 1) - _tmp12 * dp_dba(2, 0) -
                                _tmp13 * dp_dba(2, 2) + _tmp20 * _tmp37 + _tmp41 - _tmp8 * dp_dbg(2, 0) -
                                _tmp9 * dp_dbg(2, 1) - delta_p(2, 0);
        const Scalar _tmp43 = _Qi[1] * _Qj[3];
        const Scalar _tmp44 = _Qi[0] * _Qj[2];
        const Scalar _tmp45 = _Qi[2] * _Qj[0];
        const Scalar _tmp46 = _Qi[3] * _Qj[1];
        const Scalar _tmp47 = -_tmp43 + _tmp44 - _tmp45 + _tmp46;
        const Scalar _tmp48 = Scalar(0.5) * _tmp10 * dq_dbg(0, 2) + Scalar(0.5) * _tmp8 * dq_dbg(0, 0) +
                                Scalar(0.5) * _tmp9 * dq_dbg(0, 1);
        const Scalar _tmp49 = Scalar(0.5) * _tmp10 * dq_dbg(1, 2) + Scalar(0.5) * _tmp8 * dq_dbg(1, 0) +
                                Scalar(0.5) * _tmp9 * dq_dbg(1, 1);
        const Scalar _tmp50 = Scalar(0.5) * _tmp10 * dq_dbg(2, 2) + Scalar(0.5) * _tmp8 * dq_dbg(2, 0) +
                                Scalar(0.5) * _tmp9 * dq_dbg(2, 1);
        const Scalar _tmp51 = _delta_q[0] * _tmp49 - _delta_q[1] * _tmp48 + Scalar(1.0) * _delta_q[2] +
                                _delta_q[3] * _tmp50;
        const Scalar _tmp52 = _Qi[1] * _Qj[0];
        const Scalar _tmp53 = _Qi[0] * _Qj[1];
        const Scalar _tmp54 = _Qi[2] * _Qj[3];
        const Scalar _tmp55 = _Qi[3] * _Qj[2];
        const Scalar _tmp56 = _tmp52 - _tmp53 - _tmp54 + _tmp55;
        const Scalar _tmp57 = -_delta_q[0] * _tmp50 + Scalar(1.0) * _delta_q[1] + _delta_q[2] * _tmp48 +
                                _delta_q[3] * _tmp49;
        const Scalar _tmp58 = _Qi[1] * _Qj[2];
        const Scalar _tmp59 = _Qi[0] * _Qj[3];
        const Scalar _tmp60 = _Qi[2] * _Qj[1];
        const Scalar _tmp61 = _Qi[3] * _Qj[0];
        const Scalar _tmp62 = -_tmp58 - _tmp59 + _tmp60 + _tmp61;
        const Scalar _tmp63 = -_delta_q[0] * _tmp48 - _delta_q[1] * _tmp49 - _delta_q[2] * _tmp50 +
                                Scalar(1.0) * _delta_q[3];
        const Scalar _tmp64 = Scalar(1.0) * _delta_q[0] + _delta_q[1] * _tmp50 - _delta_q[2] * _tmp49 +
                                _delta_q[3] * _tmp48;
        const Scalar _tmp65 = _Qi[1] * _Qj[1];
        const Scalar _tmp66 = _Qi[0] * _Qj[0];
        const Scalar _tmp67 = _Qi[2] * _Qj[2];
        const Scalar _tmp68 = _Qi[3] * _Qj[3];
        const Scalar _tmp69 = _tmp65 + _tmp66 + _tmp67 + _tmp68;
        const Scalar _tmp70 =
            2 * _tmp47 * _tmp51 - 2 * _tmp56 * _tmp57 + 2 * _tmp62 * _tmp63 - 2 * _tmp64 * _tmp69;
        const Scalar _tmp71 =
            2 * _tmp47 * _tmp63 - 2 * _tmp51 * _tmp62 + 2 * _tmp56 * _tmp64 - 2 * _tmp57 * _tmp69;
        const Scalar _tmp72 =
            -2 * _tmp47 * _tmp64 - 2 * _tmp51 * _tmp69 + 2 * _tmp56 * _tmp63 + 2 * _tmp57 * _tmp62;
        const Scalar _tmp73 = G(0, 0) * sum_dt - Vi(0, 0) + Vj(0, 0);
        const Scalar _tmp74 = G(1, 0) * sum_dt - Vi(1, 0) + Vj(1, 0);
        const Scalar _tmp75 = G(2, 0) * sum_dt - Vi(2, 0) + Vj(2, 0);
        const Scalar _tmp76 = _tmp19 * _tmp75 + _tmp23 * _tmp74;
        const Scalar _tmp77 = -_tmp10 * dv_dbg(0, 2) - _tmp11 * dv_dba(0, 1) - _tmp12 * dv_dba(0, 0) -
                                _tmp13 * dv_dba(0, 2) + _tmp4 * _tmp73 + _tmp76 - _tmp8 * dv_dbg(0, 0) -
                                _tmp9 * dv_dbg(0, 1) - delta_v(0, 0);
        const Scalar _tmp78 = _tmp32 * _tmp75 + _tmp34 * _tmp73;
        const Scalar _tmp79 = -_tmp10 * dv_dbg(1, 2) - _tmp11 * dv_dba(1, 1) - _tmp12 * dv_dba(1, 0) -
                                _tmp13 * dv_dba(1, 2) + _tmp29 * _tmp74 + _tmp78 - _tmp8 * dv_dbg(1, 0) -
                                _tmp9 * dv_dbg(1, 1) - delta_v(1, 0);
        const Scalar _tmp80 = _tmp38 * _tmp73 + _tmp40 * _tmp74;
        const Scalar _tmp81 = -_tmp10 * dv_dbg(2, 2) - _tmp11 * dv_dba(2, 1) - _tmp12 * dv_dba(2, 0) -
                                _tmp13 * dv_dba(2, 2) + _tmp37 * _tmp75 - _tmp8 * dv_dbg(2, 0) + _tmp80 -
                                _tmp9 * dv_dbg(2, 1) - delta_v(2, 0);
        const Scalar _tmp82 = -Bai(0, 0) + Baj(0, 0);
        const Scalar _tmp83 = -Bai(1, 0) + Baj(1, 0);
        const Scalar _tmp84 = -Bai(2, 0) + Baj(2, 0);
        const Scalar _tmp85 = -Bgi(0, 0) + Bgj(0, 0);
        const Scalar _tmp86 = -Bgi(1, 0) + Bgj(1, 0);
        const Scalar _tmp87 = -Bgi(2, 0) + Bgj(2, 0);
        const Scalar _tmp88 = -_tmp4;
        const Scalar _tmp89 = -_tmp34;
        const Scalar _tmp90 = -_tmp38;
        const Scalar _tmp91 = -_tmp23;
        const Scalar _tmp92 = -_tmp29;
        const Scalar _tmp93 = -_tmp40;
        const Scalar _tmp94 = -_tmp19;
        const Scalar _tmp95 = -_tmp32;
        const Scalar _tmp96 = -_tmp37;
        const Scalar _tmp97 = -_tmp27;
        const Scalar _tmp98 = std::pow(_Qi[3], Scalar(2));
        const Scalar _tmp99 = -_tmp2;
        const Scalar _tmp100 = _tmp0 + _tmp97 + _tmp98 + _tmp99;
        const Scalar _tmp101 = _tmp100 * _tmp20 + _tmp41;
        const Scalar _tmp102 = -_tmp98;
        const Scalar _tmp103 = _tmp0 + _tmp102;
        const Scalar _tmp104 = _tmp27 + _tmp99;
        const Scalar _tmp105 = _tmp103 + _tmp104;
        const Scalar _tmp106 = -_tmp30;
        const Scalar _tmp107 = _tmp106 + _tmp39;
        const Scalar _tmp108 = -_tmp21;
        const Scalar _tmp109 = _tmp108 + _tmp22;
        const Scalar _tmp110 = _tmp105 * _tmp24 + _tmp107 * _tmp20 + _tmp109 * _tmp7;
        const Scalar _tmp111 = (Scalar(1) / Scalar(2)) * _tmp58;
        const Scalar _tmp112 = (Scalar(1) / Scalar(2)) * _tmp59;
        const Scalar _tmp113 = (Scalar(1) / Scalar(2)) * _tmp60;
        const Scalar _tmp114 = (Scalar(1) / Scalar(2)) * _tmp61;
        const Scalar _tmp115 = -_tmp111 - _tmp112 + _tmp113 + _tmp114;
        const Scalar _tmp116 = 2 * _tmp64;
        const Scalar _tmp117 = _tmp115 * _tmp116;
        const Scalar _tmp118 = (Scalar(1) / Scalar(2)) * _tmp65;
        const Scalar _tmp119 = (Scalar(1) / Scalar(2)) * _tmp66;
        const Scalar _tmp120 = (Scalar(1) / Scalar(2)) * _tmp67;
        const Scalar _tmp121 = (Scalar(1) / Scalar(2)) * _tmp68;
        const Scalar _tmp122 = -_tmp118 - _tmp119 - _tmp120 - _tmp121;
        const Scalar _tmp123 = 2 * _tmp63;
        const Scalar _tmp124 = _tmp122 * _tmp123;
        const Scalar _tmp125 = (Scalar(1) / Scalar(2)) * _tmp43;
        const Scalar _tmp126 = (Scalar(1) / Scalar(2)) * _tmp44;
        const Scalar _tmp127 = (Scalar(1) / Scalar(2)) * _tmp45;
        const Scalar _tmp128 = (Scalar(1) / Scalar(2)) * _tmp46;
        const Scalar _tmp129 = _tmp125 - _tmp126 + _tmp127 - _tmp128;
        const Scalar _tmp130 = 2 * _tmp57;
        const Scalar _tmp131 = -_tmp129 * _tmp130;
        const Scalar _tmp132 = (Scalar(1) / Scalar(2)) * _tmp52;
        const Scalar _tmp133 = (Scalar(1) / Scalar(2)) * _tmp53;
        const Scalar _tmp134 = (Scalar(1) / Scalar(2)) * _tmp54;
        const Scalar _tmp135 = (Scalar(1) / Scalar(2)) * _tmp55;
        const Scalar _tmp136 = _tmp132 - _tmp133 - _tmp134 + _tmp135;
        const Scalar _tmp137 = 2 * _tmp51;
        const Scalar _tmp138 = _tmp136 * _tmp137;
        const Scalar _tmp139 = _tmp131 + _tmp138;
        const Scalar _tmp140 = -_tmp117 + _tmp124 + _tmp139;
        const Scalar _tmp141 = _tmp122 * _tmp137;
        const Scalar _tmp142 = -_tmp115 * _tmp130;
        const Scalar _tmp143 = _tmp116 * _tmp129;
        const Scalar _tmp144 = _tmp123 * _tmp136 + _tmp143;
        const Scalar _tmp145 = -_tmp141 + _tmp142 + _tmp144;
        const Scalar _tmp146 = _tmp122 * _tmp130;
        const Scalar _tmp147 = -_tmp116 * _tmp136;
        const Scalar _tmp148 = _tmp123 * _tmp129 + _tmp147;
        const Scalar _tmp149 = -_tmp115 * _tmp137 + _tmp146 + _tmp148;
        const Scalar _tmp150 = _tmp100 * _tmp75 + _tmp80;
        const Scalar _tmp151 = _tmp105 * _tmp74 + _tmp107 * _tmp75 + _tmp109 * _tmp73;
        const Scalar _tmp152 = -_tmp18;
        const Scalar _tmp153 = _tmp152 + _tmp16;
        const Scalar _tmp154 = _tmp106 + _tmp31;
        const Scalar _tmp155 = -_tmp0;
        const Scalar _tmp156 = _tmp102 + _tmp155 + _tmp2 + _tmp27;
        const Scalar _tmp157 = _tmp153 * _tmp7 + _tmp154 * _tmp24 + _tmp156 * _tmp20;
        const Scalar _tmp158 = _tmp155 + _tmp98;
        const Scalar _tmp159 = _tmp104 + _tmp158;
        const Scalar _tmp160 = _tmp159 * _tmp7 + _tmp25;
        const Scalar _tmp161 = -_tmp125 + _tmp126 - _tmp127 + _tmp128;
        const Scalar _tmp162 = 2 * _tmp161;
        const Scalar _tmp163 = -_tmp132 + _tmp133 + _tmp134 - _tmp135;
        const Scalar _tmp164 = 2 * _tmp163;
        const Scalar _tmp165 = _tmp142 + _tmp164 * _tmp63;
        const Scalar _tmp166 = _tmp141 - _tmp162 * _tmp64 + _tmp165;
        const Scalar _tmp167 = _tmp130 * _tmp161;
        const Scalar _tmp168 = -_tmp164 * _tmp51;
        const Scalar _tmp169 = _tmp117 + _tmp168;
        const Scalar _tmp170 = _tmp124 - _tmp167 + _tmp169;
        const Scalar _tmp171 = -_tmp162 * _tmp51;
        const Scalar _tmp172 = _tmp116 * _tmp122;
        const Scalar _tmp173 = _tmp130 * _tmp163;
        const Scalar _tmp174 = _tmp115 * _tmp123 + _tmp173;
        const Scalar _tmp175 = _tmp171 - _tmp172 + _tmp174;
        const Scalar _tmp176 = _tmp153 * _tmp73 + _tmp154 * _tmp74 + _tmp156 * _tmp75;
        const Scalar _tmp177 = _tmp159 * _tmp73 + _tmp76;
        const Scalar _tmp178 = _tmp2 + _tmp97;
        const Scalar _tmp179 = _tmp158 + _tmp178;
        const Scalar _tmp180 = _tmp179 * _tmp24 + _tmp35;
        const Scalar _tmp181 = _tmp103 + _tmp178;
        const Scalar _tmp182 = _tmp15 + _tmp152;
        const Scalar _tmp183 = _tmp108 + _tmp33;
        const Scalar _tmp184 = _tmp181 * _tmp7 + _tmp182 * _tmp20 + _tmp183 * _tmp24;
        const Scalar _tmp185 = _tmp111 + _tmp112 - _tmp113 - _tmp114;
        const Scalar _tmp186 = _tmp137 * _tmp185;
        const Scalar _tmp187 = _tmp162 * _tmp63 + _tmp186;
        const Scalar _tmp188 = -_tmp146 + _tmp147 + _tmp187;
        const Scalar _tmp189 = _tmp123 * _tmp185 + _tmp171;
        const Scalar _tmp190 = -_tmp130 * _tmp136 + _tmp172 + _tmp189;
        const Scalar _tmp191 = -_tmp116 * _tmp185;
        const Scalar _tmp192 = _tmp124 - _tmp138 + _tmp167 + _tmp191;
        const Scalar _tmp193 = _tmp179 * _tmp74 + _tmp78;
        const Scalar _tmp194 = _tmp181 * _tmp73 + _tmp182 * _tmp75 + _tmp183 * _tmp74;
        const Scalar _tmp195 = _tmp34 * sum_dt;
        const Scalar _tmp196 = _tmp23 * sum_dt;
        const Scalar _tmp197 = Scalar(0.5) * _delta_q[3];
        const Scalar _tmp198 = Scalar(0.5) * _delta_q[1];
        const Scalar _tmp199 = Scalar(0.5) * _delta_q[2];
        const Scalar _tmp200 = _tmp197 * dq_dbg(0, 0) + _tmp198 * dq_dbg(2, 0) - _tmp199 * dq_dbg(1, 0);
        const Scalar _tmp201 = 2 * _tmp69;
        const Scalar _tmp202 = Scalar(0.5) * _delta_q[0];
        const Scalar _tmp203 = _tmp197 * dq_dbg(2, 0) - _tmp198 * dq_dbg(0, 0) + _tmp202 * dq_dbg(1, 0);
        const Scalar _tmp204 = 2 * _tmp47;
        const Scalar _tmp205 = _tmp197 * dq_dbg(1, 0) + _tmp199 * dq_dbg(0, 0) - _tmp202 * dq_dbg(2, 0);
        const Scalar _tmp206 = 2 * _tmp56;
        const Scalar _tmp207 = -_tmp198 * dq_dbg(1, 0) - _tmp199 * dq_dbg(2, 0) - _tmp202 * dq_dbg(0, 0);
        const Scalar _tmp208 = 2 * _tmp62;
        const Scalar _tmp209 =
            -_tmp200 * _tmp201 + _tmp203 * _tmp204 - _tmp205 * _tmp206 + _tmp207 * _tmp208;
        const Scalar _tmp210 =
            _tmp200 * _tmp206 - _tmp201 * _tmp205 - _tmp203 * _tmp208 + _tmp204 * _tmp207;
        const Scalar _tmp211 =
            -_tmp200 * _tmp204 - _tmp201 * _tmp203 + _tmp205 * _tmp208 + _tmp206 * _tmp207;
        const Scalar _tmp212 = _tmp197 * dq_dbg(0, 1) + _tmp198 * dq_dbg(2, 1) - _tmp199 * dq_dbg(1, 1);
        const Scalar _tmp213 = _tmp197 * dq_dbg(2, 1) - _tmp198 * dq_dbg(0, 1) + _tmp202 * dq_dbg(1, 1);
        const Scalar _tmp214 = _tmp197 * dq_dbg(1, 1) + _tmp199 * dq_dbg(0, 1) - _tmp202 * dq_dbg(2, 1);
        const Scalar _tmp215 = -_tmp198 * dq_dbg(1, 1) - _tmp199 * dq_dbg(2, 1) - _tmp202 * dq_dbg(0, 1);
        const Scalar _tmp216 =
            -_tmp201 * _tmp212 + _tmp204 * _tmp213 - _tmp206 * _tmp214 + _tmp208 * _tmp215;
        const Scalar _tmp217 =
            -_tmp201 * _tmp214 + _tmp204 * _tmp215 + _tmp206 * _tmp212 - _tmp208 * _tmp213;
        const Scalar _tmp218 =
            -_tmp201 * _tmp213 - _tmp204 * _tmp212 + _tmp206 * _tmp215 + _tmp208 * _tmp214;
        const Scalar _tmp219 = _tmp197 * dq_dbg(0, 2) + _tmp198 * dq_dbg(2, 2) - _tmp199 * dq_dbg(1, 2);
        const Scalar _tmp220 = _tmp197 * dq_dbg(2, 2) - _tmp198 * dq_dbg(0, 2) + _tmp202 * dq_dbg(1, 2);
        const Scalar _tmp221 = _tmp197 * dq_dbg(1, 2) + _tmp199 * dq_dbg(0, 2) - _tmp202 * dq_dbg(2, 2);
        const Scalar _tmp222 = -_tmp198 * dq_dbg(1, 2) - _tmp199 * dq_dbg(2, 2) - _tmp202 * dq_dbg(0, 2);
        const Scalar _tmp223 =
            -_tmp201 * _tmp219 + _tmp204 * _tmp220 - _tmp206 * _tmp221 + _tmp208 * _tmp222;
        const Scalar _tmp224 =
            -_tmp201 * _tmp221 + _tmp204 * _tmp222 + _tmp206 * _tmp219 - _tmp208 * _tmp220;
        const Scalar _tmp225 =
            -_tmp201 * _tmp220 - _tmp204 * _tmp219 + _tmp206 * _tmp222 + _tmp208 * _tmp221;
        const Scalar _tmp226 = _tmp118 + _tmp119 + _tmp120 + _tmp121;
        const Scalar _tmp227 = _tmp123 * _tmp226;
        const Scalar _tmp228 = _tmp191 + _tmp227;
        const Scalar _tmp229 = _tmp139 + _tmp228;
        const Scalar _tmp230 = _tmp137 * _tmp226;
        const Scalar _tmp231 = -_tmp130 * _tmp185 + _tmp144 - _tmp230;
        const Scalar _tmp232 = _tmp130 * _tmp226;
        const Scalar _tmp233 = _tmp148 - _tmp186 + _tmp232;
        const Scalar _tmp234 = -_tmp143 + _tmp165 + _tmp230;
        const Scalar _tmp235 = _tmp131 + _tmp169 + _tmp227;
        const Scalar _tmp236 = _tmp116 * _tmp226;
        const Scalar _tmp237 = -_tmp129 * _tmp137 + _tmp174 - _tmp236;
        const Scalar _tmp238 = -_tmp164 * _tmp64 + _tmp187 - _tmp232;
        const Scalar _tmp239 = -_tmp173 + _tmp189 + _tmp236;
        const Scalar _tmp240 = _tmp167 + _tmp168 + _tmp228;
        /*const Scalar _tmp241 = std::pow(_tmp38, Scalar(2));
        const Scalar _tmp242 = std::pow(_tmp4, Scalar(2));
        const Scalar _tmp243 = std::pow(_tmp34, Scalar(2));
        const Scalar _tmp244 = _tmp241 + _tmp242 + _tmp243;
        const Scalar _tmp245 = _tmp38 * _tmp40;
        const Scalar _tmp246 = _tmp29 * _tmp34;
        const Scalar _tmp247 = _tmp23 * _tmp4;
        const Scalar _tmp248 = _tmp245 + _tmp246 + _tmp247;
        const Scalar _tmp249 = _tmp19 * _tmp4;
        const Scalar _tmp250 = _tmp37 * _tmp38;
        const Scalar _tmp251 = _tmp32 * _tmp34;
        const Scalar _tmp252 = _tmp249 + _tmp250 + _tmp251;
        const Scalar _tmp253 = _tmp101 * _tmp34;
        const Scalar _tmp254 = _tmp110 * _tmp38;
        const Scalar _tmp255 = _tmp160 * _tmp38;
        const Scalar _tmp256 = _tmp157 * _tmp4;
        const Scalar _tmp257 = _tmp184 * _tmp34;
        const Scalar _tmp258 = _tmp180 * _tmp4;
        const Scalar _tmp259 = _tmp242 * sum_dt;
        const Scalar _tmp260 = _tmp241 * sum_dt;
        const Scalar _tmp261 = _tmp243 * sum_dt;
        const Scalar _tmp262 = _tmp196 * _tmp4;
        const Scalar _tmp263 = _tmp195 * _tmp29;
        const Scalar _tmp264 = _tmp245 * sum_dt;
        const Scalar _tmp265 = _tmp262 + _tmp263 + _tmp264;
        const Scalar _tmp266 = _tmp195 * _tmp32;
        const Scalar _tmp267 = _tmp250 * sum_dt;
        const Scalar _tmp268 = _tmp249 * sum_dt;
        const Scalar _tmp269 = _tmp266 + _tmp267 + _tmp268;
        const Scalar _tmp270 = _tmp34 * dp_dba(1, 0);
        const Scalar _tmp271 = _tmp4 * dp_dba(0, 0);
        const Scalar _tmp272 = _tmp38 * dp_dba(2, 0);
        const Scalar _tmp273 = _tmp34 * dp_dba(1, 1);
        const Scalar _tmp274 = _tmp4 * dp_dba(0, 1);
        const Scalar _tmp275 = _tmp38 * dp_dba(2, 1);
        const Scalar _tmp276 = _tmp34 * dp_dba(1, 2);
        const Scalar _tmp277 = _tmp4 * dp_dba(0, 2);
        const Scalar _tmp278 = _tmp38 * dp_dba(2, 2);
        const Scalar _tmp279 = _tmp34 * dp_dbg(1, 0);
        const Scalar _tmp280 = _tmp4 * dp_dbg(0, 0);
        const Scalar _tmp281 = _tmp38 * dp_dbg(2, 0);
        const Scalar _tmp282 = _tmp34 * dp_dbg(1, 1);
        const Scalar _tmp283 = _tmp4 * dp_dbg(0, 1);
        const Scalar _tmp284 = _tmp38 * dp_dbg(2, 1);
        const Scalar _tmp285 = _tmp34 * dp_dbg(1, 2);
        const Scalar _tmp286 = _tmp4 * dp_dbg(0, 2);
        const Scalar _tmp287 = _tmp38 * dp_dbg(2, 2);
        const Scalar _tmp288 = -_tmp241 - _tmp242 - _tmp243;
        const Scalar _tmp289 = -_tmp245 - _tmp246 - _tmp247;
        const Scalar _tmp290 = -_tmp249 - _tmp250 - _tmp251;
        const Scalar _tmp291 = std::pow(_tmp40, Scalar(2));
        const Scalar _tmp292 = std::pow(_tmp29, Scalar(2));
        const Scalar _tmp293 = std::pow(_tmp23, Scalar(2));
        const Scalar _tmp294 = _tmp291 + _tmp292 + _tmp293;
        const Scalar _tmp295 = _tmp29 * _tmp32;
        const Scalar _tmp296 = _tmp37 * _tmp40;
        const Scalar _tmp297 = _tmp19 * _tmp23;
        const Scalar _tmp298 = _tmp295 + _tmp296 + _tmp297;
        const Scalar _tmp299 = _tmp110 * _tmp40;
        const Scalar _tmp300 = _tmp101 * _tmp29;
        const Scalar _tmp301 = _tmp157 * _tmp23;
        const Scalar _tmp302 = _tmp160 * _tmp40;
        const Scalar _tmp303 = _tmp180 * _tmp23;
        const Scalar _tmp304 = _tmp184 * _tmp29;
        const Scalar _tmp305 = _tmp292 * sum_dt;
        const Scalar _tmp306 = _tmp291 * sum_dt;
        const Scalar _tmp307 = _tmp293 * sum_dt;
        const Scalar _tmp308 = _tmp19 * _tmp196;
        const Scalar _tmp309 = _tmp296 * sum_dt;
        const Scalar _tmp310 = _tmp295 * sum_dt;
        const Scalar _tmp311 = _tmp308 + _tmp309 + _tmp310;
        const Scalar _tmp312 = _tmp23 * dp_dba(0, 0);
        const Scalar _tmp313 = _tmp29 * dp_dba(1, 0);
        const Scalar _tmp314 = _tmp40 * dp_dba(2, 0);
        const Scalar _tmp315 = _tmp23 * dp_dba(0, 1);
        const Scalar _tmp316 = _tmp29 * dp_dba(1, 1);
        const Scalar _tmp317 = _tmp40 * dp_dba(2, 1);
        const Scalar _tmp318 = _tmp23 * dp_dba(0, 2);
        const Scalar _tmp319 = _tmp29 * dp_dba(1, 2);
        const Scalar _tmp320 = _tmp40 * dp_dba(2, 2);
        const Scalar _tmp321 = _tmp23 * dp_dbg(0, 0);
        const Scalar _tmp322 = _tmp29 * dp_dbg(1, 0);
        const Scalar _tmp323 = _tmp40 * dp_dbg(2, 0);
        const Scalar _tmp324 = _tmp23 * dp_dbg(0, 1);
        const Scalar _tmp325 = _tmp29 * dp_dbg(1, 1);
        const Scalar _tmp326 = _tmp40 * dp_dbg(2, 1);
        const Scalar _tmp327 = _tmp23 * dp_dbg(0, 2);
        const Scalar _tmp328 = _tmp29 * dp_dbg(1, 2);
        const Scalar _tmp329 = _tmp40 * dp_dbg(2, 2);
        const Scalar _tmp330 = -_tmp291 - _tmp292 - _tmp293;
        const Scalar _tmp331 = -_tmp295 - _tmp296 - _tmp297;
        const Scalar _tmp332 = std::pow(_tmp19, Scalar(2));
        const Scalar _tmp333 = std::pow(_tmp32, Scalar(2));
        const Scalar _tmp334 = std::pow(_tmp37, Scalar(2));
        const Scalar _tmp335 = _tmp332 + _tmp333 + _tmp334;
        const Scalar _tmp336 = _tmp110 * _tmp37;
        const Scalar _tmp337 = _tmp101 * _tmp32;
        const Scalar _tmp338 = _tmp160 * _tmp37;
        const Scalar _tmp339 = _tmp157 * _tmp19;
        const Scalar _tmp340 = _tmp184 * _tmp32;
        const Scalar _tmp341 = _tmp180 * _tmp19;
        const Scalar _tmp342 = _tmp334 * sum_dt;
        const Scalar _tmp343 = _tmp333 * sum_dt;
        const Scalar _tmp344 = _tmp332 * sum_dt;
        const Scalar _tmp345 = _tmp37 * dp_dba(2, 0);
        const Scalar _tmp346 = _tmp32 * dp_dba(1, 0);
        const Scalar _tmp347 = _tmp19 * dp_dba(0, 0);
        const Scalar _tmp348 = _tmp37 * dp_dba(2, 1);
        const Scalar _tmp349 = _tmp32 * dp_dba(1, 1);
        const Scalar _tmp350 = _tmp19 * dp_dba(0, 1);
        const Scalar _tmp351 = _tmp37 * dp_dba(2, 2);
        const Scalar _tmp352 = _tmp32 * dp_dba(1, 2);
        const Scalar _tmp353 = _tmp19 * dp_dba(0, 2);
        const Scalar _tmp354 = _tmp37 * dp_dbg(2, 0);
        const Scalar _tmp355 = _tmp32 * dp_dbg(1, 0);
        const Scalar _tmp356 = _tmp19 * dp_dbg(0, 0);
        const Scalar _tmp357 = _tmp37 * dp_dbg(2, 1);
        const Scalar _tmp358 = _tmp32 * dp_dbg(1, 1);
        const Scalar _tmp359 = _tmp19 * dp_dbg(0, 1);
        const Scalar _tmp360 = _tmp37 * dp_dbg(2, 2);
        const Scalar _tmp361 = _tmp32 * dp_dbg(1, 2);
        const Scalar _tmp362 = _tmp19 * dp_dbg(0, 2);
        const Scalar _tmp363 = -_tmp332 - _tmp333 - _tmp334;
        const Scalar _tmp364 = _tmp150 * _tmp34;
        const Scalar _tmp365 = _tmp151 * _tmp38;
        const Scalar _tmp366 = _tmp150 * _tmp29;
        const Scalar _tmp367 = _tmp151 * _tmp40;
        const Scalar _tmp368 = _tmp150 * _tmp32;
        const Scalar _tmp369 = _tmp151 * _tmp37;
        const Scalar _tmp370 = _tmp176 * _tmp4;
        const Scalar _tmp371 = _tmp177 * _tmp38;
        const Scalar _tmp372 = _tmp176 * _tmp23;
        const Scalar _tmp373 = _tmp177 * _tmp40;
        const Scalar _tmp374 = _tmp176 * _tmp19;
        const Scalar _tmp375 = _tmp177 * _tmp37;
        const Scalar _tmp376 = _tmp193 * _tmp4;
        const Scalar _tmp377 = _tmp194 * _tmp34;
        const Scalar _tmp378 = _tmp194 * _tmp29;
        const Scalar _tmp379 = _tmp193 * _tmp23;
        const Scalar _tmp380 = _tmp19 * _tmp193;
        const Scalar _tmp381 = _tmp194 * _tmp32;
        const Scalar _tmp382 = _tmp34 * dv_dba(1, 0);
        const Scalar _tmp383 = _tmp4 * dv_dba(0, 0);
        const Scalar _tmp384 = _tmp38 * dv_dba(2, 0);
        const Scalar _tmp385 = _tmp34 * dv_dba(1, 1);
        const Scalar _tmp386 = _tmp4 * dv_dba(0, 1);
        const Scalar _tmp387 = _tmp38 * dv_dba(2, 1);
        const Scalar _tmp388 = _tmp34 * dv_dba(1, 2);
        const Scalar _tmp389 = _tmp4 * dv_dba(0, 2);
        const Scalar _tmp390 = _tmp38 * dv_dba(2, 2);
        const Scalar _tmp391 = _tmp34 * dv_dbg(1, 0);
        const Scalar _tmp392 = _tmp4 * dv_dbg(0, 0);
        const Scalar _tmp393 = _tmp38 * dv_dbg(2, 0);
        const Scalar _tmp394 = _tmp34 * dv_dbg(1, 1);
        const Scalar _tmp395 = _tmp4 * dv_dbg(0, 1);
        const Scalar _tmp396 = _tmp38 * dv_dbg(2, 1);
        const Scalar _tmp397 = _tmp34 * dv_dbg(1, 2);
        const Scalar _tmp398 = _tmp4 * dv_dbg(0, 2);
        const Scalar _tmp399 = _tmp38 * dv_dbg(2, 2);
        const Scalar _tmp400 = -_tmp262 - _tmp263 - _tmp264;
        const Scalar _tmp401 = -_tmp266 - _tmp267 - _tmp268;
        const Scalar _tmp402 = _tmp23 * dv_dba(0, 0);
        const Scalar _tmp403 = _tmp29 * dv_dba(1, 0);
        const Scalar _tmp404 = _tmp40 * dv_dba(2, 0);
        const Scalar _tmp405 = _tmp23 * dv_dba(0, 1);
        const Scalar _tmp406 = _tmp29 * dv_dba(1, 1);
        const Scalar _tmp407 = _tmp40 * dv_dba(2, 1);
        const Scalar _tmp408 = _tmp23 * dv_dba(0, 2);
        const Scalar _tmp409 = _tmp29 * dv_dba(1, 2);
        const Scalar _tmp410 = _tmp40 * dv_dba(2, 2);
        const Scalar _tmp411 = _tmp23 * dv_dbg(0, 0);
        const Scalar _tmp412 = _tmp29 * dv_dbg(1, 0);
        const Scalar _tmp413 = _tmp40 * dv_dbg(2, 0);
        const Scalar _tmp414 = _tmp23 * dv_dbg(0, 1);
        const Scalar _tmp415 = _tmp29 * dv_dbg(1, 1);
        const Scalar _tmp416 = _tmp40 * dv_dbg(2, 1);
        const Scalar _tmp417 = _tmp23 * dv_dbg(0, 2);
        const Scalar _tmp418 = _tmp29 * dv_dbg(1, 2);
        const Scalar _tmp419 = _tmp40 * dv_dbg(2, 2);
        const Scalar _tmp420 = -_tmp308 - _tmp309 - _tmp310;
        const Scalar _tmp421 = _tmp37 * dv_dba(2, 0);
        const Scalar _tmp422 = _tmp32 * dv_dba(1, 0);
        const Scalar _tmp423 = _tmp19 * dv_dba(0, 0);
        const Scalar _tmp424 = _tmp37 * dv_dba(2, 1);
        const Scalar _tmp425 = _tmp32 * dv_dba(1, 1);
        const Scalar _tmp426 = _tmp19 * dv_dba(0, 1);
        const Scalar _tmp427 = _tmp37 * dv_dba(2, 2);
        const Scalar _tmp428 = _tmp32 * dv_dba(1, 2);
        const Scalar _tmp429 = _tmp19 * dv_dba(0, 2);
        const Scalar _tmp430 = _tmp37 * dv_dbg(2, 0);
        const Scalar _tmp431 = _tmp32 * dv_dbg(1, 0);
        const Scalar _tmp432 = _tmp19 * dv_dbg(0, 0);
        const Scalar _tmp433 = _tmp37 * dv_dbg(2, 1);
        const Scalar _tmp434 = _tmp32 * dv_dbg(1, 1);
        const Scalar _tmp435 = _tmp19 * dv_dbg(0, 1);
        const Scalar _tmp436 = _tmp37 * dv_dbg(2, 2);
        const Scalar _tmp437 = _tmp32 * dv_dbg(1, 2);
        const Scalar _tmp438 = _tmp19 * dv_dbg(0, 2);
        const Scalar _tmp439 = _tmp38 * _tmp42;
        const Scalar _tmp440 = _tmp34 * _tmp36;
        const Scalar _tmp441 = _tmp26 * _tmp4;
        const Scalar _tmp442 = _tmp40 * _tmp42;
        const Scalar _tmp443 = _tmp23 * _tmp26;
        const Scalar _tmp444 = _tmp29 * _tmp36;
        const Scalar _tmp445 = _tmp37 * _tmp42;
        const Scalar _tmp446 = _tmp19 * _tmp26;
        const Scalar _tmp447 = _tmp32 * _tmp36;
        const Scalar _tmp448 = _tmp34 * _tmp79;
        const Scalar _tmp449 = _tmp4 * _tmp77;
        const Scalar _tmp450 = _tmp38 * _tmp81;
        const Scalar _tmp451 = _tmp23 * _tmp77;
        const Scalar _tmp452 = _tmp29 * _tmp79;
        const Scalar _tmp453 = _tmp40 * _tmp81;
        const Scalar _tmp454 = _tmp32 * _tmp79;
        const Scalar _tmp455 = _tmp19 * _tmp77;
        const Scalar _tmp456 = _tmp37 * _tmp81;
        */

        // Output terms (4)
        // if (res != nullptr) {
            // Eigen::Matrix<Scalar, 15, 1>& _res = (*res);
            Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);

            residual(0, 0) = _tmp26;
            residual(1, 0) = _tmp36;
            residual(2, 0) = _tmp42;
            residual(3, 0) = _tmp70;
            residual(4, 0) = _tmp71;
            residual(5, 0) = _tmp72;
            residual(6, 0) = _tmp77;
            residual(7, 0) = _tmp79;
            residual(8, 0) = _tmp81;
            residual(9, 0) = _tmp82;
            residual(10, 0) = _tmp83;
            residual(11, 0) = _tmp84;
            residual(12, 0) = _tmp85;
            residual(13, 0) = _tmp86;
            residual(14, 0) = _tmp87;

            residual = sqrt_info * residual;
        // }

        // if (jacobian != nullptr)
        if (jacobians)
        {
            // Eigen::Matrix<Scalar, 15, 30>& _jacobian = (*jacobian);

            // _jacobian.setZero();

            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();

                jacobian_pose_i(0, 0) = _tmp88;
                jacobian_pose_i(1, 0) = _tmp89;
                jacobian_pose_i(2, 0) = _tmp90;
                jacobian_pose_i(0, 1) = _tmp91;
                jacobian_pose_i(1, 1) = _tmp92;
                jacobian_pose_i(2, 1) = _tmp93;
                jacobian_pose_i(0, 2) = _tmp94;
                jacobian_pose_i(1, 2) = _tmp95;
                jacobian_pose_i(2, 2) = _tmp96;
                jacobian_pose_i(1, 3) = _tmp101;
                jacobian_pose_i(2, 3) = _tmp110;
                jacobian_pose_i(3, 3) = _tmp140;
                jacobian_pose_i(4, 3) = _tmp145;
                jacobian_pose_i(5, 3) = _tmp149;
                jacobian_pose_i(7, 3) = _tmp150;
                jacobian_pose_i(8, 3) = _tmp151;
                jacobian_pose_i(0, 4) = _tmp157;
                jacobian_pose_i(2, 4) = _tmp160;
                jacobian_pose_i(3, 4) = _tmp166;
                jacobian_pose_i(4, 4) = _tmp170;
                jacobian_pose_i(5, 4) = _tmp175;
                jacobian_pose_i(6, 4) = _tmp176;
                jacobian_pose_i(8, 4) = _tmp177;
                jacobian_pose_i(0, 5) = _tmp180;
                jacobian_pose_i(1, 5) = _tmp184;
                jacobian_pose_i(3, 5) = _tmp188;
                jacobian_pose_i(4, 5) = _tmp190;
                jacobian_pose_i(5, 5) = _tmp192;
                jacobian_pose_i(6, 5) = _tmp193;
                jacobian_pose_i(7, 5) = _tmp194;

                jacobian_pose_i = sqrt_info * jacobian_pose_i;
            }

            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);
                jacobian_speedbias_i.setZero();

                jacobian_speedbias_i(0, 0) = -_tmp4 * sum_dt;
                jacobian_speedbias_i(1, 0) = -_tmp195;
                jacobian_speedbias_i(2, 0) = -_tmp38 * sum_dt;
                jacobian_speedbias_i(6, 0) = _tmp88;
                jacobian_speedbias_i(7, 0) = _tmp89;
                jacobian_speedbias_i(8, 0) = _tmp90;
                jacobian_speedbias_i(0, 1) = -_tmp196;
                jacobian_speedbias_i(1, 1) = -_tmp29 * sum_dt;
                jacobian_speedbias_i(2, 1) = -_tmp40 * sum_dt;
                jacobian_speedbias_i(6, 1) = _tmp91;
                jacobian_speedbias_i(7, 1) = _tmp92;
                jacobian_speedbias_i(8, 1) = _tmp93;
                jacobian_speedbias_i(0, 2) = -_tmp19 * sum_dt;
                jacobian_speedbias_i(1, 2) = -_tmp32 * sum_dt;
                jacobian_speedbias_i(2, 2) = -_tmp37 * sum_dt;
                jacobian_speedbias_i(6, 2) = _tmp94;
                jacobian_speedbias_i(7, 2) = _tmp95;
                jacobian_speedbias_i(8, 2) = _tmp96;
                jacobian_speedbias_i(0, 3) = -dp_dba(0, 0);
                jacobian_speedbias_i(1, 3) = -dp_dba(1, 0);
                jacobian_speedbias_i(2, 3) = -dp_dba(2, 0);
                jacobian_speedbias_i(6, 3) = -dv_dba(0, 0);
                jacobian_speedbias_i(7, 3) = -dv_dba(1, 0);
                jacobian_speedbias_i(8, 3) = -dv_dba(2, 0);
                jacobian_speedbias_i(9, 3) = -1;
                jacobian_speedbias_i(0, 4) = -dp_dba(0, 1);
                jacobian_speedbias_i(1, 4) = -dp_dba(1, 1);
                jacobian_speedbias_i(2, 4) = -dp_dba(2, 1);
                jacobian_speedbias_i(6, 4) = -dv_dba(0, 1);
                jacobian_speedbias_i(7, 4) = -dv_dba(1, 1);
                jacobian_speedbias_i(8, 4) = -dv_dba(2, 1);
                jacobian_speedbias_i(10, 4) = -1;
                jacobian_speedbias_i(0, 5) = -dp_dba(0, 2);
                jacobian_speedbias_i(1, 5) = -dp_dba(1, 2);
                jacobian_speedbias_i(2, 5) = -dp_dba(2, 2);
                jacobian_speedbias_i(6, 5) = -dv_dba(0, 2);
                jacobian_speedbias_i(7, 5) = -dv_dba(1, 2);
                jacobian_speedbias_i(8, 5) = -dv_dba(2, 2);
                jacobian_speedbias_i(11, 5) = -1;
                jacobian_speedbias_i(0, 6) = -dp_dbg(0, 0);
                jacobian_speedbias_i(1, 6) = -dp_dbg(1, 0);
                jacobian_speedbias_i(2, 6) = -dp_dbg(2, 0);
                jacobian_speedbias_i(3, 6) = _tmp209;
                jacobian_speedbias_i(4, 6) = _tmp210;
                jacobian_speedbias_i(5, 6) = _tmp211;
                jacobian_speedbias_i(6, 6) = -dv_dbg(0, 0);
                jacobian_speedbias_i(7, 6) = -dv_dbg(1, 0);
                jacobian_speedbias_i(8, 6) = -dv_dbg(2, 0);
                jacobian_speedbias_i(12, 6) = -1;
                jacobian_speedbias_i(0, 7) = -dp_dbg(0, 1);
                jacobian_speedbias_i(1, 7) = -dp_dbg(1, 1);
                jacobian_speedbias_i(2, 7) = -dp_dbg(2, 1);
                jacobian_speedbias_i(3, 7) = _tmp216;
                jacobian_speedbias_i(4, 7) = _tmp217;
                jacobian_speedbias_i(5, 7) = _tmp218;
                jacobian_speedbias_i(6, 7) = -dv_dbg(0, 1);
                jacobian_speedbias_i(7, 7) = -dv_dbg(1, 1);
                jacobian_speedbias_i(8, 7) = -dv_dbg(2, 1);
                jacobian_speedbias_i(13, 7) = -1;
                jacobian_speedbias_i(0, 8) = -dp_dbg(0, 2);
                jacobian_speedbias_i(1, 8) = -dp_dbg(1, 2);
                jacobian_speedbias_i(2, 8) = -dp_dbg(2, 2);
                jacobian_speedbias_i(3, 8) = _tmp223;
                jacobian_speedbias_i(4, 8) = _tmp224;
                jacobian_speedbias_i(5, 8) = _tmp225;
                jacobian_speedbias_i(6, 8) = -dv_dbg(0, 2);
                jacobian_speedbias_i(7, 8) = -dv_dbg(1, 2);
                jacobian_speedbias_i(8, 8) = -dv_dbg(2, 2);
                jacobian_speedbias_i(14, 8) = -1;
                jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;
            }

            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]);
                jacobian_pose_j.setZero();

                jacobian_pose_j(0, 0) = _tmp4;
                jacobian_pose_j(1, 0) = _tmp34;
                jacobian_pose_j(2, 0) = _tmp38;
                jacobian_pose_j(0, 1) = _tmp23;
                jacobian_pose_j(1, 1) = _tmp29;
                jacobian_pose_j(2, 1) = _tmp40;
                jacobian_pose_j(0, 2) = _tmp19;
                jacobian_pose_j(1, 2) = _tmp32;
                jacobian_pose_j(2, 2) = _tmp37;
                jacobian_pose_j(3, 3) = _tmp229;
                jacobian_pose_j(4, 3) = _tmp231;
                jacobian_pose_j(5, 3) = _tmp233;
                jacobian_pose_j(3, 4) = _tmp234;
                jacobian_pose_j(4, 4) = _tmp235;
                jacobian_pose_j(5, 4) = _tmp237;
                jacobian_pose_j(3, 5) = _tmp238;
                jacobian_pose_j(4, 5) = _tmp239;
                jacobian_pose_j(5, 5) = _tmp240;

                jacobian_pose_j = sqrt_info * jacobian_pose_j;
            }

            if (jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[3]);
                jacobian_speedbias_j.setZero();

                jacobian_speedbias_j(6, 0) = _tmp4;
                jacobian_speedbias_j(7, 0) = _tmp34;
                jacobian_speedbias_j(8, 0) = _tmp38;
                jacobian_speedbias_j(6, 1) = _tmp23;
                jacobian_speedbias_j(7, 1) = _tmp29;
                jacobian_speedbias_j(8, 1) = _tmp40;
                jacobian_speedbias_j(6, 2) = _tmp19;
                jacobian_speedbias_j(7, 2) = _tmp32;
                jacobian_speedbias_j(8, 2) = _tmp37;
                jacobian_speedbias_j(9, 3) = 1;
                jacobian_speedbias_j(10, 4) = 1;
                jacobian_speedbias_j(11, 5) = 1;
                jacobian_speedbias_j(12, 6) = 1;
                jacobian_speedbias_j(13, 7) = 1;
                jacobian_speedbias_j(14, 8) = 1;

                jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;

            }
        }

        // auto end = std::chrono::steady_clock::now();
        // std::chrono::duration<double> diff = end - start;
        // std::cout << std::fixed << "sym::imu_factor: " << diff.count() << " s\n";
        // std::cout << std::fixed << "sym::imu_factor: " << std::chrono::duration_cast<std::chrono::milliseconds>((end - start).count()) << " ms\n";
        // auto diff = end - start;
        // std::cout << "sym::imu_factor: " << std::chrono::duration <double, std::milli>(diff).count() << " ms" << std::endl;

        return true;
    }

    //bool Evaluate_Direct(double const *const *parameters, Eigen::Matrix<double, 15, 1> &residuals, Eigen::Matrix<double, 15, 30> &jacobians);

    //void checkCorrection();
    //void checkTransition();
    //void checkJacobian(double **parameters);
    IntegrationBase* pre_integration;

};

