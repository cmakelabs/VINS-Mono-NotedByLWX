#include "projection_factor.h"

Eigen::Matrix2d ProjectionFactor::sqrt_info;
double ProjectionFactor::sum_t;

ProjectionFactor::ProjectionFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j) : pts_i(_pts_i), pts_j(_pts_j)
{
#ifdef UNIT_SPHERE_ERROR
    Eigen::Vector3d b1, b2;
    Eigen::Vector3d a = pts_j.normalized();
    Eigen::Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b1 = (tmp - a * (a.transpose() * tmp)).normalized();
    b2 = a.cross(b1);
    tangent_base.block<1, 3>(0, 0) = b1.transpose();
    tangent_base.block<1, 3>(1, 0) = b2.transpose();
#endif
};

bool ProjectionFactor::Evaluate1(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    double inv_dep_i = parameters[3][0];

    Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
    Eigen::Map<Eigen::Vector2d> residual(residuals);

#ifdef UNIT_SPHERE_ERROR 
    residual =  tangent_base * (pts_camera_j.normalized() - pts_j.normalized());
#else
    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();
#endif

    residual = sqrt_info * residual;

    if (jacobians)
    {
        Eigen::Matrix3d Ri = Qi.toRotationMatrix();
        Eigen::Matrix3d Rj = Qj.toRotationMatrix();
        Eigen::Matrix3d ric = qic.toRotationMatrix();
        Eigen::Matrix<double, 2, 3> reduce(2, 3);
#ifdef UNIT_SPHERE_ERROR
        double norm = pts_camera_j.norm();
        Eigen::Matrix3d norm_jaco;
        double x1, x2, x3;
        x1 = pts_camera_j(0);
        x2 = pts_camera_j(1);
        x3 = pts_camera_j(2);
        norm_jaco << 1.0 / norm - x1 * x1 / pow(norm, 3), - x1 * x2 / pow(norm, 3),            - x1 * x3 / pow(norm, 3),
                     - x1 * x2 / pow(norm, 3),            1.0 / norm - x2 * x2 / pow(norm, 3), - x2 * x3 / pow(norm, 3),
                     - x1 * x3 / pow(norm, 3),            - x2 * x3 / pow(norm, 3),            1.0 / norm - x3 * x3 / pow(norm, 3);
        reduce = tangent_base * norm_jaco;
#else
        reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
            0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
#endif
        reduce = sqrt_info * reduce;

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

            Eigen::Matrix<double, 3, 6> jaco_i;
            jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
            jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Utility::skewSymmetric(pts_imu_i);

            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();
        }

        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

            Eigen::Matrix<double, 3, 6> jaco_j;
            jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
            jaco_j.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(pts_imu_j);

            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);
            Eigen::Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());
            Eigen::Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
            jaco_ex.rightCols<3>() = -tmp_r * Utility::skewSymmetric(pts_camera_i) + Utility::skewSymmetric(tmp_r * pts_camera_i) +
                                     Utility::skewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        if (jacobians[3])
        {
            Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[3]);
#if 1
            jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i * -1.0 / (inv_dep_i * inv_dep_i);
#else
            jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i;
#endif
        }
    }
    sum_t += tic_toc.toc();

    return true;
}

// 实现了用符号表达式来计算residual和jacobian
using Scalar = double;
bool ProjectionFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    double inv_dep_i = parameters[3][0];


    // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _Qi = Qi.coeffs();
  const Eigen::Matrix<Scalar, 4, 1>& _Qj = Qj.coeffs();
  const Eigen::Matrix<Scalar, 4, 1>& _qic = qic.coeffs();

  // Intermediate terms (373)
  const Scalar _tmp0 = std::pow(_qic[2], Scalar(2));
  const Scalar _tmp1 = -2 * _tmp0;
  const Scalar _tmp2 = std::pow(_qic[1], Scalar(2));
  const Scalar _tmp3 = 1 - 2 * _tmp2;
  const Scalar _tmp4 = _tmp1 + _tmp3;
  const Scalar _tmp5 = std::pow(_Qj[1], Scalar(2));
  const Scalar _tmp6 = -2 * _tmp5;
  const Scalar _tmp7 = std::pow(_Qj[2], Scalar(2));
  const Scalar _tmp8 = -2 * _tmp7;
  const Scalar _tmp9 = _tmp6 + _tmp8 + 1;
  const Scalar _tmp10 = 2 * _Qi[1];
  const Scalar _tmp11 = _Qi[0] * _tmp10;
  const Scalar _tmp12 = 2 * _Qi[2];
  const Scalar _tmp13 = _Qi[3] * _tmp12;
  const Scalar _tmp14 = -_tmp13;
  const Scalar _tmp15 = _tmp11 + _tmp14;
  const Scalar _tmp16 = std::pow(_qic[0], Scalar(2));
  const Scalar _tmp17 = -2 * _tmp16;
  const Scalar _tmp18 = _tmp1 + _tmp17 + 1;
  const Scalar _tmp19 = Scalar(1.0) / (inv_dep_i);
  const Scalar _tmp20 = _tmp19 * pts_i(1, 0);
  const Scalar _tmp21 = 2 * _qic[3];
  const Scalar _tmp22 = _qic[0] * _tmp21;
  const Scalar _tmp23 = -_tmp22;
  const Scalar _tmp24 = 2 * _qic[1];
  const Scalar _tmp25 = _qic[2] * _tmp24;
  const Scalar _tmp26 = _tmp23 + _tmp25;
  const Scalar _tmp27 = _tmp19 * pts_i(2, 0);
  const Scalar _tmp28 = _qic[2] * _tmp21;
  const Scalar _tmp29 = _qic[0] * _tmp24;
  const Scalar _tmp30 = _tmp28 + _tmp29;
  const Scalar _tmp31 = _tmp19 * pts_i(0, 0);
  const Scalar _tmp32 = _tmp18 * _tmp20 + _tmp26 * _tmp27 + _tmp30 * _tmp31 + tic(1, 0);
  const Scalar _tmp33 = std::pow(_Qi[2], Scalar(2));
  const Scalar _tmp34 = -2 * _tmp33;
  const Scalar _tmp35 = std::pow(_Qi[1], Scalar(2));
  const Scalar _tmp36 = -2 * _tmp35;
  const Scalar _tmp37 = _tmp34 + _tmp36 + 1;
  const Scalar _tmp38 = _qic[3] * _tmp24;
  const Scalar _tmp39 = 2 * _qic[0] * _qic[2];
  const Scalar _tmp40 = _tmp38 + _tmp39;
  const Scalar _tmp41 = -_tmp28;
  const Scalar _tmp42 = _tmp29 + _tmp41;
  const Scalar _tmp43 = _tmp20 * _tmp42 + _tmp27 * _tmp40 + _tmp31 * _tmp4 + tic(0, 0);
  const Scalar _tmp44 = _Qi[3] * _tmp10;
  const Scalar _tmp45 = _Qi[0] * _tmp12;
  const Scalar _tmp46 = _tmp44 + _tmp45;
  const Scalar _tmp47 = _tmp17 + _tmp3;
  const Scalar _tmp48 = -_tmp38;
  const Scalar _tmp49 = _tmp39 + _tmp48;
  const Scalar _tmp50 = _tmp22 + _tmp25;
  const Scalar _tmp51 = _tmp20 * _tmp50 + _tmp27 * _tmp47 + _tmp31 * _tmp49 + tic(2, 0);
  const Scalar _tmp52 = Pi(0, 0) - Pj(0, 0) + _tmp15 * _tmp32 + _tmp37 * _tmp43 + _tmp46 * _tmp51;
  const Scalar _tmp53 = 2 * _Qj[1];
  const Scalar _tmp54 = _Qj[3] * _tmp53;
  const Scalar _tmp55 = -_tmp54;
  const Scalar _tmp56 = 2 * _Qj[0] * _Qj[2];
  const Scalar _tmp57 = _tmp55 + _tmp56;
  const Scalar _tmp58 = -_tmp44;
  const Scalar _tmp59 = _tmp45 + _tmp58;
  const Scalar _tmp60 = _Qi[1] * _tmp12;
  const Scalar _tmp61 = 2 * _Qi[0] * _Qi[3];
  const Scalar _tmp62 = _tmp60 + _tmp61;
  const Scalar _tmp63 = std::pow(_Qi[0], Scalar(2));
  const Scalar _tmp64 = 1 - 2 * _tmp63;
  const Scalar _tmp65 = _tmp36 + _tmp64;
  const Scalar _tmp66 = Pi(2, 0) - Pj(2, 0) + _tmp32 * _tmp62 + _tmp43 * _tmp59 + _tmp51 * _tmp65;
  const Scalar _tmp67 = 2 * _Qj[3];
  const Scalar _tmp68 = _Qj[2] * _tmp67;
  const Scalar _tmp69 = _Qj[0] * _tmp53;
  const Scalar _tmp70 = _tmp68 + _tmp69;
  const Scalar _tmp71 = _tmp11 + _tmp13;
  const Scalar _tmp72 = _tmp34 + _tmp64;
  const Scalar _tmp73 = -_tmp61;
  const Scalar _tmp74 = _tmp60 + _tmp73;
  const Scalar _tmp75 = Pi(1, 0) - Pj(1, 0) + _tmp32 * _tmp72 + _tmp43 * _tmp71 + _tmp51 * _tmp74;
  const Scalar _tmp76 = _tmp57 * _tmp66 + _tmp70 * _tmp75;
  const Scalar _tmp77 = _tmp52 * _tmp9 + _tmp76 - tic(0, 0);
  const Scalar _tmp78 = std::pow(_Qj[0], Scalar(2));
  const Scalar _tmp79 = 1 - 2 * _tmp78;
  const Scalar _tmp80 = _tmp6 + _tmp79;
  const Scalar _tmp81 = _Qj[2] * _tmp53;
  const Scalar _tmp82 = _Qj[0] * _tmp67;
  const Scalar _tmp83 = -_tmp82;
  const Scalar _tmp84 = _tmp81 + _tmp83;
  const Scalar _tmp85 = _tmp54 + _tmp56;
  const Scalar _tmp86 = _tmp52 * _tmp85 + _tmp75 * _tmp84;
  const Scalar _tmp87 = _tmp66 * _tmp80 + _tmp86 - tic(2, 0);
  const Scalar _tmp88 = _tmp79 + _tmp8;
  const Scalar _tmp89 = _tmp81 + _tmp82;
  const Scalar _tmp90 = -_tmp68;
  const Scalar _tmp91 = _tmp69 + _tmp90;
  const Scalar _tmp92 = _tmp52 * _tmp91 + _tmp66 * _tmp89;
  const Scalar _tmp93 = _tmp75 * _tmp88 + _tmp92 - tic(1, 0);
  const Scalar _tmp94 = _tmp30 * _tmp93 + _tmp49 * _tmp87;
  const Scalar _tmp95 = _tmp4 * _tmp77 + _tmp94;
  const Scalar _tmp96 = _tmp26 * _tmp93 + _tmp40 * _tmp77;
  const Scalar _tmp97 = _tmp47 * _tmp87 + _tmp96;
  const Scalar _tmp98 = Scalar(1.0) / (_tmp97);
  const Scalar _tmp99 = _tmp95 * _tmp98 - pts_j(0, 0);
  const Scalar _tmp100 = Scalar(306.66666666666703) * _tmp99;
  const Scalar _tmp101 = _tmp42 * _tmp77 + _tmp50 * _tmp87;
  const Scalar _tmp102 = _tmp101 + _tmp18 * _tmp93;
  const Scalar _tmp103 = _tmp102 * _tmp98 - pts_j(1, 0);
  const Scalar _tmp104 = Scalar(306.66666666666703) * _tmp103;
  const Scalar _tmp105 = _tmp40 * _tmp9;
  const Scalar _tmp106 = _tmp47 * _tmp85;
  const Scalar _tmp107 = _tmp26 * _tmp91;
  const Scalar _tmp108 = _tmp105 + _tmp106 + _tmp107;
  const Scalar _tmp109 = std::pow(_tmp97, Scalar(-2));
  const Scalar _tmp110 = _tmp109 * _tmp95;
  const Scalar _tmp111 = _tmp4 * _tmp9;
  const Scalar _tmp112 = _tmp49 * _tmp85;
  const Scalar _tmp113 = _tmp30 * _tmp91;
  const Scalar _tmp114 = -_tmp108 * _tmp110 + _tmp98 * (_tmp111 + _tmp112 + _tmp113);
  const Scalar _tmp115 = Scalar(306.66666666666703) * _tmp114;
  const Scalar _tmp116 = _tmp102 * _tmp109;
  const Scalar _tmp117 = _tmp18 * _tmp91;
  const Scalar _tmp118 = _tmp50 * _tmp85;
  const Scalar _tmp119 = _tmp42 * _tmp9;
  const Scalar _tmp120 = -_tmp108 * _tmp116 + _tmp98 * (_tmp117 + _tmp118 + _tmp119);
  const Scalar _tmp121 = _tmp40 * _tmp70;
  const Scalar _tmp122 = _tmp47 * _tmp84;
  const Scalar _tmp123 = _tmp26 * _tmp88;
  const Scalar _tmp124 = _tmp121 + _tmp122 + _tmp123;
  const Scalar _tmp125 = _tmp49 * _tmp84;
  const Scalar _tmp126 = _tmp4 * _tmp70;
  const Scalar _tmp127 = _tmp30 * _tmp88;
  const Scalar _tmp128 = -_tmp110 * _tmp124 + _tmp98 * (_tmp125 + _tmp126 + _tmp127);
  const Scalar _tmp129 = Scalar(306.66666666666703) * _tmp128;
  const Scalar _tmp130 = _tmp18 * _tmp88;
  const Scalar _tmp131 = _tmp50 * _tmp84;
  const Scalar _tmp132 = _tmp42 * _tmp70;
  const Scalar _tmp133 = -_tmp116 * _tmp124 + _tmp98 * (_tmp130 + _tmp131 + _tmp132);
  const Scalar _tmp134 = Scalar(306.66666666666703) * _tmp133;
  const Scalar _tmp135 = _tmp47 * _tmp80;
  const Scalar _tmp136 = _tmp40 * _tmp57;
  const Scalar _tmp137 = _tmp26 * _tmp89;
  const Scalar _tmp138 = _tmp135 + _tmp136 + _tmp137;
  const Scalar _tmp139 = _tmp49 * _tmp80;
  const Scalar _tmp140 = _tmp4 * _tmp57;
  const Scalar _tmp141 = _tmp30 * _tmp89;
  const Scalar _tmp142 = -_tmp110 * _tmp138 + _tmp98 * (_tmp139 + _tmp140 + _tmp141);
  const Scalar _tmp143 = Scalar(306.66666666666703) * _tmp142;
  const Scalar _tmp144 = _tmp18 * _tmp89;
  const Scalar _tmp145 = _tmp50 * _tmp80;
  const Scalar _tmp146 = _tmp42 * _tmp57;
  const Scalar _tmp147 = -_tmp116 * _tmp138 + _tmp98 * (_tmp144 + _tmp145 + _tmp146);
  const Scalar _tmp148 = Scalar(306.66666666666703) * _tmp147;
  const Scalar _tmp149 = std::pow(_Qi[3], Scalar(2));
  const Scalar _tmp150 = -_tmp35;
  const Scalar _tmp151 = _tmp149 + _tmp150;
  const Scalar _tmp152 = -_tmp63;
  const Scalar _tmp153 = _tmp152 + _tmp33;
  const Scalar _tmp154 = -_tmp60;
  const Scalar _tmp155 = _tmp32 * (_tmp151 + _tmp153) + _tmp51 * (_tmp154 + _tmp73);
  const Scalar _tmp156 = -_tmp149;
  const Scalar _tmp157 = _tmp32 * _tmp74 + _tmp51 * (_tmp150 + _tmp156 + _tmp33 + _tmp63);
  const Scalar _tmp158 = -_tmp11;
  const Scalar _tmp159 = _tmp32 * _tmp46 + _tmp51 * (_tmp13 + _tmp158);
  const Scalar _tmp160 = _tmp155 * _tmp89 + _tmp157 * _tmp88 + _tmp159 * _tmp91;
  const Scalar _tmp161 = _tmp155 * _tmp80 + _tmp157 * _tmp84 + _tmp159 * _tmp85;
  const Scalar _tmp162 = _tmp155 * _tmp57 + _tmp157 * _tmp70 + _tmp159 * _tmp9;
  const Scalar _tmp163 = _tmp160 * _tmp26 + _tmp161 * _tmp47 + _tmp162 * _tmp40;
  const Scalar _tmp164 =
      -_tmp110 * _tmp163 + _tmp98 * (_tmp160 * _tmp30 + _tmp161 * _tmp49 + _tmp162 * _tmp4);
  const Scalar _tmp165 = Scalar(306.66666666666703) * _tmp164;
  const Scalar _tmp166 =
      -_tmp116 * _tmp163 + _tmp98 * (_tmp160 * _tmp18 + _tmp161 * _tmp50 + _tmp162 * _tmp42);
  const Scalar _tmp167 = Scalar(306.66666666666703) * _tmp166;
  const Scalar _tmp168 = -_tmp33;
  const Scalar _tmp169 = _tmp168 + _tmp63;
  const Scalar _tmp170 = _tmp156 + _tmp35;
  const Scalar _tmp171 = _tmp43 * (_tmp169 + _tmp170) + _tmp51 * _tmp59;
  const Scalar _tmp172 = _tmp43 * (_tmp154 + _tmp61) + _tmp51 * _tmp71;
  const Scalar _tmp173 = -_tmp45;
  const Scalar _tmp174 = _tmp43 * (_tmp173 + _tmp58) + _tmp51 * (_tmp151 + _tmp169);
  const Scalar _tmp175 = _tmp171 * _tmp89 + _tmp172 * _tmp88 + _tmp174 * _tmp91;
  const Scalar _tmp176 = _tmp171 * _tmp57 + _tmp172 * _tmp70 + _tmp174 * _tmp9;
  const Scalar _tmp177 = _tmp171 * _tmp80 + _tmp172 * _tmp84 + _tmp174 * _tmp85;
  const Scalar _tmp178 = _tmp175 * _tmp26 + _tmp176 * _tmp40 + _tmp177 * _tmp47;
  const Scalar _tmp179 =
      -_tmp110 * _tmp178 + _tmp98 * (_tmp175 * _tmp30 + _tmp176 * _tmp4 + _tmp177 * _tmp49);
  const Scalar _tmp180 = Scalar(306.66666666666703) * _tmp179;
  const Scalar _tmp181 =
      -_tmp116 * _tmp178 + _tmp98 * (_tmp175 * _tmp18 + _tmp176 * _tmp42 + _tmp177 * _tmp50);
  const Scalar _tmp182 = _tmp15 * _tmp43 + _tmp32 * (_tmp153 + _tmp170);
  const Scalar _tmp183 =
      _tmp32 * (_tmp14 + _tmp158) + _tmp43 * (_tmp149 + _tmp152 + _tmp168 + _tmp35);
  const Scalar _tmp184 = _tmp32 * (_tmp173 + _tmp44) + _tmp43 * _tmp62;
  const Scalar _tmp185 = _tmp182 * _tmp91 + _tmp183 * _tmp88 + _tmp184 * _tmp89;
  const Scalar _tmp186 = _tmp182 * _tmp85 + _tmp183 * _tmp84 + _tmp184 * _tmp80;
  const Scalar _tmp187 = _tmp182 * _tmp9 + _tmp183 * _tmp70 + _tmp184 * _tmp57;
  const Scalar _tmp188 = _tmp185 * _tmp26 + _tmp186 * _tmp47 + _tmp187 * _tmp40;
  const Scalar _tmp189 =
      -_tmp110 * _tmp188 + _tmp98 * (_tmp185 * _tmp30 + _tmp186 * _tmp49 + _tmp187 * _tmp4);
  const Scalar _tmp190 =
      -_tmp116 * _tmp188 + _tmp98 * (_tmp18 * _tmp185 + _tmp186 * _tmp50 + _tmp187 * _tmp42);
  const Scalar _tmp191 = -_tmp105 - _tmp106 - _tmp107;
  const Scalar _tmp192 = -_tmp110 * _tmp191 + _tmp98 * (-_tmp111 - _tmp112 - _tmp113);
  const Scalar _tmp193 = Scalar(306.66666666666703) * _tmp192;
  const Scalar _tmp194 = -_tmp116 * _tmp191 + _tmp98 * (-_tmp117 - _tmp118 - _tmp119);
  const Scalar _tmp195 = -_tmp121 - _tmp122 - _tmp123;
  const Scalar _tmp196 = -_tmp110 * _tmp195 + _tmp98 * (-_tmp125 - _tmp126 - _tmp127);
  const Scalar _tmp197 = Scalar(306.66666666666703) * _tmp196;
  const Scalar _tmp198 = -_tmp116 * _tmp195 + _tmp98 * (-_tmp130 - _tmp131 - _tmp132);
  const Scalar _tmp199 = Scalar(306.66666666666703) * _tmp198;
  const Scalar _tmp200 = -_tmp135 - _tmp136 - _tmp137;
  const Scalar _tmp201 = -_tmp110 * _tmp200 + _tmp98 * (-_tmp139 - _tmp140 - _tmp141);
  const Scalar _tmp202 = Scalar(306.66666666666703) * _tmp201;
  const Scalar _tmp203 = -_tmp116 * _tmp200 + _tmp98 * (-_tmp144 - _tmp145 - _tmp146);
  const Scalar _tmp204 = Scalar(306.66666666666703) * _tmp203;
  const Scalar _tmp205 = -_tmp81;
  const Scalar _tmp206 = std::pow(_Qj[3], Scalar(2));
  const Scalar _tmp207 = -_tmp206;
  const Scalar _tmp208 = -_tmp5;
  const Scalar _tmp209 = -_tmp69;
  const Scalar _tmp210 = _tmp52 * (_tmp209 + _tmp68) + _tmp66 * (_tmp205 + _tmp83) +
                         _tmp75 * (_tmp207 + _tmp208 + _tmp7 + _tmp78);
  const Scalar _tmp211 = -_tmp78;
  const Scalar _tmp212 = _tmp211 + _tmp7;
  const Scalar _tmp213 = _tmp206 + _tmp208;
  const Scalar _tmp214 = _tmp66 * (_tmp212 + _tmp213) + _tmp86;
  const Scalar _tmp215 = _tmp210 * _tmp47 + _tmp214 * _tmp26;
  const Scalar _tmp216 = -_tmp110 * _tmp215 + _tmp98 * (_tmp210 * _tmp49 + _tmp214 * _tmp30);
  const Scalar _tmp217 = Scalar(306.66666666666703) * _tmp216;
  const Scalar _tmp218 = -_tmp116 * _tmp215 + _tmp98 * (_tmp18 * _tmp214 + _tmp210 * _tmp50);
  const Scalar _tmp219 = Scalar(306.66666666666703) * _tmp218;
  const Scalar _tmp220 = _tmp207 + _tmp5;
  const Scalar _tmp221 = -_tmp7;
  const Scalar _tmp222 = _tmp221 + _tmp78;
  const Scalar _tmp223 = -_tmp56;
  const Scalar _tmp224 =
      _tmp52 * (_tmp223 + _tmp55) + _tmp66 * (_tmp220 + _tmp222) + _tmp75 * (_tmp205 + _tmp82);
  const Scalar _tmp225 = _tmp52 * (_tmp213 + _tmp222) + _tmp76;
  const Scalar _tmp226 = _tmp224 * _tmp40 + _tmp225 * _tmp47;
  const Scalar _tmp227 = -_tmp110 * _tmp226 + _tmp98 * (_tmp224 * _tmp4 + _tmp225 * _tmp49);
  const Scalar _tmp228 = Scalar(306.66666666666703) * _tmp227;
  const Scalar _tmp229 = -_tmp116 * _tmp226 + _tmp98 * (_tmp224 * _tmp42 + _tmp225 * _tmp50);
  const Scalar _tmp230 = Scalar(306.66666666666703) * _tmp229;
  const Scalar _tmp231 = _tmp75 * (_tmp206 + _tmp211 + _tmp221 + _tmp5) + _tmp92;
  const Scalar _tmp232 =
      _tmp52 * (_tmp212 + _tmp220) + _tmp66 * (_tmp223 + _tmp54) + _tmp75 * (_tmp209 + _tmp90);
  const Scalar _tmp233 = _tmp231 * _tmp40 + _tmp232 * _tmp26;
  const Scalar _tmp234 = -_tmp110 * _tmp233 + _tmp98 * (_tmp231 * _tmp4 + _tmp232 * _tmp30);
  const Scalar _tmp235 = Scalar(306.66666666666703) * _tmp234;
  const Scalar _tmp236 = -_tmp116 * _tmp233 + _tmp98 * (_tmp18 * _tmp232 + _tmp231 * _tmp42);
  const Scalar _tmp237 = Scalar(306.66666666666703) * _tmp236;
  const Scalar _tmp238 = _tmp37 * _tmp85 + _tmp59 * _tmp80 + _tmp71 * _tmp84;
  const Scalar _tmp239 = _tmp37 * _tmp91 + _tmp59 * _tmp89 + _tmp71 * _tmp88;
  const Scalar _tmp240 = _tmp37 * _tmp9 + _tmp57 * _tmp59 + _tmp70 * _tmp71 - 1;
  const Scalar _tmp241 = _tmp238 * _tmp47 + _tmp239 * _tmp26 + _tmp240 * _tmp40;
  const Scalar _tmp242 =
      -_tmp110 * _tmp241 + _tmp98 * (_tmp238 * _tmp49 + _tmp239 * _tmp30 + _tmp240 * _tmp4);
  const Scalar _tmp243 = Scalar(306.66666666666703) * _tmp242;
  const Scalar _tmp244 =
      -_tmp116 * _tmp241 + _tmp98 * (_tmp18 * _tmp239 + _tmp238 * _tmp50 + _tmp240 * _tmp42);
  const Scalar _tmp245 = _tmp15 * _tmp85 + _tmp62 * _tmp80 + _tmp72 * _tmp84;
  const Scalar _tmp246 = _tmp15 * _tmp9 + _tmp57 * _tmp62 + _tmp70 * _tmp72;
  const Scalar _tmp247 = _tmp15 * _tmp91 + _tmp62 * _tmp89 + _tmp72 * _tmp88 - 1;
  const Scalar _tmp248 = _tmp245 * _tmp47 + _tmp246 * _tmp40 + _tmp247 * _tmp26;
  const Scalar _tmp249 =
      -_tmp110 * _tmp248 + _tmp98 * (_tmp245 * _tmp49 + _tmp246 * _tmp4 + _tmp247 * _tmp30);
  const Scalar _tmp250 = Scalar(306.66666666666703) * _tmp249;
  const Scalar _tmp251 =
      -_tmp116 * _tmp248 + _tmp98 * (_tmp18 * _tmp247 + _tmp245 * _tmp50 + _tmp246 * _tmp42);
  const Scalar _tmp252 = Scalar(306.66666666666703) * _tmp251;
  const Scalar _tmp253 = _tmp46 * _tmp9 + _tmp57 * _tmp65 + _tmp70 * _tmp74;
  const Scalar _tmp254 = _tmp46 * _tmp91 + _tmp65 * _tmp89 + _tmp74 * _tmp88;
  const Scalar _tmp255 = _tmp46 * _tmp85 + _tmp65 * _tmp80 + _tmp74 * _tmp84 - 1;
  const Scalar _tmp256 = _tmp253 * _tmp40 + _tmp254 * _tmp26 + _tmp255 * _tmp47;
  const Scalar _tmp257 =
      -_tmp110 * _tmp256 + _tmp98 * (_tmp253 * _tmp4 + _tmp254 * _tmp30 + _tmp255 * _tmp49);
  const Scalar _tmp258 =
      -_tmp116 * _tmp256 + _tmp98 * (_tmp18 * _tmp254 + _tmp253 * _tmp42 + _tmp255 * _tmp50);
  const Scalar _tmp259 = -_tmp25;
  const Scalar _tmp260 = _tmp23 + _tmp259;
  const Scalar _tmp261 = -_tmp2;
  const Scalar _tmp262 = _tmp0 + _tmp261;
  const Scalar _tmp263 = std::pow(_qic[3], Scalar(2));
  const Scalar _tmp264 = -_tmp263;
  const Scalar _tmp265 = _tmp16 + _tmp264;
  const Scalar _tmp266 = _tmp262 + _tmp265;
  const Scalar _tmp267 = _tmp20 * _tmp26 + _tmp266 * _tmp27;
  const Scalar _tmp268 = -_tmp16;
  const Scalar _tmp269 = _tmp263 + _tmp268;
  const Scalar _tmp270 = _tmp262 + _tmp269;
  const Scalar _tmp271 = _tmp20 * _tmp270 + _tmp260 * _tmp27;
  const Scalar _tmp272 = -_tmp29;
  const Scalar _tmp273 = _tmp272 + _tmp28;
  const Scalar _tmp274 = _tmp20 * _tmp40 + _tmp27 * _tmp273;
  const Scalar _tmp275 = _tmp267 * _tmp72 + _tmp271 * _tmp74 + _tmp274 * _tmp71;
  const Scalar _tmp276 = _tmp267 * _tmp62 + _tmp271 * _tmp65 + _tmp274 * _tmp59;
  const Scalar _tmp277 = _tmp15 * _tmp267 + _tmp271 * _tmp46 + _tmp274 * _tmp37;
  const Scalar _tmp278 = _tmp275 * _tmp88 + _tmp276 * _tmp89 + _tmp277 * _tmp91;
  const Scalar _tmp279 = _tmp275 * _tmp84 + _tmp276 * _tmp80 + _tmp277 * _tmp85;
  const Scalar _tmp280 = _tmp275 * _tmp70 + _tmp276 * _tmp57 + _tmp277 * _tmp9;
  const Scalar _tmp281 = _tmp26 * _tmp278 + _tmp260 * _tmp87 + _tmp266 * _tmp93 + _tmp273 * _tmp77 +
                         _tmp279 * _tmp47 + _tmp280 * _tmp40;
  const Scalar _tmp282 =
      -_tmp110 * _tmp281 + _tmp98 * (_tmp278 * _tmp30 + _tmp279 * _tmp49 + _tmp280 * _tmp4);
  const Scalar _tmp283 =
      -_tmp116 * _tmp281 +
      _tmp98 * (_tmp18 * _tmp278 + _tmp270 * _tmp87 + _tmp279 * _tmp50 + _tmp280 * _tmp42 + _tmp96);
  const Scalar _tmp284 = -_tmp0;
  const Scalar _tmp285 = _tmp2 + _tmp284;
  const Scalar _tmp286 = _tmp265 + _tmp285;
  const Scalar _tmp287 = _tmp27 * _tmp49 + _tmp286 * _tmp31;
  const Scalar _tmp288 = _tmp16 + _tmp261 + _tmp263 + _tmp284;
  const Scalar _tmp289 = -_tmp39;
  const Scalar _tmp290 = _tmp289 + _tmp48;
  const Scalar _tmp291 = _tmp27 * _tmp288 + _tmp290 * _tmp31;
  const Scalar _tmp292 = _tmp22 + _tmp259;
  const Scalar _tmp293 = _tmp27 * _tmp30 + _tmp292 * _tmp31;
  const Scalar _tmp294 = _tmp287 * _tmp65 + _tmp291 * _tmp59 + _tmp293 * _tmp62;
  const Scalar _tmp295 = _tmp15 * _tmp293 + _tmp287 * _tmp46 + _tmp291 * _tmp37;
  const Scalar _tmp296 = _tmp287 * _tmp74 + _tmp291 * _tmp71 + _tmp293 * _tmp72;
  const Scalar _tmp297 = _tmp294 * _tmp57 + _tmp295 * _tmp9 + _tmp296 * _tmp70;
  const Scalar _tmp298 = _tmp294 * _tmp80 + _tmp295 * _tmp85 + _tmp296 * _tmp84;
  const Scalar _tmp299 = _tmp294 * _tmp89 + _tmp295 * _tmp91 + _tmp296 * _tmp88;
  const Scalar _tmp300 =
      _tmp26 * _tmp299 + _tmp288 * _tmp77 + _tmp297 * _tmp40 + _tmp298 * _tmp47 + _tmp94;
  const Scalar _tmp301 =
      -_tmp110 * _tmp300 + _tmp98 * (_tmp286 * _tmp87 + _tmp290 * _tmp77 + _tmp292 * _tmp93 +
                                     _tmp297 * _tmp4 + _tmp298 * _tmp49 + _tmp299 * _tmp30);
  const Scalar _tmp302 = Scalar(306.66666666666703) * _tmp301;
  const Scalar _tmp303 =
      -_tmp116 * _tmp300 + _tmp98 * (_tmp18 * _tmp299 + _tmp297 * _tmp42 + _tmp298 * _tmp50);
  const Scalar _tmp304 = _tmp289 + _tmp38;
  const Scalar _tmp305 = _tmp20 * _tmp304 + _tmp31 * _tmp50;
  const Scalar _tmp306 = _tmp0 + _tmp2 + _tmp264 + _tmp268;
  const Scalar _tmp307 = _tmp20 * _tmp306 + _tmp31 * _tmp42;
  const Scalar _tmp308 = _tmp269 + _tmp285;
  const Scalar _tmp309 = _tmp272 + _tmp41;
  const Scalar _tmp310 = _tmp20 * _tmp309 + _tmp308 * _tmp31;
  const Scalar _tmp311 = _tmp305 * _tmp74 + _tmp307 * _tmp71 + _tmp310 * _tmp72;
  const Scalar _tmp312 = _tmp15 * _tmp310 + _tmp305 * _tmp46 + _tmp307 * _tmp37;
  const Scalar _tmp313 = _tmp305 * _tmp65 + _tmp307 * _tmp59 + _tmp310 * _tmp62;
  const Scalar _tmp314 = _tmp311 * _tmp70 + _tmp312 * _tmp9 + _tmp313 * _tmp57;
  const Scalar _tmp315 = _tmp311 * _tmp84 + _tmp312 * _tmp85 + _tmp313 * _tmp80;
  const Scalar _tmp316 = _tmp311 * _tmp88 + _tmp312 * _tmp91 + _tmp313 * _tmp89;
  const Scalar _tmp317 = _tmp26 * _tmp316 + _tmp314 * _tmp40 + _tmp315 * _tmp47;
  const Scalar _tmp318 =
      -_tmp110 * _tmp317 +
      _tmp98 * (_tmp101 + _tmp30 * _tmp316 + _tmp308 * _tmp93 + _tmp314 * _tmp4 + _tmp315 * _tmp49);
  const Scalar _tmp319 =
      -_tmp116 * _tmp317 + _tmp98 * (_tmp18 * _tmp316 + _tmp304 * _tmp87 + _tmp306 * _tmp77 +
                                     _tmp309 * _tmp93 + _tmp314 * _tmp42 + _tmp315 * _tmp50);
  const Scalar _tmp320 = Scalar(306.66666666666703) * _tmp319;
  const Scalar _tmp321 = std::pow(inv_dep_i, Scalar(-2));
  const Scalar _tmp322 = _tmp321 * pts_i(0, 0);
  const Scalar _tmp323 = _tmp321 * pts_i(2, 0);
  const Scalar _tmp324 = _tmp321 * pts_i(1, 0);
  const Scalar _tmp325 = -_tmp322 * _tmp4 - _tmp323 * _tmp40 - _tmp324 * _tmp42;
  const Scalar _tmp326 = -_tmp18 * _tmp324 - _tmp26 * _tmp323 - _tmp30 * _tmp322;
  const Scalar _tmp327 = -_tmp322 * _tmp49 - _tmp323 * _tmp47 - _tmp324 * _tmp50;
  const Scalar _tmp328 = _tmp15 * _tmp326 + _tmp325 * _tmp37 + _tmp327 * _tmp46;
  const Scalar _tmp329 = _tmp325 * _tmp71 + _tmp326 * _tmp72 + _tmp327 * _tmp74;
  const Scalar _tmp330 = _tmp325 * _tmp59 + _tmp326 * _tmp62 + _tmp327 * _tmp65;
  const Scalar _tmp331 = _tmp328 * _tmp9 + _tmp329 * _tmp70 + _tmp330 * _tmp57;
  const Scalar _tmp332 = _tmp328 * _tmp85 + _tmp329 * _tmp84 + _tmp330 * _tmp80;
  const Scalar _tmp333 = _tmp328 * _tmp91 + _tmp329 * _tmp88 + _tmp330 * _tmp89;
  const Scalar _tmp334 = _tmp26 * _tmp333 + _tmp331 * _tmp40 + _tmp332 * _tmp47;
  const Scalar _tmp335 =
      -_tmp110 * _tmp334 + _tmp98 * (_tmp30 * _tmp333 + _tmp331 * _tmp4 + _tmp332 * _tmp49);
  const Scalar _tmp336 = Scalar(306.66666666666703) * _tmp335;
  const Scalar _tmp337 =
      -_tmp116 * _tmp334 + _tmp98 * (_tmp18 * _tmp333 + _tmp331 * _tmp42 + _tmp332 * _tmp50);
  const Scalar _tmp338 = Scalar(94044.444444444671);
  const Scalar _tmp339 = _tmp128 * _tmp338;
  const Scalar _tmp340 = _tmp120 * _tmp338;
  const Scalar _tmp341 = _tmp147 * _tmp338;
  const Scalar _tmp342 = _tmp142 * _tmp338;
  const Scalar _tmp343 = _tmp164 * _tmp338;
  const Scalar _tmp344 = _tmp181 * _tmp338;
  const Scalar _tmp345 = _tmp114 * _tmp338;
  const Scalar _tmp346 = _tmp189 * _tmp338;
  const Scalar _tmp347 = _tmp192 * _tmp338;
  const Scalar _tmp348 = _tmp196 * _tmp338;
  const Scalar _tmp349 = _tmp203 * _tmp338;
  const Scalar _tmp350 = _tmp201 * _tmp338;
  const Scalar _tmp351 = _tmp216 * _tmp338;
  const Scalar _tmp352 = _tmp227 * _tmp338;
  const Scalar _tmp353 = _tmp236 * _tmp338;
  const Scalar _tmp354 = _tmp234 * _tmp338;
  const Scalar _tmp355 = _tmp242 * _tmp338;
  const Scalar _tmp356 = _tmp249 * _tmp338;
  const Scalar _tmp357 = _tmp257 * _tmp338;
  const Scalar _tmp358 = _tmp282 * _tmp338;
  const Scalar _tmp359 = _tmp303 * _tmp338;
  const Scalar _tmp360 = _tmp301 * _tmp338;
  const Scalar _tmp361 = _tmp335 * _tmp338;
  const Scalar _tmp362 = _tmp166 * _tmp338;
  const Scalar _tmp363 = _tmp190 * _tmp338;
  const Scalar _tmp364 = _tmp194 * _tmp338;
  const Scalar _tmp365 = _tmp198 * _tmp338;
  const Scalar _tmp366 = _tmp133 * _tmp338;
  const Scalar _tmp367 = _tmp319 * _tmp338;
  const Scalar _tmp368 = _tmp218 * _tmp338;
  const Scalar _tmp369 = _tmp258 * _tmp338;
  const Scalar _tmp370 = _tmp337 * _tmp338;
  const Scalar _tmp371 = _tmp244 * _tmp338;
  const Scalar _tmp372 = _tmp229 * _tmp338;

  // Output terms (4)
//   if (res != nullptr) {
    // Eigen::Matrix<Scalar, 2, 1>& _res = (*res);
    Eigen::Map<Eigen::Vector2d> residual(residuals);

    residual(0, 0) = _tmp100;
    residual(1, 0) = _tmp104;
//   }

//   if (jacobian != nullptr) 
  if (jacobians)
  {
    // Eigen::Matrix<Scalar, 2, 19>& _jacobian = (*jacobian);
    if (jacobians[0])
    {
        Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

        jacobian_pose_i(0, 0) = _tmp115;
        jacobian_pose_i(1, 0) = Scalar(306.66666666666703) * _tmp120;
        jacobian_pose_i(0, 1) = _tmp129;
        jacobian_pose_i(1, 1) = _tmp134;
        jacobian_pose_i(0, 2) = _tmp143;
        jacobian_pose_i(1, 2) = _tmp148;
        jacobian_pose_i(0, 3) = _tmp165;
        jacobian_pose_i(1, 3) = _tmp167;
        jacobian_pose_i(0, 4) = _tmp180;
        jacobian_pose_i(1, 4) = Scalar(306.66666666666703) * _tmp181;
        jacobian_pose_i(0, 5) = Scalar(306.66666666666703) * _tmp189;
        jacobian_pose_i(1, 5) = Scalar(306.66666666666703) * _tmp190;
        jacobian_pose_i(0, 6) = 0;
        jacobian_pose_i(1, 6) = 0;
    }

    if (jacobians[1])
    {
        Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);
        jacobian_pose_j(0, 0) = _tmp193;
        jacobian_pose_j(1, 0) = Scalar(306.66666666666703) * _tmp194;
        jacobian_pose_j(0, 1) = _tmp197;
        jacobian_pose_j(1, 1) = _tmp199;
        jacobian_pose_j(0, 2) = _tmp202;
        jacobian_pose_j(1, 2) = _tmp204;
        jacobian_pose_j(0, 3) = _tmp217;
        jacobian_pose_j(1, 3) = _tmp219;
        jacobian_pose_j(0, 4) = _tmp228;
        jacobian_pose_j(1, 4) = _tmp230;
        jacobian_pose_j(0, 5) = _tmp235;
        jacobian_pose_j(1, 5) = _tmp237;
        jacobian_pose_j(0, 6) = 0;
        jacobian_pose_j(1, 6) = 0;
    }

    if (jacobians[2])
    {
        Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);
        jacobian_ex_pose(0, 0) = _tmp243;
        jacobian_ex_pose(1, 0) = Scalar(306.66666666666703) * _tmp244;
        jacobian_ex_pose(0, 1) = _tmp250;
        jacobian_ex_pose(1, 1) = _tmp252;
        jacobian_ex_pose(0, 2) = Scalar(306.66666666666703) * _tmp257;
        jacobian_ex_pose(1, 2) = Scalar(306.66666666666703) * _tmp258;
        jacobian_ex_pose(0, 3) = Scalar(306.66666666666703) * _tmp282;
        jacobian_ex_pose(1, 3) = Scalar(306.66666666666703) * _tmp283;
        jacobian_ex_pose(0, 4) = _tmp302;
        jacobian_ex_pose(1, 4) = Scalar(306.66666666666703) * _tmp303;
        jacobian_ex_pose(0, 5) = Scalar(306.66666666666703) * _tmp318;
        jacobian_ex_pose(1, 5) = _tmp320;
        jacobian_ex_pose(0, 6) = 0;
        jacobian_ex_pose(1, 6) = 0;
    }

    if (jacobians[3])
    {
        Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[3]);
        jacobian_feature(0, 0) = _tmp336;
        jacobian_feature(1, 0) = Scalar(306.66666666666703) * _tmp337;
    }
  }

    sum_t += tic_toc.toc();

    return true;
}

void ProjectionFactor::check(double **parameters)
{
    double *res = new double[15];
    double **jaco = new double *[4];
    jaco[0] = new double[2 * 7];
    jaco[1] = new double[2 * 7];
    jaco[2] = new double[2 * 7];
    jaco[3] = new double[2 * 1];
    Evaluate(parameters, res, jaco);
    puts("check begins");

    puts("my");

    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 1>>(res).transpose() << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[0]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[1]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[2]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Vector2d>(jaco[3]) << std::endl
              << std::endl;

    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);
    double inv_dep_i = parameters[3][0];

    Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);


    Eigen::Vector2d residual;
#ifdef UNIT_SPHERE_ERROR 
    residual =  tangent_base * (pts_camera_j.normalized() - pts_j.normalized());
#else
    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();
#endif
    residual = sqrt_info * residual;

    puts("num");
    std::cout << residual.transpose() << std::endl;

    const double eps = 1e-6;
    Eigen::Matrix<double, 2, 19> num_jacobian;
    for (int k = 0; k < 19; k++)
    {
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

        Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);
        double inv_dep_i = parameters[3][0];

        int a = k / 3, b = k % 3;
        Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

        if (a == 0)
            Pi += delta;
        else if (a == 1)
            Qi = Qi * Utility::deltaQ(delta);
        else if (a == 2)
            Pj += delta;
        else if (a == 3)
            Qj = Qj * Utility::deltaQ(delta);
        else if (a == 4)
            tic += delta;
        else if (a == 5)
            qic = qic * Utility::deltaQ(delta);
        else if (a == 6)
            inv_dep_i += delta.x();

        Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
        Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
        Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
        Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
        Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);

        Eigen::Vector2d tmp_residual;
#ifdef UNIT_SPHERE_ERROR 
        tmp_residual =  tangent_base * (pts_camera_j.normalized() - pts_j.normalized());
#else
        double dep_j = pts_camera_j.z();
        tmp_residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();
#endif
        tmp_residual = sqrt_info * tmp_residual;
        num_jacobian.col(k) = (tmp_residual - residual) / eps;
    }
    std::cout << num_jacobian << std::endl;
}
