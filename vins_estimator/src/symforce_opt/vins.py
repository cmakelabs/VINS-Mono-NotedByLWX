import symforce
symforce.set_epsilon_to_symbol()

from symforce import typing as T # 导入symforce包里面的typing模块到当前模块的命名空间，并重命名为T

from symforce.opt.noise_models import BarronNoiseModel
from symforce.opt.noise_models import ScalarNoiseModel

from symforce import codegen
from symforce.codegen import codegen_util

import symforce.symbolic as sf
from symforce.notebook_util import display

import shutil
from pathlib import Path

# FOCAL_LENGTH: double = 460.0
FOCAL_LENGTH = 460.0
# sqrt_info: sf.M22 = FOCAL_LENGTH / 1.5 * sf.Matrix22.eye()
sqrt_info: sf.M22 = FOCAL_LENGTH / 1.5 * sf.I22(2, 2)

# 重投影残差：2维
def projection_residual(
    pts_i: sf.V3,
    pts_j: sf.V3,
    Pi: sf.V3,
    Qi: sf.Rot3,
    Pj: sf.V3,
    Qj: sf.Rot3,
    tic: sf.V3,
    qic: sf.Rot3,
    inv_dep_i: sf.Scalar,
    #sqrt_info: sf.M22,
    # epsilon: sf.Scalar
    weight: sf.Scalar, # 2024-7-13
    epsilon: sf.Scalar,
    noise_model: ScalarNoiseModel,
 ) -> sf.V2:
    pts_camera_i = pts_i / inv_dep_i
    # pts_camera_i = pts_i / (inv_dep_i + epsilon)
    pts_imu_i = qic * pts_camera_i + tic
    pts_w = Qi * pts_imu_i + Pi
    pts_imu_j = Qj.inverse() * (pts_w - Pj)
    pts_camera_j = qic.inverse() * (pts_imu_j - tic)

    # 求归一化平面的重投影误差
    # dep_j = pts_camera_j.z
    # residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>()
    # pts_camera_j_Normalized = pts_camera_j / pts_camera_j.z
    pts_camera_j_Normalized = pts_camera_j / (pts_camera_j.z + epsilon)
    r_x = pts_camera_j_Normalized.x - pts_j.x
    r_y = pts_camera_j_Normalized.y - pts_j.y
    residual = sf.V2(r_x, r_y)
    # residual = sqrt_info * residual
    # return residual

    # 2024-7-13.
    warp_is_valid = 1
    reprojection_error = sqrt_info * residual
    whitened_residual = (
        warp_is_valid * sf.sqrt(weight) * noise_model.whiten_norm(reprojection_error, epsilon)
    )

    return whitened_residual

def projection_gnc_residual(
    pts_i: sf.V3,
    pts_j: sf.V3,
    Pi: sf.V3,
    Qi: sf.Rot3,
    Pj: sf.V3,
    Qj: sf.Rot3,
    tic: sf.V3,
    qic: sf.Rot3,
    inv_dep_i: sf.Scalar,
    #sqrt_info: sf.M22,
    weight: sf.Scalar,
    gnc_mu: sf.Scalar,
    gnc_scale: sf.Scalar,
    epsilon: sf.Scalar,
 ) -> sf.V2:
    # TODO:
    noise_model = BarronNoiseModel(
        alpha=BarronNoiseModel.compute_alpha_from_mu(gnc_mu, epsilon),
        scalar_information=1 / gnc_scale**2,
        x_epsilon=epsilon,
    )

    return projection_residual(
        pts_i,
        pts_j,
        Pi,
        Qi,
        Pj,
        Qj,
        tic,
        qic,
        inv_dep_i,
        weight,
        epsilon,
        noise_model,
    )

def deltaQ1(theta: sf.M31) -> sf.Quaternion:
    half_theta = theta
    half_theta /= 2.0
    # w = 1.0
    # x = half_theta.x
    # y = half_theta.y
    # z = half_theta.z

    return sf.Quaternion(xyz=half_theta, w=1.0)

def deltaQ(theta: sf.M31) -> sf.Rot3: # 2024-7-11
    half_theta = theta
    half_theta /= 2.0
    # w = 1.0
    # x = half_theta.x
    # y = half_theta.y
    # z = half_theta.z

    # return sf.Rot3(sf.Quaternion(xyz=half_theta, w=1.0))    
    return sf.Rot3.from_storage([half_theta.x, half_theta.y, half_theta.z, 1.0])    

# imu残差：15维
def imu_residual(
    Pi: sf.V3,
    Qi: sf.Rot3, # sf.Quaternion
    Vi: sf.V3,
    Bai: sf.V3,
    Bgi: sf.V3,
    Pj: sf.V3,
    Qj: sf.Rot3,
    Vj: sf.V3,
    Baj: sf.V3,
    Bgj: sf.V3,
    delta_p: sf.V3,
    # delta_q: sf.Quaternion,
    delta_q: sf.Rot3, # seems to don't have sym::Quaternion, so use sf.Rot3 here 2024-7-11
    delta_v: sf.V3,
    G: sf.V3, # gravity
    sum_dt: sf.Scalar,
    dp_dba: sf.M33,
    dp_dbg: sf.M33,
    dq_dbg: sf.M33,
    dv_dba: sf.M33,
    dv_dbg: sf.M33,
    linearized_ba: sf.V3,
    linearized_bg: sf.V3,
    sqrt_info: sf.Matrix(15, 15) # newly add on 2024-7-11
) -> sf.Matrix:
    dba = Bai - linearized_ba
    dbg = Bgi - linearized_bg

    corrected_delta_q = delta_q * deltaQ(dq_dbg * dbg)
    corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg
    corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg

    r_p = Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p
    r_q = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).q.xyz # uncomment on 2024-7-11
    # corrected_delta_q2 = sf.Rot3(corrected_delta_q) # comment on 2024-7-11
    # r_q = 2 * (corrected_delta_q2.inverse() * (Qi.inverse() * Qj)).q.xyz # comment on 2024-7-11
    r_v = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v
    r_ba = Baj - Bai
    r_bg = Bgj - Bgi

    # return sf.Matrix.block_matrix([[r_p], [r_q], [r_v], [r_ba], [r_bg]])
    return sqrt_info * sf.Matrix.block_matrix([[r_p], [r_q], [r_v], [r_ba], [r_bg]]) # 2024-7-11



output_dir="/root/dev/python_ws/test_sym"

# for projection
def generate_projection_residual_code(
    output_dir: T.Optional[Path] = None, print_code: bool = False
) -> None:
    projection_codegen = codegen.Codegen.function(
        # func=projection_residual,
        func=projection_gnc_residual,
        config=codegen.CppConfig(),
    )

    projection_data = projection_codegen.generate_function(output_dir)

    projection_codegen_with_linearization = projection_codegen.with_linearization(which_args=["Pi", "Qi", "Pj", "Qj", "tic", "qic", "inv_dep_i"])

    # 生成构建因子图的函数
    # Generate the function and print the code
    metadata = projection_codegen_with_linearization.generate_function(
        output_dir=output_dir, skip_directory_nesting=False
    )

# R = sf.Rot3.symbolic("R")
# display(f"R={R}")
# display(f"R.q={R.q}")
# display(f"R.q.xyz=\n{R.q.xyz}")

# display(R.to_storage())

# # display(sf.Rot3.symbolic("R"))

# display(sf.Quaternion.symbolic("q"))

# display(sf.Rot3())

# for imu
def generate_imu_residual_code(
    output_dir: T.Optional[Path] = None, print_code: bool = False
) -> None:
    imu_codegen = codegen.Codegen.function(
        func=imu_residual,
        config=codegen.CppConfig(),
    )

    imu_data = imu_codegen.generate_function(output_dir)

    imu_codegen_with_linearization = imu_codegen.with_linearization(which_args=["Pi", "Qi", "Vi", "Bai", "Bgi", "Pj", "Qj", "Vj", "Baj", "Bgj"])

    # 生成构建因子图的函数
    # Generate the function and print the code
    metadata = imu_codegen_with_linearization.generate_function(
        output_dir=output_dir, skip_directory_nesting=False
    )

generate_projection_residual_code(output_dir)

# generate_imu_residual_code(output_dir)

# Qi: sf.Rot3 = sf.Rot3.symbolic("Qi")
# Qj: sf.Rot3 = sf.Rot3.symbolic("Qj")
# corrected_delta_q: sf.Quaternion = sf.Quaternion.symbolic("delta_q")
# corrected_delta_q2 = sf.Rot3(corrected_delta_q)
# r_q = 2 * (corrected_delta_q2.inverse() * (Qi.inverse() * Qj)).q.xyz
# display(f"r_q=\n{r_q}")
