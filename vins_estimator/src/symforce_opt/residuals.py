
# created by wxliu on 2024-6-19

import symforce.symbolic as sf
import symforce
symforce.set_epsilon_to_symbol()

# from symforce import codegen
# from symforce.codegen import values_codegen
# from symforce.values import Values

FOCAL_LENGTH: double = 460.0
# sqrt_info: sf.M22 = FOCAL_LENGTH / 1.5 * sf.Matrix22.eye()
sqrt_info: sf.M22 = FOCAL_LENGTH / 1.5 * sf.I22(2, 2)

M_PI = 3.14159265358979323846
LIGHT_SPEED = 2.99792458e8
EARTH_ECCE_2 = 6.69437999014e-3
EARTH_SEMI_MAJOR = 6378137
R2D = 180.0 / M_PI
D2R = M_PI / 180.0


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
    corrected_delta_p: sf.V3,
    corrected_delta_q: sf.Quaternion,
    corrected_delta_v: sf.V3,
    G: sf.V3, # gravity
    sum_dt: sf.Scalar
) -> sf.Matrix:
    r_p = Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p
    r_q = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).xyz
    r_v = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v
    r_ba = Baj - Bai
    r_bg = Bgj - Bgi

    return sf.Matrix.block_matrix([[r_p], [r_q], [r_v], [r_ba], [r_bg]])

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
    epsilon: sf.Scalar
 ) -> sf.V2:
    # pts_camera_i = pts_i / inv_dep_i
    pts_camera_i = pts_i / (inv_dep_i + epsilon)
    pts_imu_i = qic * pts_camera_i + tic
    pts_w = Qi * pts_imu_i + Pi
    pts_imu_j = Qj.inverse() * (pts_w - Pj)
    pts_camera_j = qic.inverse() * (pts_imu_j - tic)

    # 求归一化平面的重投影误差
    # dep_j = pts_camera_j.z
    # residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>()
    pts_camera_j_Normalized = pts_camera_j / (pts_camera_j.z + epsilon)
    r_x = pts_camera_j_Normalized.x - pts_j.x
    r_y = pts_camera_j_Normalized.y - pts_j.y
    residual = sf.V2(r_x, r_y)
    residual = sqrt_info * residual

    return residual

# anchor point 位姿残差：6维
def anchor_pose_residual(
    curr_p: sf.V3,
    curr_q: sf.Rot3,
    anchor_p: sf.V3,
    anchor_q: sf.Rot3
) -> sf.V6:
    r_p = curr_p - anchor_p
    r_q = 2.0 * (curr_q * anchor_q.inverse()).xyz
    residual = sf.Matrix.block_matrix([[r_p], [r_q]])

    return residual

# # simplified gnss psr dopp residual: 2维
# def gnss_psr_dopp_residual2(
#     psr_estimated: sf.Scalar,
#     psr_measured: sf.scalar,

# )

# ecef2rotation：根据anchor point的坐标计算ENU系到ECEF系的旋转
from symengine import atan
# def ecef2rotation(xyz: sf.V3, epsilon: sf.Scalar) -> sf.Matrix33:
def ecef2geo(xyz: sf.V3, epsilon: sf.Scalar = 0.0) -> sf.V3:
    # step 1: ecef2geo 计算得到纬度，经度，（海拔）高度
    lla = sf.V3.zero()
    # TODO: LLA coordinate is not defined if x = 0 and y = 0
    # if xyz.x == 0 && xyz.y == 0 : return lla
    # 可是函数不允许有分支, 这里引入epsilon视图解决问题

    e2 = EARTH_ECCE_2
    a = EARTH_SEMI_MAJOR
    a2 = a * a
    b2 = a2 * (1 - e2)
    b = sf.sqrt(b2)
    ep2 = (a2 - b2) / (b2 + epsilon)
    x, y, z = xyz
    xy = sf.V2(x, y)
    p = xy.norm(epsilon=epsilon)

    # two sides and hypotenuse of right angle triangle with one angle = theta:
    s1 = xyz.z * a
    s2 = p * b
    h = sf.sqrt(s1 * s1 + s2 * s2 + epsilon**2)
    sin_theta = s1 / h
    cos_theta = s2 / h

    # two sides and hypotenuse of right angle triangle with one angle = lat:
    # s1 = xyz.z + ep2 * b * (sin_theta ** 3)
    s1 = xyz.z + ep2 * b * sf.Pow(sin_theta, 3)
    s2 = p - a * e2 * sf.Pow(cos_theta, 3)
    h = sf.sqrt(s1 * s1 + s2 * s2 + epsilon**2)
    tan_lat = s1 / (s2 + epsilon)
    sin_lat = s1 / (h + epsilon)
    cos_lat = s2 / (h + epsilon)
    lat = atan(tan_lat)
    lat_deg = lat * R2D

    N = a2 * sf.Pow((a2 * cos_lat * cos_lat + b2 * sin_lat * sin_lat), -0.5)
    altM = p / (cos_lat + epsilon) - N

    lon = sf.atan2(xyz.y(), xyz.x(), epsilon=epsilon)
    lon_deg = lon * R2D
    lla = sf.V3(lat_deg, lon_deg, altM)
    return lla

def ecef2rotation(xyz: sf.V3, epsilon: sf.Scalar) -> sf.Matrix33:
    # step2: geo2rotation 计算ENU系到ECEF系的旋转
    # ref_geo = lla
    ref_geo = ecef2geo(xyz, epsilon = epsilon)
    lat = ref_geo.x * D2R
    lon = ref_geo.y * D2R
    sin_lat = sf.sin(lat)
    cos_lat = sf.cos(lat)
    sin_lon = sf.sin(lon)
    cos_lon = sf.cos(lon)

    # Given the ECEF coordinate of the anchor point, the rotation from ENU frame to ECEF frame is:
    R_ecef_enu = sf.Matrix33(-sin_lon, -sin_lat*cos_lon, cos_lat*cos_lon, \
                            cos_lon, -sin_lat*sin_lon, cos_lat*sin_lon, \
                            0,  cos_lat, sin_lat)
    return R_ecef_enu

def ecef2enu(ref_lla: sf.V3, v_ecef: sf.V3) -> sf.V3:
    lat = ref_lla.x * D2R
    lon = ref_lla.y * D2R
    sin_lat = sf.sin(lat)
    cos_lat = sf.cos(lat)
    sin_lon = sf.sin(lon)
    cos_lon = sf.cos(lon)
    Eigen::Matrix3d R_enu_ecef
    R_enu_ecef = sf.Matrix33(-sin_lon,             cos_lon,         0, \
                             -sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat, \
                             cos_lat*cos_lon,  cos_lat*sin_lon, sin_lat)
    return (R_enu_ecef * v_ecef)

# 计算卫星到接收机的方位角/仰角
def sat_azel(rev_pos: sf.V3, sat_pos: sf.V3, epsilon: sf.Scalar = 0) -> sf.V2:
    rev_lla = ecef2geo(rev_pos)
    rev2sat_ecef = (sat_pos - rev_pos).normalized() # .normalized(epsilon=epsilon) # .normalized(epsilon=0)
    rev2sat_enu = ecef2enu(rev_lla, rev2sat_ecef)
    
    # azel[0] = rev2sat_ecef.head<2>().norm() < 1e-12 ? 0.0 : atan2(rev2sat_enu.x(), rev2sat_enu.y());
    # azel[0] += (azel[0] < 0 ? 2*M_PI : 0);
    # azel[1] = asin(rev2sat_enu.z());

    # branchless singularity handling
    # TODO: 计算出的方位角，有可能为负吗？为负应该加上2PI，可是又不允许分支
    azimuth = sf.atan2(rev2sat_enu.x, rev2sat_enu.y, epsilon = epsilon)

    elevation = sf.asin(rev2sat_enu.z)

    return sf.V2(azimuth, elevation)

def calculate_trop_delay()

# gnss psr dopp residual: 2维
def gnss_psr_dopp_residual(
    # states:
    Pi: sf.V3,
    Vi: sf.V3,
    Pj: sf.V3,
    Vj: sf.V3,
    rcv_dt: sf.Scalar,
    rcv_ddt: sf.Scalar,
    yaw_diff: sf.Scalar,
    ref_ecef: sf.V3,
    # ?
    # ion_delay: sf.Scalar,
    # tro_delay: sf.Scalar,

    # precomputed:
    ratio: sf.Scalar,
    tgd: sf.Scalar,
    sv_pos: sf.Scalar,
    sv_vel: sf.Scalar,
    svdt: sf.Scalar,
    freq: sf.Scalar,
    psr_measured: sf.Scalar,
    pr_uura: sf.Scalar,
    dp_uura: sf.Scalar,
    epsilon: sf.Scalar = 0
) -> sf.V2:
    # construct residuals here.
    local_pos = ratio * Pi + (1.0 - ratio) * Pj
    local_vel = ratio * Vi + (1.0 - ratio) * Vj
    sin_yaw_diff = sf.sin(yaw_diff)
    cos_yaw_diff = sf.cos(yaw_diff)
    R_enu_local = sf.Matrix33(cos_yaw_diff, -sin_yaw_diff, 0, \
                              sin_yaw_diff, cos_yaw_diff, 0, \
                              0, 0, 1)
    # 计算地心地固坐标系下的位置和速度
    R_ecef_enu = ecef2rotation(ref_ecef, epsilon)
    R_ecef_local = R_ecef_enu * R_enu_local
    P_ecef = R_ecef_local * local_pos + ref_ecef
    V_ecef = R_ecef_local * local_vel

    # 计算卫星的方位角/仰角
    azel = sat_azel(P_ecef, sv_pos, epsilon)
    rcv_lla = ecef2geo(P_ecef)
    # TODO: 
    # tro_delay = calculate_trop_delay(obs->time, rcv_lla, azel)
    # ion_delay = calculate_ion_delay(obs->time, iono_paras, rcv_lla, azel)

    return sf.V2(0, 0)

# 