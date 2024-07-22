
#include "estimator.h"

#include <spdlog/spdlog.h>
#include <sym/util/epsilon.h>
#include <symforce/opt/assert.h>
#include <symforce/opt/factor.h>
#include <symforce/opt/key.h>
#include <symforce/opt/optimization_stats.h>
#include <symforce/opt/optimizer.h>
#include <lcmtypes/sym/optimizer_params_t.hpp>
#include <symforce/opt/dense_cholesky_solver.h>
#include <symforce/opt/sparse_cholesky/sparse_cholesky_solver.h>
#include <symforce/opt/sparse_schur_solver.h>

#include "symforce_opt/gen/marginalization_factor_manual.h"
#include "symforce_opt/gen/imu_factor.h"
#include "symforce_opt/gen/projection_factor.h"
#include "symforce_opt/gen/projection_gnc_factor.h"

std::vector<sym::EParameterType> remain_Keys; // 2024-7-11.

template <typename Scalar>
using DenseOptimizer =
    sym::Optimizer<Scalar, sym::LevenbergMarquardtSolver<Scalar, sym::DenseCholeskySolver<Scalar>>>; // 2024-7-13

/*
* can't use.
template <typename Scalar>
using SparseSchurOptimizer =
    sym::Optimizer<Scalar, sym::LevenbergMarquardtSolver<Scalar, sym::SparseSchurSolver<Eigen::SparseMatrix<Scalar>>>>;
*/    

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    clearState();

    // for test 2024-7-8
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        std::cout << "i=" << i << " para_Pose addr=" << reinterpret_cast<long>(para_Pose[i]) << std::endl;
        std::cout << "i=" << i << " para_SpeedBias addr=" << reinterpret_cast<long>(para_SpeedBias[i]) << std::endl;
    }
    
    std::cout << "para_Ex_Pose addr=" << reinterpret_cast<long>(para_Ex_Pose[0]);
    // the end.
}

/**
 * @brief 外参，重投影置信度，延时设置
 * 
 */
void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    // 这里可以看到虚拟相机的用法
    //视觉测量残差的协方差矩阵的逆，即信息矩阵
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
}

// 所有状态全部重置 //清空或初始化滑动窗口中所有的状态量
void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;


    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}

/**
 * @brief   处理IMU数据，包括更新预积分量，和提供优化状态量的初始值
 * @Description IMU预积分，中值积分得到当前PQV作为优化初值
 * @param[in]   dt 时间间隔
 * @param[in]   linear_acceleration 线加速度
 * @param[in]   angular_velocity 角速度
 * @return  void
*/
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    // 滑窗中保留11帧，frame_count表示现在处理第几帧，一般处理到第11帧时就保持不变了
    // 由于预积分是帧间约束，因此第1个预积分量实际上是用不到的
    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    // 所以只有大于0才处理
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
            // 这个量用来做初始化用的，imu与视觉sfm对齐时会用到
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        // 保存传感器数据
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        // 又是一个中值积分，更新滑窗中状态量，本质是给非线性优化提供可信的初始值
        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        //采用的是中值积分的传播方式
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

/**
 * @brief   处理图像特征数据
 * @Description addFeatureCheckParallax()添加特征点到feature中，计算点跟踪的次数和视差，判断是否是关键帧               
 *              判断并进行外参标定
 *              进行视觉惯性联合初始化或基于滑动窗口非线性优化的紧耦合VIO
 * @param[in]   image 某帧所有特征点的[camera_id,[x,y,z,u,v,vx,vy]]所构成的map,索引为feature_id, 对于单目来说camera_id=0
 * @param[in]   header 某帧图像的头信息
 * @return  void
*/
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header)
{
    // image的构成：feature_id：[camera_id,[x,y,z,u,v,vx,vy]]
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    // Step 1 将特征点信息加到f_manager这个特征点管理器中，同时进行是否关键帧的检查
    //添加之前检测到的特征点到feature容器中，计算每一个点跟踪的次数，以及它的视差
    //通过检测两帧之间的视差决定次新帧是否作为关键帧
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
        marginalization_flag = MARGIN_OLD; // 如果上一帧是关键帧，则滑窗中最老的帧就要被移出滑窗
    else
        marginalization_flag = MARGIN_SECOND_NEW; // 否则移除上一帧（第二新的图像，次新帧），意味着只有关键帧才能进到滑窗里面

    ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    // all_image_frame用来做初始化相关操作，他保留滑窗起始到当前的所有帧
    // 有一些帧会因为不是KF，被MARGIN_SECOND_NEW，但是及时较新的帧被margin，他也会保留在这个容器中，因为初始化要求使用所有的帧，而非只要KF
    // all_image_frame还有一个点需要注意：当系统一直无法初始化时，其容量会一直增长，导致内存占用升高
    //将图像数据、时间、临时预积分值存到图像帧类中
    ImageFrame imageframe(image, header.stamp.toSec());
    imageframe.pre_integration = tmp_pre_integration;
    // 这里就是简单的把图像和预积分绑定在一起，这里预积分就是两帧之间的，滑窗中实际上是两个KF之间的
    // 实际上是准备用来初始化的相关数据
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));
    //更新临时预积分初始值
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    // 没有外参初值
    // Step 2： 外参初始化
    if(ESTIMATE_EXTRINSIC == 2)//如果没有外参则进行标定
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            // 这里标定imu和相机的旋转外参的初值
            // 因为预积分是相邻帧的约束，因为这里得到的图像关联也是相邻的
            //得到两帧之间归一化特征点
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            //标定从camera到IMU之间的旋转矩阵
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                // 标志位设置成可信的外参初值
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    if (solver_flag == INITIAL) //初始化
    {
        if (frame_count == WINDOW_SIZE)
        {
            bool result = false;
            // 要有可信的外参值，同时距离上次初始化不成功至少相邻0.1s
            //有外参且当前帧时间戳大于初始化时间戳0.1秒，就进行初始化操作
            // Step 3： VIO初始化
            if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
                //视觉惯性联合初始化
               result = initialStructure();
               //更新初始化时间戳
               initial_timestamp = header.stamp.toSec();
            }
            if(result) //初始化成功
            {
                //先进行一次滑动窗口非线性优化，得到当前帧以及第一帧的位姿
                solver_flag = NON_LINEAR;
                // Step 4： 非线性优化求解VIO
                solveOdometry();
                // Step 5： 滑动窗口
                slideWindow();
                // Step 6： 移除无效地图点
                f_manager.removeFailures();
                ROS_INFO("Initialization finish!");
                last_R = Rs[WINDOW_SIZE]; // 滑窗里最新的位姿
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0]; // 滑窗里最老的位姿
                last_P0 = Ps[0];
                
            }
            else
                slideWindow(); //初始化失败则直接滑动窗口
        }
        else
            frame_count++;
    }
    else //紧耦合的非线性优化
    {
        TicToc t_solve;
        solveOdometry();
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        //故障检测与恢复,一旦检测到故障，系统将切换回初始化阶段
        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin;
        slideWindow();
        f_manager.removeFailures();
        ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS
        // key_poses给可视化用的
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}

/**
 * @brief   视觉的结构初始化
 * @Description 确保IMU有充分运动激励
 *              relativePose()找到具有足够视差的两帧,由F矩阵恢复R、t作为初始值
 *              sfm.construct() 全局纯视觉SFM 恢复滑动窗口帧的位姿
 *              visualInitialAlign()视觉惯性联合初始化
 * @return  bool true:初始化成功
*/
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    // Step 1 通过加速度标准差判断IMU是否有充分运动以初始化。
    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        // 从第二帧开始检查imu
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1); //均值
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            // 求方差
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        // 得到标准差
        var = sqrt(var / ((int)all_image_frame.size() - 1));//标准差
        //ROS_WARN("IMU variation %f!", var);
        // 实际上检查结果并没有用到
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // Step 2 做一个纯视觉sfm
    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1; // 虽命名imu_j，但这个跟imu无关，就是存储观测特征点的帧的索引
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    //保证具有足够的视差,由F矩阵恢复Rt
    //第l帧是从第一帧开始到滑动窗口中第一个满足与当前帧的平均视差足够大的帧，会作为参考帧到下面的全局sfm使用
    //此处的relative_R，relative_T为当前帧到参考帧（第l帧）的坐标系变换Rt
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    //对窗口中每个图像帧求解sfm问题
    //得到所有图像帧相对于参考帧的姿态四元数Q、平移向量T和特征点坐标sfm_tracked_points。
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        //求解失败则边缘化最早一帧并滑动窗口
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    // Step 3 对于所有的图像帧，包括不在滑动窗口中的，提供初始的RT估计，然后solvePnP进行优化,得到每一帧的姿态
    // step2只是针对KF进行sfm，初始化需要all_image_frame中的所有元素，因此下面通过KF来求解其他的非KF的位姿
    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    // i代表跟这个帧最近的KF的索引
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        // 这一帧本身就是KF，因此可以直接得到位姿
        if((frame_it->first) == Headers[i].stamp.toSec())
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose(); // 得到Rwi
            frame_it->second.T = T[i]; // 初始化不估计平移外参
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i].stamp.toSec())
        {
            i++;
        }
        // 最近的KF提供一个初始值，Twc -> Tcw
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        //罗德里格斯公式将旋转矩阵转换成旋转向量
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        // 遍历这一帧对应的特征点
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            // 由于是单目，这里id_pts.second大小就是1
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end()) // 有对应的三角化出来的3d点
                {
                    Vector3d world_pts = it->second; // 地图点的世界坐标
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        /** 
         *bool cv::solvePnP(    求解pnp问题
         *   InputArray  objectPoints,   特征点的3D坐标数组
         *   InputArray  imagePoints,    特征点对应的图像坐标
         *   InputArray  cameraMatrix,   相机内参矩阵
         *   InputArray  distCoeffs,     失真系数的输入向量
         *   OutputArray     rvec,       旋转向量
         *   OutputArray     tvec,       平移向量
         *   bool    useExtrinsicGuess = false, 为真则使用提供的初始估计值
         *   int     flags = SOLVEPNP_ITERATIVE 采用LM优化
         *)   
         */
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        // cv -> eigen,同时Tcw -> Twc
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        // Twc -> Twi
        // 由于尺度未恢复，因此平移暂时不转到imu系
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    // 到此就求解出用来做视觉惯性对齐的所有视觉帧的位姿
    // Step 4 视觉惯性对齐
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);

    //triangulat on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        f_manager.triangulate(Ps, tic, ric);
        ROS_DEBUG("triangulation costs %f", t_tri.toc());
        // TicToc t_opt;
        // optimization();
        // std::cout << "opt:" << t_opt.toc() << std::endl;
        symOptimization(); // 2024-7-12.
        // std::cout << "sym opt:" << t_opt.toc() << std::endl;
    }
}

/**
 * @brief 由于ceres的参数块都是double数组，因此这里把参数块从eigen的表示转成double数组
 * 
 */
//vector转换成double数组，因为ceres使用数值数组
//Ps、Rs转变成para_Pose，Vs、Bas、Bgs转变成para_SpeedBias
void Estimator::vector2double()
{
    // KF的位姿
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    // 外参
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    // 特征点逆深度
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    // 传感器时间同步
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

/**
 * @brief double -> eigen 同时fix第一帧的yaw和平移，固定了四自由度的零空间
 * 
 */
// 数据转换，vector2double的相反过程
// 同时这里为防止优化结果往零空间变化，会根据优化前后第一帧的位姿差进行修正。
void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if(relocalization_info)
    { 
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) + origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;   
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;

    }
}

bool Estimator::failureDetection()
{
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_INFO(" big z translation");
        return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

/**
 * @brief   基于滑动窗口紧耦合的非线性优化，残差项的构造和求解
 * @Description 添加要优化的变量 (p,v,q,ba,bg) 一共15个自由度，IMU的外参也可以加进来
 *              添加残差，残差项分为4块 先验残差+IMU残差+视觉残差+闭环检测残差
 *              根据倒数第二帧是不是关键帧确定边缘化的结果           
 * @return      void
*/
void Estimator::optimization()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0); // temp comment
    // loss_function = NULL; // temply add for test.
    // Step 1 定义待优化的参数块，类似g2o的顶点。参数块：即待优化的变量
    // 参数块 1： 滑窗中位姿包括位置和姿态，共11帧
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        // 由于姿态不满足正常的加法，也就是李群上没有加法，因此需要自己定义它的加法
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    // 参数块 2： 相机imu间的外参
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC)
        {
            ROS_DEBUG("fix extinsic param");
            // 如果不需要优化外参就设置为fix
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }
    // 参数块 3： 时间延迟TD
    // 传感器的时间同步：相机和IMU硬件不同步时估计两者的时间偏差
    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1);
        //problem.SetParameterBlockConstant(para_Td[0]);
    }

    // 参数块，实际上还有地图点（的逆深度），其实频繁的参数块不需要调用AddParameterBlock，增加残差块接口时会自动绑定

    TicToc t_whole, t_prepare;
    // 转换eigen -> double
    vector2double();

    // Step 2 通过残差约束来添加残差块，类似g2o的边
    // 上一次的边缘化结果作为这一次的先验
    // if (0 && last_marginalization_info)
    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        // last_marginalization_parameter_blocks存储的是待优化的状态量的地址的vector
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    //imu预积分的约束
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        // 时间过长这个约束就不可信了
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        // imu预积分是从第1帧到第10帧总计10帧，而位姿是从第0帧到第10帧，总计11帧
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        // 11个视觉帧之间的imu,总共就只有10帧，即只有10个预积分增量的约束
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }
    int f_m_cnt = 0;
    int feature_index = -1;
    // 视觉重投影的约束
    // 遍历每一个特征点
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // 进行特征点有效性的检查
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
 
        ++feature_index;

        // 第一个观测到这个特征点的帧idx
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        // 特征点在第一个帧下的归一化相机系坐标
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        // 遍历看到这个特征点的所有KF
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j) // 自己跟自己不能形成重投影
            {
                continue;
            }
            // 取出另一帧的归一化相机坐标
            Vector3d pts_j = it_per_frame.point;
            // 带有时间延时的是另一种形式
            if (ESTIMATE_TD)
            {
                    ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                     it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                     it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                    /*
                    double **para = new double *[5];
                    para[0] = para_Pose[imu_i];
                    para[1] = para_Pose[imu_j];
                    para[2] = para_Ex_Pose[0];
                    para[3] = para_Feature[feature_index];
                    para[4] = para_Td[0];
                    f_td->check(para);
                    */
            }
            else
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j); // 构造函数就是同一个特征点在不同帧的观测
                // 约束的变量是该特征点的第一个观测帧以及其他一个观测帧，加上外参和特征点逆深度
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());
    // std::cout << "prepare time cost:" << t_prepare.toc() << std::endl;

    // 回环检测相关的约束
    if(relocalization_info)
    {
        //printf("set relocalization factor! \n");
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization); // 需要优化的回环帧位姿
        int retrive_feature_index = 0;
        int feature_index = -1;
        // 遍历现有地图点
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            ++feature_index;
            int start = it_per_id.start_frame;
            if(start <= relo_frame_local_index) // 这个地图点能被对应的当前帧看到
            {   
                // 寻找回环帧能看到的地图点
                while((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)
                {
                    retrive_feature_index++;
                }
                // 这个地图点也能被回环帧看到
                if((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    // 构建一个重投影约束，这个地图点的起始帧和该回环帧之间
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                    
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
                    retrive_feature_index++;
                }     
            }
        }

    }

    // Step 3 ceres优化求解
    ceres::Solver::Options options;

    // options.linear_solver_type = ceres::DENSE_SCHUR; // tmp comment 2024-7-18
    // options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY; // test 2024-7-13 can not use if strategy is ceres::LEVENBERG_MARQUARDT
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; // test 2024-7-18 can use if strategy is ceres::LEVENBERG_MARQUARDT, but is not applicable to marginalization.
    //options.num_threads = 2;
    // options.trust_region_strategy_type = ceres::DOGLEG; // tmp comment 2024-7-18
    // test 2024-7-18.
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.min_lm_diagonal = 1.0e-8;
    options.max_lm_diagonal = 1.0e6;
    // the end.
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0; // 下面的边缘化老的操作比较多，因此给他优化时间就少一些
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary); // ceres优化求解
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solver costs: %f", t_solver.toc());

    // 把优化后double -> eigen
    double2vector();

    // Step 4 边缘化
    // 科普一下舒尔补
    TicToc t_whole_marginalization;
    // 如果次新帧是关键帧，将边缘化最老帧，及其看到的路标点和IMU数据，将其转化为先验：
    if (marginalization_flag == MARGIN_OLD)
    {
        int marg_feature_count = 0; // added on 2024-7-5
        // 一个用来边缘化操作的对象
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        // 这里类似手写高斯牛顿，因此也需要都转成double数组
        vector2double();
        // 关于边缘化有几点注意的地方
        // 1、找到需要边缘化的参数块，这里是地图点，第0帧位姿，第0帧速度零偏
        // 2、找到构造高斯牛顿下降时跟这些待边缘化相关的参数块有关的残差约束，那就是预积分约束，重投影约束，以及上一次边缘化约束
        // 3、这些约束连接的参数块中，不需要被边缘化的参数块，就是被提供先验约束的部分，也就是滑窗中剩下的位姿和速度零偏

        //1、将上一次先验残差项传递给marginalization_info
        // 上一次的边缘化结果
        if (last_marginalization_info)
        {
            vector<int> drop_set;
            // last_marginalization_parameter_blocks是上一次边缘化对哪些当前参数块有约束
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                // 涉及到的待边缘化的上一次边缘化留下来的当前参数块只有位姿和速度零偏
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // 处理方式和其他残差块相同
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        //2、将第0帧和第1帧间的IMU因子IMUFactor(pre_integrations[1])，添加到marginalization_info中
        // 只有第1个预积分和待边缘化参数块相连
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                // 跟构建ceres约束问题一样，这里也需要得到残差和雅克比
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1}); // 这里就是第0和1个参数块是需要被边缘化的
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        //3、将第一次观测为第0帧的所有路标点对应的视觉观测，添加到marginalization_info中
        // 遍历视觉重投影的约束
        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                // 只找能被第0帧看到的特征点
                if (imu_i != 0)
                    continue;

                marg_feature_count++;
                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                // 遍历看到这个特征点的所有KF，通过这个特征点，建立和第0帧的约束
                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.point;
                    // 根据是否约束延时确定残差阵
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        // 重点说一下vector<int>{0, 3}: 表示要被drop或者marg掉的参数序号有0和3，对应到的参数块即为para_Pose[imu_i]和para_Feature[feature_index]，即第0帧的位姿和路标点需要被marg
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3}); // 这里第0帧和地图点被margin
                        marginalization_info->addResidualBlockInfo(residual_block_info);                        
                    }
                }
            }
        }
        std::cout << "marg old: marg_feature_count=" << marg_feature_count << std::endl;

        // 所有的残差块都收集好了
        TicToc t_pre_margin;
        //4、计算每个残差，对应的Jacobian，并将各参数块拷贝到统一的内存（parameter_block_data）中
        // 进行预处理
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        //5、多线程构造先验项舒尔补AX=b的结构，在X0处线性化计算Jacobian和残差
        // 边缘化操作
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        //6.调整参数块在下一次窗口中对应的位置（往前移一格），注意这里是指针，后面slideWindow中会赋新值，这里只是提前占座
        // 即将滑窗，因此记录新地址对应的老地址
        // 关于addr_shift补充说明一点：其作用是为了记录边缘化之后保留的参数块，在滑窗中新的位置。
        // 比如marg老帧时，原来滑窗第1帧的位姿其位置就移动到第0帧（往前移一格），以此类推
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            // 位姿和速度都要滑窗移动
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        // 外参和时间延时不变
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        // parameter_blocks实际上就是addr_shift的索引的集合及搬进去的新地址
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info; // 本次边缘化的所有信息
        last_marginalization_parameter_blocks = parameter_blocks; // 代表该次边缘化对某些参数块形成约束，这些参数块在滑窗之后的地址
        setRemainParameterKey(); // 2024-7-9
        
    }
    else // 如果次新帧不是关键帧：// 边缘化倒数第二帧
    {
        // 要求有上一次边缘化的结果，同时即将被margin掉的（para_Pose[WINDOW_SIZE - 1]）在上一次边缘化后的约束中
        // 预积分结果合并，因此只有位姿margin掉
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1])) // 统计para_Pose[WINDOW_SIZE - 1]在上一次边缘化的参数块中出现的次数
        {

            //1.保留次新帧的IMU测量，丢弃该帧的视觉测量，将上一次先验残差项传递给marginalization_info
            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    // 速度零偏只会margin第1个（并且只有在marg老帧的时候，才会marg第一个速度零偏），不可能出现倒数第二个
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    // 这种case只会margin掉倒数第二个位姿
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                // 这里只会更新一下margin factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            // 这里的操作如出一辙
            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            //2、premargin
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            //3、marginalize
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            //4.调整参数块在下一次窗口中对应的位置（去掉次新帧）
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE) // 滑窗，最新帧成为次新帧
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else // 其他不变
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }
            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            setRemainParameterKey(); // 2024-7-9
            
        }
    }
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());
    
    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0].stamp.toSec();
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;
 
                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }

                all_image_frame.erase(all_image_frame.begin(), it_0);
                all_image_frame.erase(t_0);

            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew();
        }
    }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}

void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t = _relo_t;
    prev_relo_r = _relo_r;
    for(int i = 0; i < WINDOW_SIZE; i++)
    {
        if(relo_frame_stamp == Headers[i].stamp.toSec())
        {
            relo_frame_local_index = i;
            relocalization_info = 1;
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j];
        }
    }
}

void Estimator::setRemainParameterKey()
{
    // return ;
    remain_Keys.clear();

    int i = 0, j = 0;
    int remain_count = last_marginalization_parameter_blocks.size();
    // std::cout << "remain paramters count=" << remain_count << std::endl;
    char szTmp[10] = { 0 };

    for(i = 0; i < remain_count; i++)
    {
        double *addr = last_marginalization_parameter_blocks.at(i);

        for(j = 0; j <= WINDOW_SIZE; j++)
        {
            if(addr == para_Pose[j])
            {
                // char szTmp[10] = { 0 };
                // sprintf(szTmp, "pose%d", j);
                // remain_Keys.emplace_back(szTmp);
                remain_Keys.emplace_back((sym::EParameterType)j);
                break ;
            }
            // else if(addr == para_SpeedBias[j])
            // {
            //     // char szTmp[10] = { 0 };
            //     sprintf(szTmp, "vbabg%d", j);
            //     remain_Keys.emplace_back(szTmp);
            //     break ;
            // }
        }

        if(addr == para_SpeedBias[0])
        {
            remain_Keys.emplace_back(sym::VBABG0);
        }

        if(addr == para_Ex_Pose[0])
        {
            // char szTmp[10] = { "ex_pose" };
            // strcpy(szTmp, "ex_pose");
            // remain_Keys.emplace_back(szTmp);

            remain_Keys.emplace_back(sym::EX_POSE);
        }
    }

    // std::cout << "output remain keys. size=" << remain_Keys.size() << std::endl;
    // for(auto key: remain_Keys)
    // {
    //     std::cout << key << std::endl;
    // }

    sym::g_pMarginalizationInfo = last_marginalization_info;
    sym::g_pVecKeys = &remain_Keys;

    sym::last_marg_para_blocks = last_marginalization_parameter_blocks;
}

inline sym::optimizer_params_t OptimizerParams() {
  sym::optimizer_params_t params{};
  params.iterations = 50;
  params.verbose = true;
  params.initial_lambda = 1.0;
  params.lambda_update_type = sym::lambda_update_type_t::STATIC;
  params.lambda_up_factor = 10.0;
  params.lambda_down_factor = 1 / 10.0;
  params.lambda_lower_bound = 1.0e-8;
  params.lambda_upper_bound = 1000000.0;
  params.early_exit_min_reduction = 1.0e-6;
  params.use_unit_damping = true;
  params.use_diagonal_damping = false;
  params.keep_max_diagonal_damping = false;
  params.diagonal_damping_min = 1e-6;
  params.enable_bold_updates = false;
  return params;
}

inline sym::optimizer_params_t RobotLocalizationOptimizerParams() {
  sym::optimizer_params_t params = sym::DefaultOptimizerParams();
  // 迭代次数少了，就是不行
  params.iterations = 50;//8;
  // 似乎1.0e-8、 1.0e6是最好的参数，试验过很多次，改成其它的都未必能跑好
  params.lambda_lower_bound = 1.0e-8;//1.0e-16;
  params.lambda_upper_bound = 1.0e6;//1.0e32;
//   params.verbose = true;
  params.initial_lambda = 1;//0;//1e4;
//   params.lambda_down_factor = 1 / 2.;
  params.lambda_up_factor = 10.0;
  params.lambda_down_factor = 1 / 10.0;
  params.use_unit_damping = true;
  params.use_diagonal_damping = true;
  params.keep_max_diagonal_damping = true;
  return params;
}

void Estimator::symOptimization() // use symforce to optimize.
{
    constexpr double epsilon = 1e-10;
    constexpr double reprojection_error_gnc_scale = 10;//1.0;//10;

    // 创建Values和创建Factor的先后顺序可以调换
    // build values.
    sym::Valuesd values;
    // sym::Values<double> values;

    // 以下两个参数用于类BarronNoiseModel的，噪声模型相关的两个量
    // 尺度参数 The scale parameter
    values.Set({sym::Var::GNC_SCALE}, reprojection_error_gnc_scale); 
    // μ凸性参数The mu convexity parameter 范围0->1, 用于计算参数alpha
    values.Set(sym::Var::GNC_MU, 0.0); // huber loss ?
    values.Set(sym::Var::GNC_MU, 0.5); // Cauchy loss ?
    values.Set({sym::Var::MATCH_WEIGHT, 1, 1}, 1.0); // 点有效权重为1，否则为0

    // poses: p q v ba bg
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        values.Set({'P', i}, Ps[i]);
        values.Set({'Q', i}, sym::Rot3<double>::FromRotationMatrix(Rs[i]));
        values.Set({'V', i}, Vs[i]);
        values.Set({'A', i}, Bas[i]);
        values.Set({'G', i}, Bgs[i]);
    }

    // extrinsic parameters: pbc qbc
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        values.Set({'t', i}, tic[i]);
        values.Set({'r', i}, sym::Rot3<double>::FromRotationMatrix(ric[i]));
    }

    // landmark's inverse depth
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
    {
        values.Set({'L', i}, dep(i));
    }

    // td
    if (ESTIMATE_TD)
        values.Set({'d'}, td);

    // epsilon
    values.Set('e', sym::kDefaultEpsilond);

    // preintegration
    values.Set('g', G); // gravity
    
    //imu预积分是从第1帧开始，到第10帧结束，总共10帧
    for (int i = 1; i <= WINDOW_SIZE; i++)
    {
        // delta_p
        values.Set({'p', i}, pre_integrations[i]->delta_p);
        // delta_q
        // TODO: 测试发现似乎并没有sym::Quaternion<Scalar>这个类，还是用sym::Rot3吧
        // values.Set({'q', i}, sym::Rot3<double>::FromRotationMatrix(pre_integrations[i]->delta_q));
        values.Set({'q', i}, sym::Rot3<double>::FromQuaternion(pre_integrations[i]->delta_q));
        // delta_v
        values.Set({'v', i}, pre_integrations[i]->delta_v);
        // sum_dt
        values.Set({'s', i}, pre_integrations[i]->sum_dt);
        // dp_dba
        values.Set({'i', i, 1}, pre_integrations[i]->jacobian.block<3, 3>(O_P, O_BA));
        // dp_dbg
        values.Set({'i', i, 2}, pre_integrations[i]->jacobian.block<3, 3>(O_P, O_BG));
        // dq_dbg
        values.Set({'i', i, 3}, pre_integrations[i]->jacobian.block<3, 3>(O_R, O_BG));
        // dv_dba
        values.Set({'i', i, 4}, pre_integrations[i]->jacobian.block<3, 3>(O_V, O_BA));
        // dv_dbg
        values.Set({'i', i, 5}, pre_integrations[i]->jacobian.block<3, 3>(O_V, O_BG));
        // linearized_ba
        values.Set({'i', i, 6}, pre_integrations[i]->linearized_ba);
        // linearized_bg
        values.Set({'i', i, 7}, pre_integrations[i]->linearized_bg);

        // sqrt_info
        Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integrations[i]->covariance.inverse()).matrixL().transpose();
        values.Set({'i', i, 8}, sqrt_info);
    }

    // set feature values
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // 进行特征点有效性的检查
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
 
        // ++feature_index;

        // 第一个观测到这个特征点的帧idx
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        // 特征点在第一个帧下的归一化相机系坐标
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        // 以'f' + 特征点序号 + 特征点所属帧序号作为key.
        values.Set({'f', it_per_id.feature_id, imu_i}, pts_i);

        // 遍历看到这个特征点的所有KF
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j) // 自己跟自己不能形成重投影
            {
                continue;
            }
            // 取出另一帧的归一化相机坐标
            Vector3d pts_j = it_per_frame.point;

            values.Set({'f', it_per_id.feature_id, imu_j}, pts_j);
        }
      
    }

    // set relocation values
    if(relocalization_info)
    {
        int feature_index = 0;
        for(auto match_point : match_points)
        {
            values.Set({'f', feature_index}, Vector3d(match_point.x(), match_point.y(), 1.0));
            feature_index++;
        }

        // relo_Pose
        values.Set({'P', 1, 1}, Vector3d(relo_Pose[0], relo_Pose[1], relo_Pose[2]));
        values.Set({'Q', 1, 1}, sym::Rot3<double>(Eigen::Quaternion<double>(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5])));
    }


    // build factors.
    std::vector<sym::Factord> factors;
    // std::vector<sym::Factor<double>> factors;

    // use last marginalization result to create a marg factor.

    std::cout << "last_marginalization_info addr = " << reinterpret_cast<long>(last_marginalization_info) << std::endl;

    // if (last_marginalization_info)
    if (0 && last_marginalization_info) // close marginalization
    {
        /*
        factors.push_back(sym::Factor<double>::Hessian(
            sym::MargFactor<double, RESIDUAL_DIM1>,
            // input arguments
            {
                {'P', 0}, {'Q', 0},
                {'P', 1}, {'Q', 1},
                {'P', 2}, {'Q', 2},
                {'P', 3}, {'Q', 3},
                {'P', 4}, {'Q', 4},
                {'P', 5}, {'Q', 5},
                {'P', 6}, {'Q', 6},
                {'P', 7}, {'Q', 7},
                {'P', 8}, {'Q', 8},
                {'P', 9}, {'Q', 9},
                {'t', 0}, {'r', 0},
                {'V', 0}, {'A', 0}, {'G', 0}
            },
            // keys to optimize
            {}));
        */

        
        if(last_marginalization_flag == MARGIN_OLD)
        {         
            const std::vector<sym::Key> factor_keys = {
                {'P', 0}, {'Q', 0},
                {'P', 1}, {'Q', 1},
                {'P', 2}, {'Q', 2},
                {'P', 3}, {'Q', 3},
                {'P', 4}, {'Q', 4},
                {'P', 5}, {'Q', 5},
                {'P', 6}, {'Q', 6},
                {'P', 7}, {'Q', 7},
                {'P', 8}, {'Q', 8},
                {'P', 9}, {'Q', 9},
                {'t', 0}, {'r', 0},
                {'V', 0}, {'A', 0}, {'G', 0}};
/*
            const std::vector<sym::Key> optimized_keys = {
                {'P', 0}, {'Q', 0},
                {'P', 1}, {'Q', 1},
                {'P', 2}, {'Q', 2},
                {'P', 3}, {'Q', 3},
                {'P', 4}, {'Q', 4},
                {'P', 5}, {'Q', 5},
                {'P', 6}, {'Q', 6},
                {'P', 7}, {'Q', 7},
                {'P', 8}, {'Q', 8},
                {'P', 9}, {'Q', 9},
                {'t', 0}, {'r', 0},
                {'V', 0}, {'A', 0}, {'G', 0}};
            */

            std::vector<sym::Key> optimized_keys;
#if 1            
            for(int i = 0; i <= 9; i++)
            {
                optimized_keys.push_back({'P', i});
                optimized_keys.push_back({'Q', i});
            }

            // if ESTIMATE_EXTRINSIC !=0 optimize extrinsic parameters
            // otherwise it's another way to fix them.(set constant)
            // if (ESTIMATE_EXTRINSIC)
            {
                optimized_keys.push_back({'t', 0});
                optimized_keys.push_back({'r', 0});
            }

            optimized_keys.push_back({'V', 0});
            optimized_keys.push_back({'A', 0});
            optimized_keys.push_back({'G', 0});
#else           
            for(auto key : remain_Keys)
            {
                switch (key)
                {
                case sym::POSE0:
                    {
                        optimized_keys.push_back({'P', 0});
                        optimized_keys.push_back({'Q', 0});
                    }        
                    break;

                case sym::EParameterType::POSE1:
                    {
                        optimized_keys.push_back({'P', 1});
                        optimized_keys.push_back({'Q', 1});
                    }        
                    break;

                case sym::POSE2:
                    {
                        optimized_keys.push_back({'P', 2});
                        optimized_keys.push_back({'Q', 2});
                    }        
                    break;        
                
                case sym::POSE3:
                    {
                        optimized_keys.push_back({'P', 3});
                        optimized_keys.push_back({'Q', 3});
                    }        
                    break;

                case sym::POSE4:
                    {
                        optimized_keys.push_back({'P', 4});
                        optimized_keys.push_back({'Q', 4});
                    }        
                    break;

                case sym::POSE5:
                    {
                        optimized_keys.push_back({'P', 5});
                        optimized_keys.push_back({'Q', 5});
                    }        
                    break;  

                case sym::POSE6:
                    {
                        optimized_keys.push_back({'P', 6});
                        optimized_keys.push_back({'Q', 6});
                    }        
                    break;

                case sym::POSE7:
                    {
                        optimized_keys.push_back({'P', 7});
                        optimized_keys.push_back({'Q', 7});
                    }        
                    break;

                case sym::POSE8:
                    {
                        optimized_keys.push_back({'P', 8});
                        optimized_keys.push_back({'Q', 8});
                    }        
                    break;  

                case sym::POSE9:
                    {
                        optimized_keys.push_back({'P', 9});
                        optimized_keys.push_back({'Q', 9});
                    }        
                    break;

                case sym::EX_POSE:
                    {
                        optimized_keys.push_back({'t', 0});
                        optimized_keys.push_back({'r', 0});
                    }        
                    break;

                case sym::VBABG0:
                    {
                        optimized_keys.push_back({'V', 0});
                        optimized_keys.push_back({'A', 0});
                        optimized_keys.push_back({'G', 0});
                    }        
                    break;  

                default:
                    break;
                }
            }
#endif
            factors.push_back(sym::Factor<double>::Hessian(
                // sym::MargFactor<double, sym::RESIDUAL_DIM1>, factor_keys,
                sym::MargOldFactor<double>, factor_keys,
                optimized_keys));
        }
        else if(last_marginalization_flag == MARGIN_SECOND_NEW)
        {
            const std::vector<sym::Key> factor_keys = {
                {'P', 0}, {'Q', 0},
                {'P', 1}, {'Q', 1},
                {'P', 2}, {'Q', 2},
                {'P', 3}, {'Q', 3},
                {'P', 4}, {'Q', 4},
                {'P', 5}, {'Q', 5},
                {'P', 6}, {'Q', 6},
                {'P', 7}, {'Q', 7},
                {'P', 8}, {'Q', 8},
                // {'P', 9}, {'Q', 9},
                {'t', 0}, {'r', 0},
                {'V', 0}, {'A', 0}, {'G', 0}};

            std::vector<sym::Key> optimized_keys;
#if 1
            for(int i = 0; i <= 8; i++)
            {
                optimized_keys.push_back({'P', i});
                optimized_keys.push_back({'Q', i});
            }

            // if ESTIMATE_EXTRINSIC !=0 optimize extrinsic parameters
            // otherwise it's another way to fix them.(set constant)
            // if (ESTIMATE_EXTRINSIC)
            {
                optimized_keys.push_back({'t', 0});
                optimized_keys.push_back({'r', 0});
            }

            optimized_keys.push_back({'V', 0});
            optimized_keys.push_back({'A', 0});
            optimized_keys.push_back({'G', 0});
#else
            for(auto key : remain_Keys)
            {
                switch (key)
                {
                case sym::POSE0:
                    {
                        optimized_keys.push_back({'P', 0});
                        optimized_keys.push_back({'Q', 0});
                    }        
                    break;

                case sym::POSE1:
                    {
                        optimized_keys.push_back({'P', 1});
                        optimized_keys.push_back({'Q', 1});
                    }        
                    break;

                case sym::POSE2:
                    {
                        optimized_keys.push_back({'P', 2});
                        optimized_keys.push_back({'Q', 2});
                    }        
                    break;        
                
                case sym::POSE3:
                    {
                        optimized_keys.push_back({'P', 3});
                        optimized_keys.push_back({'Q', 3});
                    }        
                    break;

                case sym::POSE4:
                    {
                        optimized_keys.push_back({'P', 4});
                        optimized_keys.push_back({'Q', 4});
                    }        
                    break;

                case sym::POSE5:
                    {
                        optimized_keys.push_back({'P', 5});
                        optimized_keys.push_back({'Q', 5});
                    }        
                    break;  

                case sym::POSE6:
                    {
                        optimized_keys.push_back({'P', 6});
                        optimized_keys.push_back({'Q', 6});
                    }        
                    break;

                case sym::POSE7:
                    {
                        optimized_keys.push_back({'P', 7});
                        optimized_keys.push_back({'Q', 7});
                    }        
                    break;

                case sym::POSE8:
                    {
                        optimized_keys.push_back({'P', 8});
                        optimized_keys.push_back({'Q', 8});
                    }        
                    break;  

                case sym::POSE9:
                    {
                        // optimized_keys.push_back({'P', 9});
                        // optimized_keys.push_back({'Q', 9});
                    }        
                    break;

                case sym::EX_POSE:
                    {
                        optimized_keys.push_back({'t', 0});
                        optimized_keys.push_back({'r', 0});
                    }        
                    break;

                case sym::VBABG0:
                    {
                        optimized_keys.push_back({'V', 0});
                        optimized_keys.push_back({'A', 0});
                        optimized_keys.push_back({'G', 0});
                    }        
                    break;  

                default:
                    break;
                } // switch(key)
            }
#endif
            factors.push_back(sym::Factor<double>::Hessian(
                // sym::MargFactor<double, sym::RESIDUAL_DIM2>, factor_keys,
                sym::MargNewFactor<double>, factor_keys,
                optimized_keys));
        }
    
    } // if (last_marginalization_info)

    // imu factor
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        // 时间过长这个约束就不可信了
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;

        std::vector<sym::Key> factor_keys;
        std::vector<sym::Key> optimized_keys;
        
        // frame i's PQVBABG
        factor_keys.push_back({'P', i});
        factor_keys.push_back({'Q', i});
        factor_keys.push_back({'V', i});
        factor_keys.push_back({'A', i});
        factor_keys.push_back({'G', i});

        // frame j's PQVBABG
        factor_keys.push_back({'P', j});
        factor_keys.push_back({'Q', j});
        factor_keys.push_back({'V', j});
        factor_keys.push_back({'A', j});
        factor_keys.push_back({'G', j});

        // preintegration: delta_p, delta_q, delta_v
        factor_keys.push_back({'p', j});
        factor_keys.push_back({'q', j});
        factor_keys.push_back({'v', j});

        // gravity
        factor_keys.push_back('g');

        // sum_dt
        factor_keys.push_back({'s', j});

        // dp_dba etc.
        factor_keys.push_back({'i', j, 1});
        factor_keys.push_back({'i', j, 2});
        factor_keys.push_back({'i', j, 3});
        factor_keys.push_back({'i', j, 4});
        factor_keys.push_back({'i', j, 5});
        factor_keys.push_back({'i', j, 6});
        factor_keys.push_back({'i', j, 7});
        factor_keys.push_back({'i', j, 8});

        // frame i's PQVBABG
        optimized_keys.push_back({'P', i});
        optimized_keys.push_back({'Q', i});
        optimized_keys.push_back({'V', i});
        optimized_keys.push_back({'A', i});
        optimized_keys.push_back({'G', i});

        // frame j's PQVBABG
        optimized_keys.push_back({'P', j});
        optimized_keys.push_back({'Q', j});
        optimized_keys.push_back({'V', j});
        optimized_keys.push_back({'A', j});
        optimized_keys.push_back({'G', j});

        factors.push_back(sym::Factor<double>::Hessian(
            sym::ImuFactor<double>, factor_keys,
            optimized_keys));
    }

    // projection factor
    // int f_m_cnt = 0;
    int feature_index = -1;
    // 视觉重投影的约束
    // 遍历每一个特征点
    // it_per_id 是FeaturePerId类型
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // 进行特征点有效性的检查
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
 
        ++feature_index;

        // 第一个观测到这个特征点的帧idx
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        // 特征点在第一个帧下的归一化相机系坐标
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        // 遍历看到这个特征点的所有KF
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j) // 自己跟自己不能形成重投影
            {
                continue;
            }
            // 取出另一帧的归一化相机坐标
            Vector3d pts_j = it_per_frame.point;
            // 带有时间延时的是另一种形式
            if (ESTIMATE_TD)
            {
                // TODO: if necessary, realize it.
                /* 
                    ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                     it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                     it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                */
            }
            else
            {
                std::vector<sym::Key> factor_keys;
                std::vector<sym::Key> optimized_keys;
                factor_keys.push_back({'f', it_per_id.feature_id, imu_i});
                factor_keys.push_back({'f', it_per_id.feature_id, imu_j});
                factor_keys.push_back({'P', imu_i});
                factor_keys.push_back({'Q', imu_i});
                factor_keys.push_back({'P', imu_j});
                factor_keys.push_back({'Q', imu_j});
                factor_keys.push_back({'t', 0});
                factor_keys.push_back({'r', 0});
                factor_keys.push_back({'L', feature_index});
                
                // factor_keys.push_back({sym::Var::MATCH_WEIGHT, 1, 1});
                // factor_keys.push_back(sym::Var::GNC_MU);
                // factor_keys.push_back(sym::Var::GNC_SCALE);
                // factor_keys.push_back(sym::Var::EPSILON);

                optimized_keys.push_back({'P', imu_i});
                optimized_keys.push_back({'Q', imu_i});
                optimized_keys.push_back({'P', imu_j});
                optimized_keys.push_back({'Q', imu_j});
                // if (ESTIMATE_EXTRINSIC)
                {
                    optimized_keys.push_back({'t', 0});
                    optimized_keys.push_back({'r', 0});
                }
                optimized_keys.push_back({'L', feature_index});

                factors.push_back(sym::Factor<double>::Hessian(
                    sym::ProjectionFactor<double>, factor_keys,
                    // sym::ProjectionGncFactor<double>, factor_keys,
                    optimized_keys));

            }
            // f_m_cnt++;
        }
    } // for (auto &it_per_id : f_manager.feature)

    // 回环检测相关的约束
    if(relocalization_info)
    {
        //printf("set relocalization factor! \n");
        
        int retrive_feature_index = 0;
        int feature_index = -1;
        // 遍历现有地图点
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            ++feature_index;
            int start = it_per_id.start_frame;
            if(start <= relo_frame_local_index) // 这个地图点能被对应的当前帧看到
            {   
                // 寻找回环帧能看到的地图点
                while((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)
                {
                    retrive_feature_index++;
                }
                // 这个地图点也能被回环帧看到
                if((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    std::vector<sym::Key> factor_keys;
                    std::vector<sym::Key> optimized_keys;
                    factor_keys.push_back({'f', it_per_id.feature_id, it_per_id.start_frame});
                    factor_keys.push_back({'f', retrive_feature_index});
                    factor_keys.push_back({'P', start});
                    factor_keys.push_back({'Q', start});
                    factor_keys.push_back({'P', 1, 1}); // relo_Pose
                    factor_keys.push_back({'Q', 1, 1}); // relo_Pose
                    factor_keys.push_back({'t', 0});
                    factor_keys.push_back({'r', 0});
                    factor_keys.push_back({'L', feature_index});
                    
                    // factor_keys.push_back({sym::Var::MATCH_WEIGHT, 1, 1});
                    // factor_keys.push_back(sym::Var::GNC_MU);
                    // factor_keys.push_back(sym::Var::GNC_SCALE);
                    // factor_keys.push_back(sym::Var::EPSILON);

                    optimized_keys.push_back({'P', start});
                    optimized_keys.push_back({'Q', start});
                    optimized_keys.push_back({'P', 1, 1});
                    optimized_keys.push_back({'Q', 1, 1});
                    // if (ESTIMATE_EXTRINSIC)
                    {
                        optimized_keys.push_back({'t', 0});
                        optimized_keys.push_back({'r', 0});
                    }
                    optimized_keys.push_back({'L', feature_index});

                    factors.push_back(sym::Factor<double>::Hessian(
                        sym::ProjectionFactor<double>, factor_keys,
                        // sym::ProjectionGncFactor<double>, factor_keys,
                        optimized_keys));

                    retrive_feature_index++;
                }     
            }
        }

    }


    // Create and set up Optimizer
   
    // auto params = sym::DefaultOptimizerParams();
    auto params = OptimizerParams();
    params.iterations = 50;
    params.verbose = false;//true;
    /* 
     * work well
    params.use_diagonal_damping = true;
    // params.use_unit_damping = false;
    params.use_unit_damping = true;
    params.keep_max_diagonal_damping = true;
    // params.lambda_update_type = sym::lambda_update_type_t::DYNAMIC;
    */

    // 2024-7-15
/*    params.iterations = 50;
    params.initial_lambda = 1.0e0;
    // params.lambda_update_type = sym::lambda_update_type_t::DYNAMIC;
    params.lambda_up_factor = 4.0;
    params.lambda_down_factor = 0.05;
    params.lambda_lower_bound = 1.0e-8;
    params.lambda_upper_bound = 1.0e6;
*/ 
    // 以下三行必须为true
    params.use_diagonal_damping = true;
    params.use_unit_damping = true;
    params.keep_max_diagonal_damping = true;

    /*
     *  lambda DYNAMIC updating is inapplicable to diagonal damping
    // auto params2 = sym::DefaultOptimizerParams();
    params.lambda_update_type = sym::lambda_update_type_t::DYNAMIC;
    params.initial_lambda = 1.0;
    params.lambda_lower_bound = 1.0e-8;
    params.lambda_upper_bound = 1.0e6;
    */
    // the end.

    /*
     * 1、Ceres迭代策略改为LM的情况下，不管是dense schur和dense cholesky分解，轨迹均发散；
     * 2、Ceres迭代策略改为LM的情况下，使用sparse cholesky分解，开始会报dense矩阵分解失败，但仍能完成整个轨迹；
     * 3、symforce 只有LM策略，选择sparse cholesky分解，很容易轨迹发散，在迭代次数用完的情况下偶尔能跑成功一次，选择dense cholesky，大部分情况下在迭代次数用完时得到一个满意的精度的轨迹；
     * 4、symforce 将lambda设为0，程序会报错Internal Error: Damped hessian factorization failed.
    */

    
    // 对于symforce又总结出一些心得。其适合于求解问题规模比较小（优化维数小）或者稀疏的Hessian矩阵，
    // 作者举例均是要么问题的维数较少，要么路标点个数远远超过相机个数（维数），
    // 即当路标点的个数远远超过位姿个数的情况下，其能表现优异，一旦位姿个数较多，维数较大的情况，其不能良好工作。经常把迭代次数用完HIT_ITERATION_LIMIT
    // 论文中对于稠密矩阵的比较，仅仅是矩阵相乘的测试dense matrix multiplication，
    // 比较了矩阵相乘符号化之后Flattend和Eigen的稠密和稀疏的矩阵乘法消耗的时间
    // SymForce与Eigen比较稀疏和密集矩阵乘法
    // 因此，推断作者可能没有想要实现稠密矩阵求解的方式，因为根本不能提高效率。
    // 另外又发现：
    // 当 params.initial_lambda = 1.0; params.lambda_lower_bound = 1.0e-8; params.lambda_upper_bound = 1.0e6;
    // 并且边缘化不开的情况下：sym::Optimizer<double>和 DenseOptimizer<double> 均在HIT_ITERATION_LIMIT时得到较好的精度

    // 2024-7-22再补充: 并不是LM法不能用，只能说dog leg迭代方法选择步长更合理，Dog-Leg略优于LM。迭代只是为了构建一个新的H矩阵，最终求解，就需要用到各种分解，比如 Schur, LDLT, Cholesky(LLT)分解等
    // 总之，symforce的LM迭代，和SparseCholesky或者DenseCholesky分解对于求解后是否能够收敛，就没有那么好的效果

    // sym::Optimizer<double> optimizer(params, factors);
    sym::Optimizer<double> optimizer(RobotLocalizationOptimizerParams(), factors);
    // SparseSchurOptimizer<double> optimizer(RobotLocalizationOptimizerParams(), factors); // failed.
    // DenseOptimizer<double> optimizer(params, factors); // dense optimizer better. 2024-7-15.
    // DenseOptimizer<double> optimizer(params2, factors); // dense optimizer better. 2024-7-15.
    // DenseOptimizer<double> optimizer(RobotLocalizationOptimizerParams(), factors); // dense optimizer better. 2024-7-15.
    /*
    sym::Optimizerd optimizer(optimizer_params, factors, "BundleAdjustmentOptimizer", optimized_keys,
                            params.epsilon);

    sym::Optimizerd optimizer(optimizer_params, {BuildFactor()}, "BundleAdjustmentOptimizer", {},
                            params.epsilon);
    */                        

    // Optimize
    const auto stats = optimizer.Optimize(values);
    // const sym::Optimizerd::Stats stats = optimizer.Optimize(values);

    const auto& iteration_stats = stats.iterations;
  const auto& first_iter = iteration_stats.front();
  const auto& last_iter = iteration_stats.back();

  // Note that the best iteration (with the best error, and representing the Values that gives
  // that error) may not be the last iteration, if later steps did not successfully decrease the
  // cost
  const auto& best_iter = iteration_stats[stats.best_index];

  spdlog::info("Iterations: {}", last_iter.iteration);
  spdlog::info("Lambda: {}", last_iter.current_lambda);
  spdlog::info("Initial error: {}", first_iter.new_error);
  spdlog::info("Final error: {}", best_iter.new_error);
  spdlog::info("Status: {}", stats.status);


    // update states
    // TODO:
    // if(stats.status == sym::optimization_status_t::SUCCESS)
    {
        Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
        Vector3d origin_P0 = Ps[0];

        if (failure_occur)
        {
            origin_R0 = Utility::R2ypr(last_R0);
            origin_P0 = last_P0;
            failure_occur = 0;
        }
        sym::Rot3d Q0= values.At<sym::Rot3d>({'Q', 0});
        Vector3d origin_R00 = Utility::R2ypr(Q0.ToRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();

        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            rot_diff = Rs[0] * Q0.ToRotationMatrix().transpose();
        }

        Vector3d P0 = values.At<Eigen::Vector3d>({'P', 0});
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            sym::Rot3d Qi= values.At<sym::Rot3d>({'Q', i});
            Rs[i] = rot_diff * Qi.Quaternion().normalized().toRotationMatrix();
            
            Vector3d Pi = values.At<Eigen::Vector3d>({'P', i});
            Ps[i] = rot_diff * (Pi - P0) + origin_P0;

            Vs[i] = rot_diff * values.At<Eigen::Vector3d>({'V', i});

            Bas[i] = values.At<Eigen::Vector3d>({'A', i});

            Bgs[i] = values.At<Eigen::Vector3d>({'G', i});
        }

        if (ESTIMATE_EXTRINSIC)
        {
            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                tic[i] = values.At<Eigen::Vector3d>({'t', i});
                ric[i] = values.At<sym::Rot3d>({'r', i}).ToRotationMatrix();
            }
        }

        // VectorXd dep = f_manager.getDepthVector();
        for (int i = 0; i < f_manager.getFeatureCount(); i++)
            dep(i) = values.At<double>({'L', i});
        f_manager.setDepth(dep);
        if (ESTIMATE_TD)
            td =  values.At<double>('d');

        // relative info between two loop frame
        if(relocalization_info)
        { 
            Matrix3d relo_r;
            Vector3d relo_t;
            // relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
            relo_r = rot_diff * values.At<sym::Rot3d>({'Q', 1, 1}).Quaternion().normalized().toRotationMatrix();
            relo_t = rot_diff * (values.At<Eigen::Vector3d>({'P', 1, 1}) - P0) + origin_P0;
            double drift_correct_yaw;
            drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
            drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
            drift_correct_t = prev_relo_t - drift_correct_r * relo_t;   
            relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
            relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
            relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
            //cout << "vins relo " << endl;
            //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
            //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
            relocalization_info = 0;

        }
    }

    // marginalization
    // TODO:
    computeMarginalizationResult();
    // TODO: should use sym::factor & sym::Linearizer to construct marginalization result.

    last_marginalization_flag = marginalization_flag;
}

void Estimator::computeMarginalizationResult()
{
    ceres::LossFunction *loss_function;
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);

    // // 把优化后double -> eigen
    // double2vector();

    // 把优化后eigen -> double
    vector2double();

    // Step 4 边缘化
    // 科普一下舒尔补
    TicToc t_whole_marginalization;
    // 如果次新帧是关键帧，将边缘化最老帧，及其看到的路标点和IMU数据，将其转化为先验：
    if (marginalization_flag == MARGIN_OLD)
    {
        int marg_feature_count = 0; // added on 2024-7-5
        // 一个用来边缘化操作的对象
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        // 这里类似手写高斯牛顿，因此也需要都转成double数组
        vector2double();
        // 关于边缘化有几点注意的地方
        // 1、找到需要边缘化的参数块，这里是地图点，第0帧位姿，第0帧速度零偏
        // 2、找到构造高斯牛顿下降时跟这些待边缘化相关的参数块有关的残差约束，那就是预积分约束，重投影约束，以及上一次边缘化约束
        // 3、这些约束连接的参数块中，不需要被边缘化的参数块，就是被提供先验约束的部分，也就是滑窗中剩下的位姿和速度零偏

        //1、将上一次先验残差项传递给marginalization_info
        // 上一次的边缘化结果
        if (last_marginalization_info)
        {
            vector<int> drop_set;
            // last_marginalization_parameter_blocks是上一次边缘化对哪些当前参数块有约束
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                // 涉及到的待边缘化的上一次边缘化留下来的当前参数块只有位姿和速度零偏
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // 处理方式和其他残差块相同
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        //2、将第0帧和第1帧间的IMU因子IMUFactor(pre_integrations[1])，添加到marginalization_info中
        // 只有第1个预积分和待边缘化参数块相连
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                // 跟构建ceres约束问题一样，这里也需要得到残差和雅克比
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1}); // 这里就是第0和1个参数块是需要被边缘化的
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        //3、将第一次观测为第0帧的所有路标点对应的视觉观测，添加到marginalization_info中
        // 遍历视觉重投影的约束
        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                // 只找能被第0帧看到的特征点
                if (imu_i != 0)
                    continue;

                marg_feature_count++;
                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                // 遍历看到这个特征点的所有KF，通过这个特征点，建立和第0帧的约束
                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.point;
                    // 根据是否约束延时确定残差阵
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        // 重点说一下vector<int>{0, 3}: 表示要被drop或者marg掉的参数序号有0和3，对应到的参数块即为para_Pose[imu_i]和para_Feature[feature_index]，即第0帧的位姿和路标点需要被marg
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3}); // 这里第0帧和地图点被margin
                        marginalization_info->addResidualBlockInfo(residual_block_info);                        
                    }
                }
            }
        }
        std::cout << "marg old: marg_feature_count=" << marg_feature_count << std::endl;

        // 所有的残差块都收集好了
        TicToc t_pre_margin;
        //4、计算每个残差，对应的Jacobian，并将各参数块拷贝到统一的内存（parameter_block_data）中
        // 进行预处理
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        //5、多线程构造先验项舒尔补AX=b的结构，在X0处线性化计算Jacobian和残差
        // 边缘化操作
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        //6.调整参数块在下一次窗口中对应的位置（往前移一格），注意这里是指针，后面slideWindow中会赋新值，这里只是提前占座
        // 即将滑窗，因此记录新地址对应的老地址
        // 关于addr_shift补充说明一点：其作用是为了记录边缘化之后保留的参数块，在滑窗中新的位置。
        // 比如marg老帧时，原来滑窗第1帧的位姿其位置就移动到第0帧（往前移一格），以此类推
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            // 位姿和速度都要滑窗移动
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        // 外参和时间延时不变
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        // parameter_blocks实际上就是addr_shift的索引的集合及搬进去的新地址
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info; // 本次边缘化的所有信息
        last_marginalization_parameter_blocks = parameter_blocks; // 代表该次边缘化对某些参数块形成约束，这些参数块在滑窗之后的地址
        setRemainParameterKey(); // 2024-7-9
        
    }
    else // 如果次新帧不是关键帧：// 边缘化倒数第二帧
    {
        // 要求有上一次边缘化的结果，同时即将被margin掉的（para_Pose[WINDOW_SIZE - 1]）在上一次边缘化后的约束中
        // 预积分结果合并，因此只有位姿margin掉
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1])) // 统计para_Pose[WINDOW_SIZE - 1]在上一次边缘化的参数块中出现的次数
        {

            //1.保留次新帧的IMU测量，丢弃该帧的视觉测量，将上一次先验残差项传递给marginalization_info
            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    // 速度零偏只会margin第1个（并且只有在marg老帧的时候，才会marg第一个速度零偏），不可能出现倒数第二个
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    // 这种case只会margin掉倒数第二个位姿
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                // 这里只会更新一下margin factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            // 这里的操作如出一辙
            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            //2、premargin
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            //3、marginalize
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            //4.调整参数块在下一次窗口中对应的位置（去掉次新帧）
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE) // 滑窗，最新帧成为次新帧
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else // 其他不变
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }
            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            setRemainParameterKey(); // 2024-7-9
            
        }
    }
}