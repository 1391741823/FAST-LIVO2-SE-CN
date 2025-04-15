/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "IMU_Processing.h"

const bool time_list(PointType &x, PointType &y) { return (x.curvature < y.curvature); }//PointType：点云数据类型，包含曲率信息（curvature）

ImuProcess::ImuProcess() : Eye3d(M3D::Identity()),
                           Zero3d(0, 0, 0), b_first_frame(true), imu_need_init(true)//b_first_frame为第一帧
{
  init_iter_num = 1;
  cov_acc = V3D(0.1, 0.1, 0.1);
  cov_gyr = V3D(0.1, 0.1, 0.1);//加速度和角速度的协方差
  cov_bias_gyr = V3D(0.1, 0.1, 0.1);//M3D：三维矩阵，用于表示旋转矩阵
  cov_bias_acc = V3D(0.1, 0.1, 0.1);//V3D：三维向量，用于表示加速度、角速度等
  cov_inv_expo = 0.2;
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);//加速度和角速度的均值
  angvel_last = Zero3d;
  acc_s_last = Zero3d;
  Lid_offset_to_IMU = Zero3d;
  Lid_rot_to_IMU = Eye3d;//LiDAR到IMU的平移和旋转
  last_imu.reset(new sensor_msgs::Imu());//存储上一帧IMU数据
  cur_pcl_un_.reset(new PointCloudXYZI());//当前未处理的点云数据
}//IMU数据的初始化

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset()
{
  ROS_WARN("Reset ImuProcess");
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = Zero3d;
  imu_need_init = true;
  init_iter_num = 1;
  IMUpose.clear();
  last_imu.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::disable_imu()
{
  cout << "IMU Disabled !!!!!" << endl;
  imu_en = false;//IMU功能开关
  imu_need_init = false;
}

void ImuProcess::disable_gravity_est()
{
  cout << "Online Gravity Estimation Disabled !!!!!" << endl;
  gravity_est_en = false;//重力估计开关
}

void ImuProcess::disable_bias_est()
{
  cout << "Bias Estimation Disabled !!!!!" << endl;
  ba_bg_est_en = false;//偏差估计开关
}

void ImuProcess::disable_exposure_est()
{
  cout << "Online Time Offset Estimation Disabled !!!!!" << endl;
  exposure_estimate_en = false;//时间偏移估计开关
}

void ImuProcess::set_extrinsic(const MD(4, 4) & T)
{
  Lid_offset_to_IMU = T.block<3, 1>(0, 3);
  Lid_rot_to_IMU = T.block<3, 3>(0, 0);
}

void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lid_offset_to_IMU = transl;
  Lid_rot_to_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lid_offset_to_IMU = transl;//雷达到imu的时间补偿
  Lid_rot_to_IMU = rot;//雷达到imu的旋转矩阵
}

void ImuProcess::set_gyr_cov_scale(const V3D &scaler) { cov_gyr = scaler; }//角速度协方差的缩放因子

void ImuProcess::set_acc_cov_scale(const V3D &scaler) { cov_acc = scaler; }//加速度协方差的缩放因子

void ImuProcess::set_gyr_bias_cov(const V3D &b_g) { cov_bias_gyr = b_g; }//角速度偏差的协方差

void ImuProcess::set_inv_expo_cov(const double &inv_expo) { cov_inv_expo = inv_expo; }//逆曝光时间的协方差

void ImuProcess::set_acc_bias_cov(const V3D &b_a) { cov_bias_acc = b_a; }//加速度偏差的协方差

void ImuProcess::set_imu_init_frame_num(const int &num) { MAX_INI_COUNT = num; }//IMU初始化帧数

void ImuProcess::IMU_init(const MeasureGroup &meas, StatesGroup &state_inout, int &N)//函数用于初始化IMU模块，包括重力、陀螺仪偏差、加速度和角速度的协方差
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  ROS_INFO("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);
  V3D cur_acc, cur_gyr;

  if (b_first_frame)
  {
    Reset();
    N = 1;//当前初始化帧数
    b_first_frame = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;//加速度
    const auto &gyr_acc = meas.imu.front()->angular_velocity;//角速度
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    // first_lidar_time = meas.lidar_frame_beg_time;
    // cout<<"init acc norm: "<<mean_acc.norm()<<endl;
  }
  //通过多帧IMU数据计算加速度和角速度的均值，并初始化状态变量（如重力、旋转矩阵和偏差）
  for (const auto &imu : meas.imu)
  {//meas：包含IMU和LiDAR数据的测量组
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc += (cur_acc - mean_acc) / N;
    mean_gyr += (cur_gyr - mean_gyr) / N;//求平均值

    // cov_acc = cov_acc * (N - 1.0) / N + (cur_acc -
    // mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N); cov_gyr
    // = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr -
    // mean_gyr) * (N - 1.0) / (N * N);

    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

    N++;
  }
  IMU_mean_acc_norm = mean_acc.norm();
  state_inout.gravity = -mean_acc / mean_acc.norm() * G_m_s2;//state_inout：状态变量（如重力、旋转矩阵、偏差等）
  state_inout.rot_end = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  state_inout.bias_g = Zero3d; // mean_gyr;

  last_imu = meas.imu.back();//存储当前IMU数据
}
//与lio前向传播
/*
无imu数据时，通过 LiDAR 数据进行状态传播，并将结果存储到 pcl_out 中
*/
void ImuProcess::Forward_without_imu(LidarMeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out)//函数用于在没有IMU数据的情况下，通过LiDAR数据进行状态传播
{
  // const double &pcl_beg_time = meas.lidar_frame_beg_time;//meas：包含LiDAR数据的测量组

  /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);//对点云数据按时间排序（当前代码中排序部分被注释）
  // sort(pcl_out->points.begin(), pcl_out->points.end(), time_list);
  const double &pcl_beg_time = meas.lidar_frame_beg_time;//赋值
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);

  //点云处理的总时间=点云处理开始的时间+计算点云的结束时间
  const double &pcl_end_time = pcl_beg_time + pcl_out.points.back().curvature / double(1000);//计算点云的结束时间
  // V3D acc_imu, angvel_avr, acc_avr, vel_imu(state_inout.vel_end),
  //     pos_imu(state_inout.pos_end);
  // M3D R_imu(state_inout.rot_end);
  meas.last_lio_update_time = pcl_end_time;//lio的时间
  // MD(DIM_STATE, DIM_STATE)
  // F_x, cov_w;
  const double &pcl_end_offset_time = pcl_out.points.back().curvature / double(1000);//计算点云的结束时间
  MD(DIM_STATE, DIM_STATE) F_x, cov_w;//状态

  double dt = 0;//时间间隔计算：根据点云数据的时间戳计算dts

  if (b_first_frame)//是否处理第一帧
  {
    dt = 0.1;
    b_first_frame = false;
  }
  else { dt = pcl_beg_time - time_last_scan; }

  time_last_scan = pcl_beg_time;
  // for (size_t i = 0; i < pcl_out->points.size(); i++) {
  //   if (dt < pcl_out->points[i].curvature) {
  //     dt = pcl_out->points[i].curvature;
  //   }
  // }
  // dt = dt / (double)1000;
  // std::cout << "dt:" << dt << std::endl;
  // double dt = pcl_out->points.back().curvature / double(1000);

  /* covariance propagation */
  // M3D acc_avr_skew;
  M3D Exp_f = Exp(state_inout.bias_g, dt);
  //状态传播：通过状态转移矩阵F_x和噪声协方差矩阵cov_w更新状态和协方差
  F_x.setIdentity();
  cov_w.setZero();

  F_x.block<3, 3>(0, 0) = Exp(state_inout.bias_g, -dt);
  // F_x.block<3, 3>(0, 9) = Eye3d * dt;
  // F_x.block<3, 3>(3, 6) = Eye3d * dt;
  F_x.block<3, 3>(0, 10) = Eye3d * dt;
  F_x.block<3, 3>(3, 7) = Eye3d * dt;
  // F_x.block<3, 3>(6, 0)  = - R_imu * acc_avr_skew * dt;
  // F_x.block<3, 3>(6, 12) = - R_imu * dt;
  // F_x.block<3, 3>(6, 15) = Eye3d * dt;

  // cov_w.block<3, 3>(9, 9).diagonal() = cov_gyr * dt * dt; // for omega in constant model
  // cov_w.block<3, 3>(6, 6).diagonal() = cov_acc * dt * dt; // for velocity in constant model
  cov_w.block<3, 3>(10, 10).diagonal() = cov_gyr * dt * dt; // for omega in constant model
  cov_w.block<3, 3>(7, 7).diagonal() = cov_acc * dt * dt; // for velocity in constant model
  // cov_w.block<3, 3>(6, 6) =
  //     R_imu * cov_acc.asDiagonal() * R_imu.transpose() * dt * dt;
  // cov_w.block<3, 3>(9, 9).diagonal() =
  //     cov_bias_gyr * dt * dt; // bias gyro covariance
  // cov_w.block<3, 3>(12, 12).diagonal() =
  //     cov_bias_acc * dt * dt; // bias acc covariance

  // std::cout << "before propagete:" << state_inout.cov.diagonal().transpose()
  //           << std::endl;
  state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;//状态传播：通过状态转移矩阵F_x和噪声协方差矩阵cov_w更新状态和协方差
  // std::cout << "cov_w:" << cov_w.diagonal().transpose() << std::endl;
  // std::cout << "after propagete:" << state_inout.cov.diagonal().transpose()
  //           << std::endl;
  state_inout.rot_end = state_inout.rot_end * Exp_f;//旋转矩阵更新：使用指数映射更新旋转矩阵
  state_inout.pos_end = state_inout.pos_end + state_inout.vel_end * dt;//位置更新：根据速度和时间间隔更新位置
  /*
  输出的是：
  状态传播：通过状态转移矩阵F_x和噪声协方差矩阵cov_w更新状态和协方差
  旋转矩阵更新：使用指数映射更新旋转矩阵
  位置更新：根据速度和时间间隔更新位置
  */
  if (lidar_type != L515)//雷达类型
  {
    auto it_pcl = pcl_out.points.end() - 1;
    double dt_j = 0.0;
    for(; it_pcl != pcl_out.points.begin(); it_pcl--)
    {
        dt_j= pcl_end_offset_time - it_pcl->curvature/double(1000);
        M3D R_jk(Exp(state_inout.bias_g, - dt_j));
        V3D P_j(it_pcl->x, it_pcl->y, it_pcl->z);
        // Using rotation and translation to un-distort points
        V3D p_jk;
        p_jk = - state_inout.rot_end.transpose() * state_inout.vel_end * dt_j;
        V3D P_compensate =  R_jk * P_j + p_jk;
       /// save Undistorted points and their rotation
        it_pcl->x = P_compensate(0);
        it_pcl->y = P_compensate(1);
        it_pcl->z = P_compensate(2);
    }
  }
}
/*
点云畸变处理
*/
void ImuProcess::UndistortPcl(LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out)
{
  double t0 = omp_get_wtime();
  pcl_out.clear();
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  MeasureGroup &meas = lidar_meas.measures.back();
  // cout<<"meas.imu.size: "<<meas.imu.size()<<endl;
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu);//容器v_imu插入最后一帧imu
  //时间范围计算
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  const double prop_beg_time = last_prop_end_time;
  // printf("[ IMU ] undistort input size: %zu \n", lidar_meas.pcl_proc_cur->points.size());
  // printf("[ IMU ] IMU data sequence size: %zu \n", meas.imu.size());
  // printf("[ IMU ] lidar_scan_index_now: %d \n", lidar_meas.lidar_scan_index_now);

  const double prop_end_time = lidar_meas.lio_vio_flg == LIO ? meas.lio_time : meas.vio_time;

  /*** cut lidar point based on the propagation-start time and required
   * propagation-end time ***/
  // const double pcl_offset_time = (prop_end_time -
  // lidar_meas.lidar_frame_beg_time) * 1000.; // the offset time w.r.t scan
  // start time auto pcl_it = lidar_meas.pcl_proc_cur->points.begin() +
  // lidar_meas.lidar_scan_index_now; auto pcl_it_end =
  // lidar_meas.lidar->points.end(); printf("[ IMU ] pcl_it->curvature: %lf
  // pcl_offset_time: %lf \n", pcl_it->curvature, pcl_offset_time); while
  // (pcl_it != pcl_it_end && pcl_it->curvature <= pcl_offset_time)
  // {
  //   pcl_wait_proc.push_back(*pcl_it);
  //   pcl_it++;
  //   lidar_meas.lidar_scan_index_now++;
  // }

  // cout<<"pcl_out.size(): "<<pcl_out.size()<<endl;
  // cout<<"pcl_offset_time:  "<<pcl_offset_time<<"pcl_it->curvature:
  // "<<pcl_it->curvature<<endl;
  // cout<<"lidar_meas.lidar_scan_index_now:"<<lidar_meas.lidar_scan_index_now<<endl;

  // printf("[ IMU ] last propagation end time: %lf \n", lidar_meas.last_lio_update_time);
  if (lidar_meas.lio_vio_flg == LIO)//lio模式初始化与预处理
  {
    pcl_wait_proc.resize(lidar_meas.pcl_proc_cur->points.size());//调整点云数据大小
    pcl_wait_proc = *(lidar_meas.pcl_proc_cur);//复制点云数据
    lidar_meas.lidar_scan_index_now = 0;//重置扫描索引
    IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, state_inout.vel_end, state_inout.pos_end, state_inout.rot_end));//记录 IMU 位姿
  }

  // printf("[ IMU ] pcl_wait_proc size: %zu \n", pcl_wait_proc.points.size());

  // sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  // lidar_meas.debug_show();
  // cout<<"UndistortPcl [ IMU ]: Process lidar from "<<prop_beg_time<<" to
  // "<<prop_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to
  //          "<<imu_end_time<<endl;
  // cout<<"[ IMU ]: point size: "<<lidar_meas.lidar->points.size()<<endl;

  /*** Initialize IMU pose ***/
  // IMUpose.clear();

  /*** forward propagation at each imu point ***/
  V3D acc_imu(acc_s_last), angvel_avr(angvel_last), acc_avr, vel_imu(state_inout.vel_end), pos_imu(state_inout.pos_end);
  // cout << "[ IMU ] input state: " << state_inout.vel_end.transpose() << " " << state_inout.pos_end.transpose() << endl;
  M3D R_imu(state_inout.rot_end);//前向传播得到的旋转矩阵  //后面会进行输出

  MD(DIM_STATE, DIM_STATE) F_x, cov_w;
  double dt, dt_all = 0.0;
  double offs_t;
  // double imu_time;
  double tau;
  if (!imu_time_init)//如果没有时间的初始化
  {
    // imu_time = v_imu.front()->header.stamp.toSec() - first_lidar_time;
    // tau = 1.0 / (0.25 * sin(2 * CV_PI * 0.5 * imu_time) + 0.75);
    tau = 1.0;
    imu_time_init = true;
  }
  else
  {
    tau = state_inout.inv_expo_time;
    // ROS_ERROR("tau: %.6f !!!!!!", tau);
  }
  // state_inout.cov(6, 6) = 0.01;

  // ROS_ERROR("lidar_meas.lio_vio_flg");
  // cout<<"lidar_meas.lio_vio_flg: "<<lidar_meas.lio_vio_flg<<endl;
  switch (lidar_meas.lio_vio_flg)
  {
  case LIO://开始处理完了

  case VIO://直接开始处理imu->vio
    dt = 0;//时间间隔
    for (int i = 0; i < v_imu.size() - 1; i++)
    {
      auto head = v_imu[i];
      auto tail = v_imu[i + 1];

      if (tail->header.stamp.toSec() < prop_beg_time) continue;//条过无效数据
      //计算角速度和加速度的平均值
      angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x), 0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
          0.5 * (head->angular_velocity.z + tail->angular_velocity.z);

      // angvel_avr<<tail->angular_velocity.x, tail->angular_velocity.y,
      // tail->angular_velocity.z;

      acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x), 0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
          0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

      // cout<<"angvel_avr: "<<angvel_avr.transpose()<<endl;
      // cout<<"acc_avr: "<<acc_avr.transpose()<<endl;

      // #ifdef DEBUG_PRINT
      fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;
      // #endif

      // imu_time = head->header.stamp.toSec() - first_lidar_time;
      //去除偏置
      angvel_avr -= state_inout.bias_g;
      acc_avr = acc_avr * G_m_s2 / mean_acc.norm() - state_inout.bias_a;
      //计算时间间隔
      if (head->header.stamp.toSec() < prop_beg_time)
      {
        // printf("00 \n");
        dt = tail->header.stamp.toSec() - last_prop_end_time;
        offs_t = tail->header.stamp.toSec() - prop_beg_time;
      }
      else if (i != v_imu.size() - 2)
      {
        // printf("11 \n");
        dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
        offs_t = tail->header.stamp.toSec() - prop_beg_time;
      }
      else
      {
        // printf("22 \n");
        dt = prop_end_time - head->header.stamp.toSec();
        offs_t = prop_end_time - prop_beg_time;
      }

      dt_all += dt;//时间间隔
      // printf("[ LIO Propagation ] dt: %lf \n", dt);

      /* covariance propagation */
      //状态传播
      M3D acc_avr_skew;
      M3D Exp_f = Exp(angvel_avr, dt);
      acc_avr_skew << SKEW_SYM_MATRX(acc_avr);

      F_x.setIdentity();//状态转移矩阵
      cov_w.setZero();//噪声协方差矩阵

      F_x.block<3, 3>(0, 0) = Exp(angvel_avr, -dt);
      if (ba_bg_est_en) F_x.block<3, 3>(0, 10) = -Eye3d * dt;
      // F_x.block<3,3>(3,0)  = R_imu * off_vel_skew * dt;
      F_x.block<3, 3>(3, 7) = Eye3d * dt;
      F_x.block<3, 3>(7, 0) = -R_imu * acc_avr_skew * dt;
      if (ba_bg_est_en) F_x.block<3, 3>(7, 13) = -R_imu * dt;
      if (gravity_est_en) F_x.block<3, 3>(7, 16) = Eye3d * dt;

      // tau = 1.0 / (0.25 * sin(2 * CV_PI * 0.5 * imu_time) + 0.75);
      // F_x(6,6) = 0.25 * 2 * CV_PI * 0.5 * cos(2 * CV_PI * 0.5 * imu_time) * (-tau*tau); F_x(18,18) = 0.00001;
      if (exposure_estimate_en) cov_w(6, 6) = cov_inv_expo * dt * dt;
      cov_w.block<3, 3>(0, 0).diagonal() = cov_gyr * dt * dt;
      cov_w.block<3, 3>(7, 7) = R_imu * cov_acc.asDiagonal() * R_imu.transpose() * dt * dt;
      cov_w.block<3, 3>(10, 10).diagonal() = cov_bias_gyr * dt * dt; // bias gyro covariance
      cov_w.block<3, 3>(13, 13).diagonal() = cov_bias_acc * dt * dt; // bias acc covariance

      state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;//更新状态协方差
      // state_inout.cov.block<18,18>(0,0) = F_x.block<18,18>(0,0) *
      // state_inout.cov.block<18,18>(0,0) * F_x.block<18,18>(0,0).transpose() +
      // cov_w.block<18,18>(0,0);

      // tau = tau + 0.25 * 2 * CV_PI * 0.5 * cos(2 * CV_PI * 0.5 * imu_time) *
      // (-tau*tau) * dt;

      // tau = 1.0 / (0.25 * sin(2 * CV_PI * 0.5 * imu_time) + 0.75);
      //更新旋转矩阵
      /* propogation of IMU attitude */
      R_imu = R_imu * Exp_f;
      //更新位置和速度
      /* Specific acceleration (global frame) of IMU */
      acc_imu = R_imu * acc_avr + state_inout.gravity;

      /* propogation of IMU */
      pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

      /* velocity of IMU */
      vel_imu = vel_imu + acc_imu * dt;

      /* save the poses at each IMU measurements */
      angvel_last = angvel_avr;
      acc_s_last = acc_imu;

      // cout<<setw(20)<<"offset_t: "<<offs_t<<"tail->header.stamp.toSec():
      // "<<tail->header.stamp.toSec()<<endl; printf("[ LIO Propagation ]
      // offs_t: %lf \n", offs_t);
      // 保存当前 IMU 位姿
      IMUpose.push_back(set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
    }

    // unbiased_gyr = V3D(IMUpose.back().gyr[0], IMUpose.back().gyr[1], IMUpose.back().gyr[2]);
    // cout<<"prop end - start: "<<prop_end_time - prop_beg_time<<" dt_all: "<<dt_all<<endl;
    lidar_meas.last_lio_update_time = prop_end_time;//lio升级时间
    // dt = prop_end_time - imu_end_time;
    // printf("[ LIO Propagation ] dt: %lf \n", dt);
    break;
  }

  state_inout.vel_end = vel_imu;//IMU 的当前速度，通过 IMU 数据的前向传播计算得到
  state_inout.rot_end = R_imu;// IMU 的当前旋转矩阵，通过 IMU 数据的前向传播计算得到
  state_inout.pos_end = pos_imu;//IMU 的当前位置，通过 IMU 数据的前向传播计算得到
  state_inout.inv_expo_time = tau;//逆曝光时间，通过时间偏移估计或初始化得到

  /*** calculated the pos and attitude prediction at the frame-end ***/
  // if (imu_end_time>prop_beg_time)
  // {
  //   double note = prop_end_time > imu_end_time ? 1.0 : -1.0;
  //   dt = note * (prop_end_time - imu_end_time);
  //   state_inout.vel_end = vel_imu + note * acc_imu * dt;
  //   state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
  //   state_inout.pos_end = pos_imu + note * vel_imu * dt + note * 0.5 *
  //   acc_imu * dt * dt;
  // }
  // else
  // {
  //   double note = prop_end_time > prop_beg_time ? 1.0 : -1.0;
  //   dt = note * (prop_end_time - prop_beg_time);
  //   state_inout.vel_end = vel_imu + note * acc_imu * dt;
  //   state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
  //   state_inout.pos_end = pos_imu + note * vel_imu * dt + note * 0.5 *
  //   acc_imu * dt * dt;
  // }

  // cout<<"[ Propagation ] output state: "<<state_inout.vel_end.transpose() <<
  // state_inout.pos_end.transpose()<<endl;

  last_imu = v_imu.back();//传出imu的数据
  last_prop_end_time = prop_end_time;//传出最后的imu计算十佳年
  double t1 = omp_get_wtime();//时间标记
  // auto pos_liD_e = state_inout.pos_end + state_inout.rot_end *
  // Lid_offset_to_IMU; auto R_liD_e   = state_inout.rot_end * Lidar_R_to_IMU;

  // cout<<"[ IMU ]: vel "<<state_inout.vel_end.transpose()<<" pos
  // "<<state_inout.pos_end.transpose()<<"
  // ba"<<state_inout.bias_a.transpose()<<" bg
  // "<<state_inout.bias_g.transpose()<<endl; cout<<"propagated cov:
  // "<<state_inout.cov.diagonal().transpose()<<endl;

  //   cout<<"UndistortPcl Time:";
  //   for (auto it = IMUpose.begin(); it != IMUpose.end(); ++it) {
  //     cout<<it->offset_time<<" ";
  //   }
  //   cout<<endl<<"UndistortPcl size:"<<IMUpose.size()<<endl;
  //   cout<<"Undistorted pcl_out.size: "<<pcl_out.size()
  //          <<"lidar_meas.size: "<<lidar_meas.lidar->points.size()<<endl;
  if (pcl_wait_proc.points.size() < 1) return;//点云尺寸过小则不要

  /*** undistort each lidar point (backward propagation), ONLY working for LIO
   * update ***/

  //对 LiDAR 点云进行去畸变处理（Undistortion），以补偿 LiDAR 扫描过程中由于传感器运动引起的点云畸变
  /*
  具体计算畸变的过程 
  计算出extR_Ri转换到雷达的旋转与exrR_extT的雷达到imu的旋转补偿
  遍历IMU位姿 提取IMU数据
  */
  if (lidar_meas.lio_vio_flg == LIO)
  {//初始化变量
    auto it_pcl = pcl_wait_proc.points.end() - 1; //点云尺寸初始化
    M3D extR_Ri(Lid_rot_to_IMU.transpose() * state_inout.rot_end.transpose());//转换到雷达的旋转
    V3D exrR_extT(Lid_rot_to_IMU.transpose() * Lid_offset_to_IMU);//雷达到imu的旋转补偿
    //遍历 IMU 位姿
    for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
    {
      auto head = it_kp - 1;
      auto tail = it_kp;
      //提取 IMU 数据
      R_imu << MAT_FROM_ARRAY(head->rot);//旋转矩阵
      acc_imu << VEC_FROM_ARRAY(head->acc);//加速度
      // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
      vel_imu << VEC_FROM_ARRAY(head->vel);//速度
      pos_imu << VEC_FROM_ARRAY(head->pos);//位置
      angvel_avr << VEC_FROM_ARRAY(head->gyr);//角速度

      // printf("head->offset_time: %lf \n", head->offset_time);
      // printf("it_pcl->curvature: %lf pt dt: %lf \n", it_pcl->curvature,
      // it_pcl->curvature / double(1000) - head->offset_time);
      //遍历点云并去畸变  遍历点云中所有时间戳晚于 head->offset_time 的点
      for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--)
      {//计算时间差
        dt = it_pcl->curvature / double(1000) - head->offset_time;

        /* Transform to the 'end' frame */
        //计算点云去畸变的变换
        M3D R_i(R_imu * Exp(angvel_avr, dt));//使用 IMU 的角速度 angvel_avr 和时间差 dt，通过指数映射计算点云点的旋转矩阵
        V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - state_inout.pos_end);//计算点云点的平移量，考虑了 IMU 的位置、速度和加速度
        //应用去畸变变换
        V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);// 当前点云点的坐标
        // V3D P_compensate = Lid_rot_to_IMU.transpose() *
        // (state_inout.rot_end.transpose() * (R_i * (Lid_rot_to_IMU * P_i +
        // Lid_offset_to_IMU) + T_ei) - Lid_offset_to_IMU);

        //去畸变后的点云点坐标，经过 LiDAR 到 IMU 的外参变换和时间同步补偿

        V3D P_compensate = (extR_Ri * (R_i * (Lid_rot_to_IMU * P_i + Lid_offset_to_IMU) + T_ei) - exrR_extT);
        //保存去畸变后的点云
        /// save Undistorted points and their rotation
        it_pcl->x = P_compensate(0);
        it_pcl->y = P_compensate(1);
        it_pcl->z = P_compensate(2);//输出畸变的点云处理

        if (it_pcl == pcl_wait_proc.points.begin()) break;//
      }
    }
    //清理和输出点云
    pcl_out = pcl_wait_proc;//承接前项传播i得到的点云数据
    pcl_wait_proc.clear();//清除进程的点云数据
    IMUpose.clear();//清除位置数据
  }
  // printf("[ IMU ] time forward: %lf, backward: %lf.\n", t1 - t0, omp_get_wtime() - t1);
}

//imu进程主函数
/*
LidarMeasureGroup &lidar_meas: 包含当前帧的 LiDAR 和 IMU 数据的测量组。
StatesGroup &stat: 当前帧的状态变量，包括速度、位置、旋转矩阵等。
PointCloudXYZI::Ptr cur_pcl_un_: 当前帧的点云数据指针，用于存储去畸变后的点云
*/
void ImuProcess::Process2(LidarMeasureGroup &lidar_meas, StatesGroup &stat, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1, t2, t3;
  t1 = omp_get_wtime();
  ROS_ASSERT(lidar_meas.lidar != nullptr);//非空
  if (!imu_en)//无imu模式
  {//用于在没有 IMU 数据的情况下，通过 LiDAR 数据进行状态传播。
    Forward_without_imu(lidar_meas, stat, *cur_pcl_un_);//前项传播
    return;
  }

  MeasureGroup meas = lidar_meas.measures.back();//给出当前帧的IMU和LiDAR数据

  if (imu_need_init)//imu初始化
  {//时间对齐 计算点云结束时间
    double pcl_end_time = lidar_meas.lio_vio_flg == LIO ? meas.lio_time : meas.vio_time;
    // lidar_meas.last_lio_update_time = pcl_end_time;

    if (meas.imu.empty()) { return; };//imu是空的
    /// The very first lidar frame
    IMU_init(meas, stat, init_iter_num);//包括重力、偏置和协方差的初始化

    imu_need_init = true;

    last_imu = meas.imu.back();
    //init_iter_num为帧数  所以间隔为20帧开始重新初始化
    if (init_iter_num > MAX_INI_COUNT)
    {
      // cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init = false;
      ROS_INFO("IMU Initials: Gravity: %.4f %.4f %.4f %.4f; acc covarience: "
               "%.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f \n",
               stat.gravity[0], stat.gravity[1], stat.gravity[2], mean_acc.norm(), cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1],
               cov_gyr[2]);
      ROS_INFO("IMU Initials: ba covarience: %.8f %.8f %.8f; bg covarience: "
               "%.8f %.8f %.8f",//偏置加速度的协方差 
               cov_bias_acc[0], cov_bias_acc[1], cov_bias_acc[2], cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"), ios::out);//打印
    }

    return;
  }
  //如果imu使用则用于对 LiDAR 点云进行去畸变处理，以补偿 LiDAR 扫描过程中由于传感器运动引起的点云畸变
  UndistortPcl(lidar_meas, stat, *cur_pcl_un_);//点云去畸变处理
  // cout << "[ IMU ] undistorted point num: " << cur_pcl_un_->size() << endl;
}