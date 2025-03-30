/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "/home/ao/jiguang/LIVO2/ws_livo2/src/FAST-LIVO2/include/LIVMapper.h"

LIVMapper::LIVMapper(ros::NodeHandle &nh)
    : extT(0, 0, 0),//平移
      extR(M3D::Identity())//旋转
{
  extrinT.assign(3, 0.0);
  extrinR.assign(9, 0.0);
  cameraextrinT.assign(3, 0.0);
  cameraextrinR.assign(9, 0.0);

  p_pre.reset(new Preprocess());//IMU数据
  p_imu.reset(new ImuProcess());//LiDAR预处理

  readParameters(nh);
  VoxelMapConfig voxel_config;
  loadVoxelConfig(nh, voxel_config);

  visual_sub_map.reset(new PointCloudXYZI());
  feats_undistort.reset(new PointCloudXYZI());
  feats_down_body.reset(new PointCloudXYZI());
  feats_down_world.reset(new PointCloudXYZI());
  pcl_w_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_save.reset(new PointCloudXYZRGB());
  pcl_wait_save_intensity.reset(new PointCloudXYZI());
  voxelmap_manager.reset(new VoxelMapManager(voxel_config, voxel_map));
  vio_manager.reset(new VIOManager());
  root_dir = ROOT_DIR;
  initializeFiles();//初始化文件
  initializeComponents();//初始化组成
  path.header.stamp = ros::Time::now();
  path.header.frame_id = "camera_init";
}

LIVMapper::~LIVMapper() {}

void LIVMapper::readParameters(ros::NodeHandle &nh)
{
  nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");//LiDAR 数据的 ROS 话题名称，默认值为 /livox/lidar
  nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");//IMU 数据的 ROS 话题名称，默认值为 /livox/imu
  nh.param<bool>("common/ros_driver_bug_fix", ros_driver_fix_en, false);
  nh.param<int>("common/img_en", img_en, 1);//是否启用 LiDAR，默认值为 1（启用）
  nh.param<int>("common/lidar_en", lidar_en, 1);
  nh.param<string>("common/img_topic", img_topic, "/left_camera/image");//图像数据的 ROS 话题名称，默认值为 /left_camera/image

  nh.param<bool>("vio/normal_en", normal_en, true);//是否启用法向量计算，默认值为 true
  nh.param<bool>("vio/inverse_composition_en", inverse_composition_en, false);//是否启用逆合成，默认值为 false
  nh.param<int>("vio/max_iterations", max_iterations, 5);//最大迭代次数，默认值为 5
  nh.param<double>("vio/img_point_cov", IMG_POINT_COV, 100);//图像点的协方差，默认值为 100
  nh.param<bool>("vio/raycast_en", raycast_en, false);//是否启用射线投射，默认值为 false
  nh.param<bool>("vio/exposure_estimate_en", exposure_estimate_en, true);//是否启用曝光估计，默认值为 true
  nh.param<double>("vio/inv_expo_cov", inv_expo_cov, 0.2);//逆曝光协方差，默认值为 0.2
  nh.param<int>("vio/grid_size", grid_size, 5);//图像网格大小，默认值为 5
  nh.param<int>("vio/grid_n_height", grid_n_height, 17);//图像网格高度，默认值为 17
  nh.param<int>("vio/patch_pyrimid_level", patch_pyrimid_level, 3);//图像金字塔层数，默认值为 3
  nh.param<int>("vio/patch_size", patch_size, 8);//图像 patch 大小，默认值为 8
  nh.param<double>("vio/outlier_threshold", outlier_threshold, 1000);//异常值阈值，默认值为 1000

  nh.param<double>("time_offset/exposure_time_init", exposure_time_init, 0.0);//初始曝光时间，默认值为 0.0
  nh.param<double>("time_offset/img_time_offset", img_time_offset, 0.0);//图像时间戳的偏移量，默认值为 0.0
  nh.param<double>("time_offset/imu_time_offset", imu_time_offset, 0.0);//IMU 时间戳的偏移量，默认值为 0.0
  nh.param<double>("time_offset/lidar_time_offset", lidar_time_offset, 0.0);
  nh.param<bool>("uav/imu_rate_odom", imu_prop_enable, false);
  nh.param<bool>("uav/gravity_align_en", gravity_align_en, false);//是否启用重力对齐，默认值为 false

  nh.param<string>("evo/seq_name", seq_name, "01");
  nh.param<bool>("evo/pose_output_en", pose_output_en, false);
  nh.param<double>("imu/gyr_cov", gyr_cov, 1.0);//陀螺仪的协方差，默认值为 1.0
  nh.param<double>("imu/acc_cov", acc_cov, 1.0);//加速度计的协方差，默认值为 1.0
  nh.param<int>("imu/imu_int_frame", imu_int_frame, 3);//IMU 数据插值的帧数，默认值为 3
  nh.param<bool>("imu/imu_en", imu_en, false);//是否启用 IMU，默认值为 false
  nh.param<bool>("imu/gravity_est_en", gravity_est_en, true);
  nh.param<bool>("imu/ba_bg_est_en", ba_bg_est_en, true);
/*
点云预处理
*/
  nh.param<double>("preprocess/blind", p_pre->blind, 0.01);//盲区距离，默认值为 0.01
  nh.param<double>("preprocess/filter_size_surf", filter_size_surf_min, 0.5);//点云降采样的最小体素大小，默认值为 0.5
  nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
  nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 6);
  nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 3);
  nh.param<bool>("preprocess/feature_extract_enabled", p_pre->feature_enabled, false);//是否启用特征提取，默认值为 false

  nh.param<int>("pcd_save/interval", pcd_save_interval, -1);//点云保存的间隔，默认值为 -1（不保存）
  nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);//是否启用点云保存，默认值为 false
  nh.param<bool>("pcd_save/colmap_output_en", colmap_output_en, false);//是否启用 COLMAP 输出，默认值为 false
  nh.param<double>("pcd_save/filter_size_pcd", filter_size_pcd, 0.5);//点云保存时的体素大小，默认值为 0.5
  nh.param<vector<double>>("extrin_calib/extrinsic_T", extrinT, vector<double>());//LiDAR 到 IMU 的外部平移向量
  nh.param<vector<double>>("extrin_calib/extrinsic_R", extrinR, vector<double>());//LiDAR 到 IMU 的外部旋转矩阵
  nh.param<vector<double>>("extrin_calib/Pcl", cameraextrinT, vector<double>());//相机到 IMU 的外部平移向量
  nh.param<vector<double>>("extrin_calib/Rcl", cameraextrinR, vector<double>());//相机到 IMU 的外部旋转矩阵

  /*
  调试与发布
  */
  nh.param<double>("debug/plot_time", plot_time, -10);//绘图时间  case LO:，默认值为 -10
  nh.param<int>("debug/frame_cnt", frame_cnt, 6);//帧计数器，默认值为 6
/*
发布配置
*/
  nh.param<double>("publish/blind_rgb_points", blind_rgb_points, 0.01);//盲区 RGB 点的阈值，默认值为 0.01
  nh.param<int>("publish/pub_scan_num", pub_scan_num, 1);//发布的点云帧数，默认值为 1
  nh.param<bool>("publish/pub_effect_point_en", pub_effect_point_en, false);//是否发布有效点云，默认值为 false
  nh.param<bool>("publish/dense_map_en", dense_map_en, false);//是否启用稠密地图，默认值为 false

  p_pre->blind_sqr = p_pre->blind * p_pre->blind;
}
/*
体素降采样：通过downSizeFilterSurf减少LiDAR点数
VIO管理器：加载相机模型、设置算法参数（网格尺寸、补丁大小）、配置传感器外参
IMU处理器：设置噪声参数、启用/禁用估计功能（偏置、重力）
*/
void LIVMapper::initializeComponents() 
{
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);//点云降采样滤波器初始化 设置点云降采样滤波器的体素大小，用于减少点云数据量
  extT << VEC_FROM_ARRAY(extrinT);
  extR << MAT_FROM_ARRAY(extrinR);//将 LiDAR 到 IMU 的外部平移 (extrinT) 和旋转 (extrinR) 参数赋值给 extT 和 extR

  voxelmap_manager->extT_ << VEC_FROM_ARRAY(extrinT);
  voxelmap_manager->extR_ << MAT_FROM_ARRAY(extrinR);//同步设置体素地图管理器 (voxelmap_manager) 的外部参数，确保地图构建时坐标转换正确
/*
从 ROS 参数服务器的 laserMapping 命名空间加载相机模型参数到 VIO 管理器
*/
  if (!vk::camera_loader::loadFromRosNs("laserMapping", vio_manager->cam)) throw std::runtime_error("Camera model not correctly specified.");

  vio_manager->grid_size = grid_size;//图像网格大小，用于特征提取
  vio_manager->patch_size = patch_size;//图像 patch 大小，用于光流跟踪
  vio_manager->outlier_threshold = outlier_threshold;//异常值阈值，过滤错误匹配
  vio_manager->setImuToLidarExtrinsic(extT, extR);//设置 IMU 到 LiDAR 的外参
  vio_manager->setLidarToCameraExtrinsic(cameraextrinR, cameraextrinT);//设置 LiDAR 到相机的外参
  vio_manager->state = &_state;
  vio_manager->state_propagat = &state_propagat;
  vio_manager->max_iterations = max_iterations;//优化最大迭代次数
  vio_manager->img_point_cov = IMG_POINT_COV;
  vio_manager->normal_en = normal_en;
  vio_manager->inverse_composition_en = inverse_composition_en;
  vio_manager->raycast_en = raycast_en;
  vio_manager->grid_n_width = grid_n_width;
  vio_manager->grid_n_height = grid_n_height;
  vio_manager->patch_pyrimid_level = patch_pyrimid_level;
  vio_manager->exposure_estimate_en = exposure_estimate_en;//是否启用曝光时间估计
  vio_manager->colmap_output_en = colmap_output_en;
  vio_manager->initializeVIO();

  p_imu->set_extrinsic(extT, extR);//IMU 到 LiDAR 的平移和旋转
  p_imu->set_gyr_cov_scale(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov_scale(V3D(acc_cov, acc_cov, acc_cov));//陀螺仪 (gyr_cov) 和加速度计 (acc_cov) 的协方差缩放因子
  p_imu->set_inv_expo_cov(inv_expo_cov);
  p_imu->set_gyr_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_acc_bias_cov(V3D(0.0001, 0.0001, 0.0001));//陀螺仪和加速度计偏差的协方差（硬编码为 0.0001）
  p_imu->set_imu_init_frame_num(imu_int_frame);//imu_int_frame 控制 IMU 初始化所需的帧数

  if (!imu_en) p_imu->disable_imu();
  if (!gravity_est_en) p_imu->disable_gravity_est();
  if (!ba_bg_est_en) p_imu->disable_bias_est();
  if (!exposure_estimate_en) p_imu->disable_exposure_est();


/*
若同时启用图像和 LiDAR (img_en && lidar_en)，选择 LIVO（LiDAR-Inertial-Visual Odometry）
否则，若启用 IMU (imu_en)，选择 ONLY_LIO（仅 LiDAR-惯性里程计）。
否则，选择 ONLY_LO（仅 LiDAR 里程计）
*/
  slam_mode_ = (img_en && lidar_en) ? LIVO : imu_en ? ONLY_LIO : ONLY_LO;//
}

void LIVMapper::initializeFiles() //初始化文件
{
  if (pcd_save_en && colmap_output_en)
  {
      const std::string folderPath = std::string(ROOT_DIR) + "/scripts/colmap_output.sh";//日志输出
      
      std::string chmodCommand = "chmod +x " + folderPath;
      
      int chmodRet = system(chmodCommand.c_str());  
      if (chmodRet != 0) {
          std::cerr << "Failed to set execute permissions for the script." << std::endl;
          return;
      }

      int executionRet = system(folderPath.c_str());
      if (executionRet != 0) {
          std::cerr << "Failed to execute the script." << std::endl;
          return;
      }
  }
  if(colmap_output_en) fout_points.open(std::string(ROOT_DIR) + "Log/Colmap/sparse/0/points3D.txt", std::ios::out);
  if(pcd_save_interval > 0) fout_pcd_pos.open(std::string(ROOT_DIR) + "Log/PCD/scans_pos.json", std::ios::out);
  fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), std::ios::out);
  fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), std::ios::out);
}
/*
主函数首先运行的函数
*/
void LIVMapper::initializeSubscribersAndPublishers(ros::NodeHandle &nh, image_transport::ImageTransport &it) //初始化ros订阅部分
{
  sub_pcl = p_pre->lidar_type == AVIA ? //根据 LiDAR 类型选择不同的回调函数
            nh.subscribe(lid_topic, 200000, &LIVMapper::livox_pcl_cbk, this): 
            nh.subscribe(lid_topic, 200000, &LIVMapper::standard_pcl_cbk, this);
  sub_imu = nh.subscribe(imu_topic, 200000, &LIVMapper::imu_cbk, this);//订阅 IMU 和图像数据
  sub_img = nh.subscribe(img_topic, 200000, &LIVMapper::img_cbk, this);
  
  pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);//发布配准后的点云（去畸变、降采样）
  pubNormal = nh.advertise<visualization_msgs::MarkerArray>("visualization_marker", 100);
  pubSubVisualMap = nh.advertise<sensor_msgs::PointCloud2>("/cloud_visual_sub_map_before", 100);
  pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100);//发布有效特征点云（用于状态估计）
  pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);//发布全局地图点云
  pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);//发布优化后的里程计信息
  pubPath = nh.advertise<nav_msgs::Path>("/path", 10);//发布历史路径

  plane_pub = nh.advertise<visualization_msgs::Marker>("/planner_normal", 1);//发布平面法线（用于可视化平面特征）
  voxel_pub = nh.advertise<visualization_msgs::MarkerArray>("/voxels", 1);//发布体素结构（调试体素地图）

  pubLaserCloudDyn = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj", 100);
  pubLaserCloudDynRmed = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj_removed", 100);
  pubLaserCloudDynDbg = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj_dbg_hist", 100);
  mavros_pose_publisher = nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 10);//兼容 MAVROS 的位姿信息

  pubImage = it.advertise("/rgb_img", 1);//发布处理后的 RGB 图像

  pubImuPropOdom = nh.advertise<nav_msgs::Odometry>("/LIVO2/imu_propagate", 10000);//发布 IMU 传播的中间里程计
  imu_prop_timer = nh.createTimer(ros::Duration(0.004), &LIVMapper::imu_prop_callback, this);
  voxelmap_manager->voxel_map_pub_= nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);
}

void LIVMapper::handleFirstFrame() //第一帧传入数据 run函数中的
{
  if (!is_first_frame)
  {
    _first_lidar_time = LidarMeasures.last_lio_update_time;
    p_imu->first_lidar_time = _first_lidar_time; // Only for IMU data log
    is_first_frame = true;
    cout << "FIRST LIDAR FRAME!" << endl;
  }
}

void LIVMapper::gravityAlignment() //重力初始化及相关参数设置 processImu函数中引用
{
  if (!p_imu->imu_need_init && !gravity_align_finished) 
  {
    std::cout << "Gravity Alignment Starts" << std::endl;
    V3D ez(0, 0, -1), gz(_state.gravity);
    Quaterniond G_q_I0 = Quaterniond::FromTwoVectors(gz, ez);
    M3D G_R_I0 = G_q_I0.toRotationMatrix();

    _state.pos_end = G_R_I0 * _state.pos_end;
    _state.rot_end = G_R_I0 * _state.rot_end;
    _state.vel_end = G_R_I0 * _state.vel_end;
    _state.gravity = G_R_I0 * _state.gravity;
    gravity_align_finished = true;
    std::cout << "Gravity Alignment Finished" << std::endl;
  }
}

void LIVMapper::processImu() //imu进程
{
  // double t0 = omp_get_wtime();

  p_imu->Process2(LidarMeasures, _state, feats_undistort);

  if (gravity_align_en) gravityAlignment();//引用上面的重力初始化

  state_propagat = _state;
  voxelmap_manager->state_ = _state;
  voxelmap_manager->feats_undistort_ = feats_undistort;

  // double t_prop = omp_get_wtime();

  // std::cout << "[ Mapping ] feats_undistort: " << feats_undistort->size() << std::endl;
  // std::cout << "[ Mapping ] predict cov: " << _state.cov.diagonal().transpose() << std::endl;
  // std::cout << "[ Mapping ] predict sta: " << state_propagat.pos_end.transpose() << state_propagat.vel_end.transpose() << std::endl;
}

void LIVMapper::stateEstimationAndMapping() //状态估计 run函数中引用
{
  switch (LidarMeasures.lio_vio_flg) //lio-vio完成了第一次测量时执行
  {
    case VIO:
      handleVIO();
      break;
    case LIO:
    case LO:
      handleLIO();
      break;
  }
}

void LIVMapper::handleVIO() //vio执行
{
  euler_cur = RotMtoEuler(_state.rot_end);// 从旋转矩阵提取欧拉角
  fout_pre << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "// 弧度转角度
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "// 位置 速度 陀螺仪偏置 
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << std::endl;// 加速度计偏置 逆曝光时间
    
  if (pcl_w_wait_pub->empty() || (pcl_w_wait_pub == nullptr)) //确保待处理的点云数据有效，避免空指针或空点云导致后续崩溃
  {
    std::cout << "[ VIO ] No point!!!" << std::endl;
    return;
  }
    
  std::cout << "[ VIO ] Raw feature num: " << pcl_w_wait_pub->points.size() << std::endl;//输出当前帧的原始特征点数量，用于调试或监控特征提取性能

  if (fabs((LidarMeasures.last_lio_update_time - _first_lidar_time) - plot_time) < (frame_cnt / 2 * 0.1)) //根据时间差动态启用/禁用可视化绘图功能
  {
    vio_manager->plot_flag = true;
  } 
  else 
  {
    vio_manager->plot_flag = false;
  }

  vio_manager->processFrame(LidarMeasures.measures.back().img, _pv_list, voxelmap_manager->voxel_map_, LidarMeasures.last_lio_update_time - _first_lidar_time);

  if (imu_prop_enable) //若启用IMU传播，更新扩展卡尔曼滤波（EKF）状态
  {
    ekf_finish_once = true;//标记EKF完成
    latest_ekf_state = _state;// 更新最新状态
    latest_ekf_time = LidarMeasures.last_lio_update_time;// 记录时间戳
    state_update_flg = true;// 触发状态更新标志
  }

  // int size_sub_map = vio_manager->visual_sub_map_cur.size();
  // visual_sub_map->reserve(size_sub_map);
  // for (int i = 0; i < size_sub_map; i++) 
  // {
  //   PointType temp_map;
  //   temp_map.x = vio_manager->visual_sub_map_cur[i]->pos_[0];
  //   temp_map.y = vio_manager->visual_sub_map_cur[i]->pos_[1];
  //   temp_map.z = vio_manager->visual_sub_map_cur[i]->pos_[2];
  //   temp_map.intensity = 0.;
  //   visual_sub_map->push_back(temp_map);
  // }

  publish_frame_world(pubLaserCloudFullRes, vio_manager);//// 发布全局点云
  publish_img_rgb(pubImage, vio_manager);// 发布RGB图像
//// 记录后处理日志
  euler_cur = RotMtoEuler(_state.rot_end);
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " " << feats_undistort->points.size() << std::endl;
}//点云与图像发布：将处理后的点云和图像发送到ROS话题

void LIVMapper::handleLIO() //lio部分代码
{    
  euler_cur = RotMtoEuler(_state.rot_end);//将旋转矩阵转换为欧拉角（通常为Z-Y-X顺序）
  fout_pre << setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
           << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
           << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << endl;//inv_expo_time：与相机曝光时间补偿相关，此处仅记录X分量
           
  if (feats_undistort->empty() || (feats_undistort == nullptr)) //确保去畸变后的点云数据有效，避免空数据导致后续崩溃
  {
    std::cout << "[ LIO ]: No point!!!" << std::endl;
    return;
  }

  double t0 = omp_get_wtime();

  downSizeFilterSurf.setInputCloud(feats_undistort);//通过体素滤波减少点云密度，提升后续处理效率
  downSizeFilterSurf.filter(*feats_down_body);//filter_size_surf_min 控制下采样体素尺寸（例如0.5米）
  
  double t_down = omp_get_wtime();

  feats_down_size = feats_down_body->points.size();
  voxelmap_manager->feats_down_body_ = feats_down_body;
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, feats_down_world);//将降采样后的点云从LiDAR坐标系转换到世界坐标系
  voxelmap_manager->feats_down_world_ = feats_down_world;
  voxelmap_manager->feats_down_size_ = feats_down_size;
  
  if (!lidar_map_inited) //首次运行时构建体素地图，存储环境结构信息。依赖条件：lidar_map_inited 标志确保仅初始化一次
  {
    lidar_map_inited = true;
    /*
    在voxel_map.cpp文件中
    */
    voxelmap_manager->BuildVoxelMap();//构建新的体素地图
  }

  double t1 = omp_get_wtime();
/*
具体在voxel_map函数中 调用状态估计函数进行下一部分调用
*/
  voxelmap_manager->StateEstimation(state_propagat);//通过点云与体素地图匹配（如ICP算法），优化LiDAR位姿
  _state = voxelmap_manager->state_;
  _pv_list = voxelmap_manager->pv_list_;

  double t2 = omp_get_wtime();

  if (imu_prop_enable) //若启用，更新扩展卡尔曼滤波（EKF）状态，用于高频位姿输出
  {
    ekf_finish_once = true;//ekf_finish_once 表示EKF已完成一次更新，可用于后续IMU插值
    latest_ekf_state = _state;//将优化后的状态 _state 保存到 latest_ekf_state，供IMU传播线程使用
    latest_ekf_time = LidarMeasures.last_lio_update_time;//latest_ekf_time 确保IMU数据与LiDAR时间戳对齐
    state_update_flg = true;
  }

  if (pose_output_en)//将优化后的位姿（位置+四元数）写入 seq_name.txt，用于后续评测（如EVO工具）
  {
    static bool pos_opend = false;
    static int ocount = 0;
    std::ofstream outFile, evoFile;
    if (!pos_opend) 
    {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt", std::ios::out);
      pos_opend = true;
      if (!evoFile.is_open()) ROS_ERROR("open fail\n");
    } 
    else 
    {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt", std::ios::app);
      if (!evoFile.is_open()) ROS_ERROR("open fail\n");
    }
    Eigen::Matrix4d outT;
    Eigen::Quaterniond q(_state.rot_end);
    evoFile << std::fixed;
    evoFile << LidarMeasures.last_lio_update_time << " " << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
  
  euler_cur = RotMtoEuler(_state.rot_end);//欧拉角转四元数：使用ROS的tf库生成四元数消息
  geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
  publish_odometry(pubOdomAftMapped);//将位姿、速度、协方差封装为nav_msgs/Odometry消息，通过pubOdomAftMapped话题发布

  double t3 = omp_get_wtime();

  PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI());
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, world_lidar);// 点云从LiDAR系到世界系的变换
  for (size_t i = 0; i < world_lidar->points.size(); i++) //协方差传播：考虑LiDAR-IMU外参 extR 和状态协方差
  {
    voxelmap_manager->pv_list_[i].point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
    M3D point_crossmat = voxelmap_manager->cross_mat_list_[i];
    M3D var = voxelmap_manager->body_cov_list_[i];
    var = (_state.rot_end * extR) * var * (_state.rot_end * extR).transpose() +
          (-point_crossmat) * _state.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + _state.cov.block<3, 3>(3, 3);
    voxelmap_manager->pv_list_[i].var = var;
  }
  /*
  重要的步骤
  voxel_map.cpp函数中调用的更新体素地图的函数 进行插帧操作
  */
  voxelmap_manager->UpdateVoxelMap(voxelmap_manager->pv_list_);// 插入当前帧点云到地图
  std::cout << "[ LIO ] Update Voxel Map" << std::endl;
  _pv_list = voxelmap_manager->pv_list_;
  
  double t4 = omp_get_wtime();

  if(voxelmap_manager->config_setting_.map_sliding_en)
  {
    /*
    调用voxel_map检查是否超出
    如果没超出则更新地图
    判断当前位置 position_last_ 是否移动足够远，如果是，则更新地图
    */
    voxelmap_manager->mapSliding();// 移除超出滑动窗口的区域
  }
  
  PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort : feats_down_body);// 世界系点云
  int size = laserCloudFullRes->points.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++) 
  {
    RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
  }
  *pcl_w_wait_pub = *laserCloudWorld; // 缓存待发布点云

  if (!img_en) publish_frame_world(pubLaserCloudFullRes, vio_manager);
  if (pub_effect_point_en) publish_effect_world(pubLaserCloudEffect, voxelmap_manager->ptpl_list_);
  if (voxelmap_manager->config_setting_.is_pub_plane_map_) voxelmap_manager->pubVoxelMap();//pubVoxelMap为发布的平面数据
  publish_path(pubPath);// 发布路径
  publish_mavros(mavros_pose_publisher);// 发送到MAVROS

  frame_num++;
  aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t4 - t0) / frame_num;

  // aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t2 - t1) / frame_num;
  // aver_time_map_inre = aver_time_map_inre * (frame_num - 1) / frame_num + (t4 - t3) / frame_num;
  // aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time) / frame_num;
  // aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_const_H_time / frame_num;
  // printf("[ mapping time ]: per scan: propagation %0.6f downsample: %0.6f match: %0.6f solve: %0.6f  ICP: %0.6f  map incre: %0.6f total: %0.6f \n"
  //         "[ mapping time ]: average: icp: %0.6f construct H: %0.6f, total: %0.6f \n",
  //         t_prop - t0, t1 - t_prop, match_time, solve_time, t3 - t1, t5 - t3, t5 - t0, aver_time_icp, aver_time_const_H_time, aver_time_consu);

  // printf("\033[1;36m[ LIO mapping time ]: current scan: icp: %0.6f secs, map incre: %0.6f secs, total: %0.6f secs.\033[0m\n"
  //         "\033[1;36m[ LIO mapping time ]: average: icp: %0.6f secs, map incre: %0.6f secs, total: %0.6f secs.\033[0m\n",
  //         t2 - t1, t4 - t3, t4 - t0, aver_time_icp, aver_time_map_inre, aver_time_consu);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|                         LIO Mapping Time                    |\033[0m\n");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "DownSample", t_down - t0);//点云降采样耗时
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "ICP", t2 - t1);//点云匹配优化耗时
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "updateVoxelMap", t4 - t3);//体素地图更新耗时
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Current Total Time", t4 - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Average Total Time", aver_time_consu);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " " << feats_undistort->points.size() << std::endl;
}
/*
保存存储的各种状态
*/
void LIVMapper::savePCD() 
{
  // if (pcd_save_en && pcl_wait_save->points.size() > 0 && pcd_save_interval < 0) 
  if (pcd_save_en && (pcl_wait_save->points.size() > 0 || pcl_wait_save_intensity->points.size() > 0) && pcd_save_interval < 0) 
  {
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    // pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
    // voxel_filter.setInputCloud(pcl_wait_save);
    // voxel_filter.setLeafSize(filter_size_pcd, filter_size_pcd, filter_size_pcd);
    // voxel_filter.filter(*downsampled_cloud);

    std::string raw_points_dir = std::string(ROOT_DIR) + "Log/PCD/all_raw_points.pcd";
    std::string downsampled_points_dir = std::string(ROOT_DIR) + "Log/PCD/all_downsampled_points.pcd";

    pcl::PCDWriter pcd_writer;

    // Save the raw point cloud data
    // pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save);
    // std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir 
    //           << " with point count: " << pcl_wait_save->points.size() << RESET << std::endl;

    // // Save the downsampled point cloud data
    // pcd_writer.writeBinary(downsampled_points_dir, *downsampled_cloud);
    // std::cout << GREEN << "Downsampled point cloud data saved to: " << downsampled_points_dir 
    //       << " with point count after filtering: " << downsampled_cloud->points.size() << RESET << std::endl;

    // if(colmap_output_en)
    // {
    //   fout_points << "# 3D point list with one line of data per point\n";
    //   fout_points << "#  POINT_ID, X, Y, Z, R, G, B, ERROR\n";
    //   for (size_t i = 0; i < downsampled_cloud->size(); ++i)    
      // {
      //     const auto& point = downsampled_cloud->points[i];
      //     fout_points << i << " "
      //                 << std::fixed << std::setprecision(6)
      //                 << point.x << " " << point.y << " " << point.z << " "
      //                 << static_cast<int>(point.r) << " "
      //                 << static_cast<int>(point.g) << " "
      //                 << static_cast<int>(point.b) << " "
      //                 << 0 << std::endl;
      // }
      if (img_en)
      {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
        voxel_filter.setInputCloud(pcl_wait_save);
        voxel_filter.setLeafSize(filter_size_pcd, filter_size_pcd, filter_size_pcd);
        voxel_filter.filter(*downsampled_cloud);
        pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save); // Save the raw point cloud data
        std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir 
                  << " with point count: " << pcl_wait_save->points.size() << RESET << std::endl;
        pcd_writer.writeBinary(downsampled_points_dir, *downsampled_cloud); // Save the downsampled point cloud data
        std::cout << GREEN << "Downsampled point cloud data saved to: " << downsampled_points_dir 
                  << " with point count after filtering: " << downsampled_cloud->points.size() << RESET << std::endl;
        if(colmap_output_en)
        {
          fout_points << "# 3D point list with one line of data per point\n";
          fout_points << "#  POINT_ID, X, Y, Z, R, G, B, ERROR\n";
          for (size_t i = 0; i < downsampled_cloud->size(); ++i) 
          {
              const auto& point = downsampled_cloud->points[i];

              fout_points << i << " "
                          << std::fixed << std::setprecision(6)
                          << point.x << " " << point.y << " " << point.z << " "
                          << static_cast<int>(point.r) << " "
                          << static_cast<int>(point.g) << " "
                          << static_cast<int>(point.b) << " "
                          << 0 << std::endl;
      }
    }
  }
    else
    {      
      pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save_intensity);
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir 
                << " with point count: " << pcl_wait_save_intensity->points.size() << RESET << std::endl;
     }
  }
}


/*
主循环，负责驱动整个SLAM系统的处理流程
*/
void LIVMapper::run() 
{
  ros::Rate rate(5000);// 设置循环频率为5000Hz
  while (ros::ok()) 
  {
    ros::spinOnce();// 处理回调队列中的消息
    if (!sync_packages(LidarMeasures)) // 同步LiDAR和IMU数据包
    {
      rate.sleep();// 无数据时休眠
      continue;
    }   
    handleFirstFrame();// 处理首帧初始化

    processImu();// 处理IMU预积分

    // if (!p_imu->imu_time_init) continue;

    stateEstimationAndMapping();// 执行状态估计与建图
  }
  savePCD();// 退出时保存点云地图
}


void LIVMapper::prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr)//基于IMU测量值进行一次状态传播（姿态、位置、速度）
{
  // 1. 加速度归一化与偏差补偿
  double mean_acc_norm = p_imu->IMU_mean_acc_norm;//mean_acc_norm：加速度计测量值的归一化因子，用于消除重力影响
  acc_avr = acc_avr * G_m_s2 / mean_acc_norm - imu_prop_state.bias_a;//bias_a 和 bias_g：加速度计和陀螺仪偏置，从状态估计中获取
  angvel_avr -= imu_prop_state.bias_g;
// 2. 姿态更新（旋转矩阵指数映射）
  M3D Exp_f = Exp(angvel_avr, dt);
  /* propogation of IMU attitude */
  imu_prop_state.rot_end = imu_prop_state.rot_end * Exp_f;
// 3. 加速度转换到世界坐标系并计算位置和速度
  /* Specific acceleration (global frame) of IMU */
  V3D acc_imu = imu_prop_state.rot_end * acc_avr + V3D(imu_prop_state.gravity[0], imu_prop_state.gravity[1], imu_prop_state.gravity[2]);

  /* propogation of IMU */
  imu_prop_state.pos_end = imu_prop_state.pos_end + imu_prop_state.vel_end * dt + 0.5 * acc_imu * dt * dt;

  /* velocity of IMU */
  imu_prop_state.vel_end = imu_prop_state.vel_end + acc_imu * dt;
}

/*
定时器回调，处理IMU数据的高频状态传播与里程计发布
*/
void LIVMapper::imu_prop_callback(const ros::TimerEvent &e)
{
  if (p_imu->imu_need_init || !new_imu || !ekf_finish_once) { return; }
  mtx_buffer_imu_prop.lock();// 加锁保护共享数据
  new_imu = false; // 控制propagate频率和IMU频率一致
  if (imu_prop_enable && !prop_imu_buffer.empty())
  {
    static double last_t_from_lidar_end_time = 0;
    // 情况1：有新的EKF状态更新
    if (state_update_flg)
    {
      imu_propagate = latest_ekf_state;// 从EKF获取最新状态
      // drop all useless imu pkg
      // 丢弃过时的IMU数据
      while ((!prop_imu_buffer.empty() && prop_imu_buffer.front().header.stamp.toSec() < latest_ekf_time))
      {
        prop_imu_buffer.pop_front();
      }
      last_t_from_lidar_end_time = 0;
      // 遍历剩余IMU数据包进行传播
      for (int i = 0; i < prop_imu_buffer.size(); i++)
      {
        double t_from_lidar_end_time = prop_imu_buffer[i].header.stamp.toSec() - latest_ekf_time;
        double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
        // cout << "prop dt" << dt << ", " << t_from_lidar_end_time << ", " << last_t_from_lidar_end_time << endl;
        V3D acc_imu(prop_imu_buffer[i].linear_acceleration.x, prop_imu_buffer[i].linear_acceleration.y, prop_imu_buffer[i].linear_acceleration.z);// 提取加速度
        V3D omg_imu(prop_imu_buffer[i].angular_velocity.x, prop_imu_buffer[i].angular_velocity.y, prop_imu_buffer[i].angular_velocity.z);// 提取角速度
        prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
        last_t_from_lidar_end_time = t_from_lidar_end_time;
      }
      state_update_flg = false;// 清除更新标志
    }
    else
    // 情况2：无新状态，仅用最新IMU数据单步传播
    {
      V3D acc_imu(newest_imu.linear_acceleration.x, newest_imu.linear_acceleration.y, newest_imu.linear_acceleration.z);// 提取加速度
      V3D omg_imu(newest_imu.angular_velocity.x, newest_imu.angular_velocity.y, newest_imu.angular_velocity.z);// 提取角速度
      double t_from_lidar_end_time = newest_imu.header.stamp.toSec() - latest_ekf_time;
      double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
      prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
      last_t_from_lidar_end_time = t_from_lidar_end_time;
    }

    V3D posi, vel_i;
    Eigen::Quaterniond q;
    posi = imu_propagate.pos_end;
    vel_i = imu_propagate.vel_end;
    q = Eigen::Quaterniond(imu_propagate.rot_end);
    // 构造并发布里程计消息
    imu_prop_odom.header.frame_id = "world";
    imu_prop_odom.header.stamp = newest_imu.header.stamp;
    imu_prop_odom.pose.pose.position.x = posi.x();
    imu_prop_odom.pose.pose.position.y = posi.y();
    imu_prop_odom.pose.pose.position.z = posi.z();
    imu_prop_odom.pose.pose.orientation.w = q.w();
    imu_prop_odom.pose.pose.orientation.x = q.x();
    imu_prop_odom.pose.pose.orientation.y = q.y();
    imu_prop_odom.pose.pose.orientation.z = q.z();
    imu_prop_odom.twist.twist.linear.x = vel_i.x();
    imu_prop_odom.twist.twist.linear.y = vel_i.y();
    imu_prop_odom.twist.twist.linear.z = vel_i.z();
    pubImuPropOdom.publish(imu_prop_odom);
  }
  mtx_buffer_imu_prop.unlock();// 释放锁
}

/*
坐标系变换函数组
*/
void LIVMapper::transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud)
{
  PointCloudXYZI().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); i++)
  {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR * p + extT) + t);
    PointType pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}
/*
 pointBodyToWorld 函数族
*/
// 版本1：普通点类型
void LIVMapper::pointBodyToWorld(const PointType &pi, PointType &po)
{
  V3D p_body(pi.x, pi.y, pi.z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po.x = p_global(0);
  po.y = p_global(1);
  po.z = p_global(2);
  po.intensity = pi.intensity;
}
// 版本2：模板化Eigen向量
template <typename T> void LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}
// 版本3：返回Eigen向量
template <typename T> Matrix<T, 3, 1> LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi)
{
  V3D p(pi[0], pi[1], pi[2]);
  p = (_state.rot_end * (extR * p + extT) + _state.pos_end);
  Matrix<T, 3, 1> po(p[0], p[1], p[2]);
  return po;
}
//函数名包含"RGB"但未处理颜色数据，可能为未来扩展预留或命名错误
void LIVMapper::RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void LIVMapper::standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  if (!lidar_en) return;// 若LiDAR未启用则直接返回
  mtx_buffer.lock();// 加锁保护共享资源
  double cur_head_time = msg->header.stamp.toSec() + lidar_time_offset;
  // cout<<"got feature"<<endl;
   // 1. 时间戳检查（防止旧数据干扰）
  // if (msg->header.stamp.toSec() < last_timestamp_lidar)
  if (cur_head_time < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();// 时间回退时清空缓冲区
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  // 2. 点云预处理
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);// 降采样/去畸变/特征提取（假设）
  if (!ptr || ptr->empty()) {
    ROS_ERROR("Received an empty point cloud");
    mtx_buffer.unlock();
    return;
  }

  // 3. 数据入队
  lid_raw_data_buffer.push_back(ptr);
  // lid_header_time_buffer.push_back(msg->header.stamp.toSec());
  // last_timestamp_lidar = msg->header.stamp.toSec();// 更新最新时间戳
  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();// 解锁
  sig_buffer.notify_all();// 通知处理线程
}
/*
雷达回调函数  处理Livox LiDAR数据，进行预处理、时间同步检查及数据入队
*/
void LIVMapper::livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg_in)
{
  if (!lidar_en) return;// LiDAR未启用则直接返回
  mtx_buffer.lock();
  livox_ros_driver::CustomMsg::Ptr msg(new livox_ros_driver::CustomMsg(*msg_in));
  // if ((abs(msg->header.stamp.toSec() - last_timestamp_lidar) > 0.2 && last_timestamp_lidar > 0) || sync_jump_flag)
  // {
  //   ROS_WARN("lidar jumps %.3f\n", msg->header.stamp.toSec() - last_timestamp_lidar);
  //   sync_jump_flag = true;
  //   msg->header.stamp = ros::Time().fromSec(last_timestamp_lidar + 0.1);
  // }
  // ---- 时间同步检查 ----
  // 检测IMU与LiDAR时间戳差异过大（仅当IMU数据存在时）
  if (abs(last_timestamp_imu - msg->header.stamp.toSec()) > 1.0 && !imu_buffer.empty())
  {
    double timediff_imu_wrt_lidar = last_timestamp_imu - msg->header.stamp.toSec();
    printf("\033[95mSelf sync IMU and LiDAR, HARD time lag is %.10lf \n\033[0m", timediff_imu_wrt_lidar - 0.100);
    // imu_time_offset = timediff_imu_wrt_lidar;
  }

  double cur_head_time = msg->header.stamp.toSec();
  ROS_INFO("Get LiDAR, its header time: %.6f", cur_head_time);
  // 时间回退检查（如传感器重启）
  if (cur_head_time < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  // ---- 数据预处理与入队 ----
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);// 去畸变、降采样等
  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();// 唤醒主处理线程
}
/*
处理IMU数据，进行时间偏移修正、同步检查及数据入队
*/
void LIVMapper::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
  if (!imu_en) return;// IMU未启用则直接返回

  if (last_timestamp_lidar < 0.0) return; // 等待LiDAR初始化
  // ROS_INFO("get imu at time: %.6f", msg_in->header.stamp.toSec());
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
  // ---- 时间偏移修正 ----
  msg->header.stamp = ros::Time().fromSec(msg->header.stamp.toSec() - imu_time_offset);
  double timestamp = msg->header.stamp.toSec();
  // 检测LiDAR与IMU时间同步状态
  if (fabs(last_timestamp_lidar - timestamp) > 0.5 && (!ros_driver_fix_en))
  {
    ROS_WARN("IMU and LiDAR not synced! delta time: %lf .\n", last_timestamp_lidar - timestamp);
  }
  // 强制时间对齐（若启用）
  if (ros_driver_fix_en) timestamp += std::round(last_timestamp_lidar - timestamp);
  msg->header.stamp = ros::Time().fromSec(timestamp);

  mtx_buffer.lock();
// 时间回退检查
  if (last_timestamp_imu > 0.0 && timestamp < last_timestamp_imu)
  {
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    ROS_ERROR("imu loop back, offset: %lf \n", last_timestamp_imu - timestamp);
    return;
  }

  // if (last_timestamp_imu > 0.0 && timestamp > last_timestamp_imu + 0.2)
  // {

  //   ROS_WARN("imu time stamp Jumps %0.4lf seconds \n", timestamp - last_timestamp_imu);
  //   mtx_buffer.unlock();
  //   sig_buffer.notify_all();
  //   return;
  // }

  last_timestamp_imu = timestamp;

  imu_buffer.push_back(msg);
  // cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<endl;
  mtx_buffer.unlock();
  // ---- IMU传播处理 ----
  if (imu_prop_enable)
  {
    mtx_buffer_imu_prop.lock();
    if (imu_prop_enable && !p_imu->imu_need_init) 
    { prop_imu_buffer.push_back(*msg); }// 入队传播缓冲区

    newest_imu = *msg;// 更新最新IMU数据
    new_imu = true;// 标记新数据到达
    mtx_buffer_imu_prop.unlock();
  }
  sig_buffer.notify_all();
}

cv::Mat LIVMapper::getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)//获取图像
{
  cv::Mat img;
  img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
  return img;
}

// static int i = 0;
void LIVMapper::img_cbk(const sensor_msgs::ImageConstPtr &msg_in)
{
  if (!img_en) return;//有图像
  sensor_msgs::Image::Ptr msg(new sensor_msgs::Image(*msg_in));
  // if ((abs(msg->header.stamp.toSec() - last_timestamp_img) > 0.2 && last_timestamp_img > 0) || sync_jump_flag)
  // {
  //   ROS_WARN("img jumps %.3f\n", msg->header.stamp.toSec() - last_timestamp_img);
  //   sync_jump_flag = true;
  //   msg->header.stamp = ros::Time().fromSec(last_timestamp_img + 0.1);
  // }

  // Hiliti2022 40Hz
  // if (hilti_en)
  // {
  //   i++;
  //   if (i % 4 != 0) return;
  // }
  // double msg_header_time =  msg->header.stamp.toSec();
  double msg_header_time = msg->header.stamp.toSec() + img_time_offset;
  if (abs(msg_header_time - last_timestamp_img) < 0.001) return;
  ROS_INFO("Get image, its header time: %.6f", msg_header_time);
  if (last_timestamp_lidar < 0) return;

  if (msg_header_time < last_timestamp_img)
  {
    ROS_ERROR("image loop back. \n");
    return;
  }

  mtx_buffer.lock();

  double img_time_correct = msg_header_time; // last_timestamp_lidar + 0.105;

  if (img_time_correct - last_timestamp_img < 0.02)//时间对称
  {
    ROS_WARN("Image need Jumps: %.6f", img_time_correct);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    return;
  }

  cv::Mat img_cur = getImageFromMsg(msg);
  img_buffer.push_back(img_cur);
  img_time_buffer.push_back(img_time_correct);

  // ROS_INFO("Correct Image time: %.6f", img_time_correct);

  last_timestamp_img = img_time_correct;
  // cv::imshow("img", img);
  // cv::waitKey(1);
  // cout<<"last_timestamp_img:::"<<last_timestamp_img<<endl;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}
/*
LIVMapper类中用于多传感器（LiDAR、IMU、相机）数据同步的核心函数
其目标是根据SLAM模式（ONLY_LIO或LIVO）将不同传感器的数据按时间对齐，确保后续算法处理时数据一致性
*/
bool LIVMapper::sync_packages(LidarMeasureGroup &meas)//在run中引用
{
  // 检查各传感器缓冲区是否为空（若对应传感器使能）
  if (lid_raw_data_buffer.empty() && lidar_en) return false;//雷达
  if (img_buffer.empty() && img_en) return false;//图像
  if (imu_buffer.empty() && imu_en) return false;//imu

  switch (slam_mode_)//根据SLAM模式（ONLY_LIO或LIVO）组织数据包，确保时间戳对齐，为后续的状态估计与建图提供输入 只有两种模式
  {
    /*
    两个模式：
    ONLY_LIO模式：仅同步LiDAR和IMU数据。
    LIVO模式：同步LiDAR、IMU和相机数据，支持多模态融合（里面加了相机）
    */
  case ONLY_LIO://纯LiDAR-IMU模式
  {
    /*
    同步一帧LiDAR数据及其对应的IMU数据
    时间计算：假设LiDAR点云中curvature字段存储相对于帧头的时间偏移（单位ms），需确认数据格式与实际硬件一致。
    IMU覆盖检查：若IMU最新时间戳小于LiDAR结束时间，返回false等待更多IMU数据，可能导致处理延迟。
    线程安全：通过mtx_buffer保护共享缓冲区，但需确保所有访问缓冲区的代码均正确加锁
    */
    // 初始化LiDAR处理时间
    if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();
    if (!lidar_pushed)// 首次处理LiDAR帧
    {
      // If not push the lidar into measurement data buffer
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic// 取出最早的LiDAR数据

      if (meas.lidar->points.size() <= 1) return false;

      meas.lidar_frame_beg_time = lid_header_time_buffer.front();    // LiDAR起始时间                                            // generate lidar_frame_beg_time
      meas.lidar_frame_end_time = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time 计算LiDAR扫描结束时间
      meas.pcl_proc_cur = meas.lidar;
      lidar_pushed = true;              //处理第一帧的标志位                                                                         // flag
    }
    // 等待IMU数据覆盖LiDAR时间段
    if (imu_en && last_timestamp_imu < meas.lidar_frame_end_time)
    { // waiting imu message needs to be
      // larger than _lidar_frame_end_time,
      // make sure complete propagate.
      // ROS_ERROR("out sync");
      return false;// IMU数据不足，等待
    }
    // 收集IMU数据
    struct MeasureGroup m; // standard method to keep imu message.

    m.imu.clear();
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    while (!imu_buffer.empty())//清空缓冲数据
    {
      if (imu_buffer.front()->header.stamp.toSec() > meas.lidar_frame_end_time) break;
      m.imu.push_back(imu_buffer.front());
      imu_buffer.pop_front();
    }
    // 弹出已处理的LiDAR数据
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    // 标记为LIO处理并返回
    meas.lio_vio_flg = LIO; // process lidar topic, so timestamp should be lidar scan end.
/*
先进行lio补齐 再进行vio补齐
*/

    meas.measures.push_back(m);
    // ROS_INFO("ONlY HAS LiDAR and IMU, NO IMAGE!");
    lidar_pushed = false; // sync one whole lidar scan.
    return true;

    break;
  }

  case LIVO://LiDAR-IMU-相机融合模式
  /*
  LIVO模式通过状态机管理处理流程，交替处理LIO（LiDAR-IMU）和VIO（视觉-IMU）数据
  */
  {
    /*** For LIVO mode, the time of LIO update is set to be the same as VIO, LIO
     * first than VIO imediatly ***/
    //LIVO模式通过状态机管理处理流程，交替处理LIO（LiDAR-IMU）和VIO（视觉-IMU）数据

    EKF_STATE last_lio_vio_flg = meas.lio_vio_flg;
    // double t0 = omp_get_wtime();
    switch (last_lio_vio_flg)
    {
    // double img_capture_time = meas.lidar_frame_beg_time + exposure_time_init;
    case WAIT:
    case VIO:
    //根据图像捕获时间切割LiDAR点云，准备LIO处理
    {
      // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);
      // 计算图像捕获时间（LiDAR起始时间 + 曝光补偿）
      double img_capture_time = img_time_buffer.front() + exposure_time_init;
      /*** has img topic, but img topic timestamp larger than lidar end time,
       * process lidar topic. After LIO update, the meas.lidar_frame_end_time
       * will be refresh. ***/
      if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();
      // printf("[ Data Cut ] wait \n");
      // printf("[ Data Cut ] last_lio_update_time: %lf \n",
      // meas.last_lio_update_time);

      double lid_newest_time = lid_header_time_buffer.back() + lid_raw_data_buffer.back()->points.back().curvature / double(1000);
      double imu_newest_time = imu_buffer.back()->header.stamp.toSec();
    // 检查图像时间是否合理
      if (img_capture_time < meas.last_lio_update_time + 0.00001)
      {
        img_buffer.pop_front();// 丢弃过期图像
        img_time_buffer.pop_front();
        ROS_ERROR("[ Data Cut ] Throw one image frame! \n");
        return false;
      }

      if (img_capture_time > lid_newest_time || img_capture_time > imu_newest_time)
      {
        // ROS_ERROR("lost first camera frame");
        // printf("img_capture_time, lid_newest_time, imu_newest_time: %lf , %lf
        // , %lf \n", img_capture_time, lid_newest_time, imu_newest_time);
        return false;
      }

      // 收集IMU数据到LiDAR结束时间
      struct MeasureGroup m;

      // printf("[ Data Cut ] LIO \n");
      // printf("[ Data Cut ] img_capture_time: %lf \n", img_capture_time);
      m.imu.clear();
      m.lio_time = img_capture_time;
      mtx_buffer.lock();
      while (!imu_buffer.empty())
      {
        if (imu_buffer.front()->header.stamp.toSec() > m.lio_time) break;

        if (imu_buffer.front()->header.stamp.toSec() > meas.last_lio_update_time) m.imu.push_back(imu_buffer.front());

        imu_buffer.pop_front();
        // printf("[ Data Cut ] imu time: %lf \n",
        // imu_buffer.front()->header.stamp.toSec());
      }
      mtx_buffer.unlock();
      sig_buffer.notify_all();

      *(meas.pcl_proc_cur) = *(meas.pcl_proc_next);
      PointCloudXYZI().swap(*meas.pcl_proc_next);

      int lid_frame_num = lid_raw_data_buffer.size();
      int max_size = meas.pcl_proc_cur->size() + 24000 * lid_frame_num;
      meas.pcl_proc_cur->reserve(max_size);
      meas.pcl_proc_next->reserve(max_size);
      // deque<PointCloudXYZI::Ptr> lidar_buffer_tmp;
        // 分割LiDAR点云到当前帧和下一帧
      while (!lid_raw_data_buffer.empty())
      {
        if (lid_header_time_buffer.front() > img_capture_time) break;
        auto pcl(lid_raw_data_buffer.front()->points);
        double frame_header_time(lid_header_time_buffer.front());
        float max_offs_time_ms = (m.lio_time - frame_header_time) * 1000.0f;

        for (int i = 0; i < pcl.size(); i++)
        {
          auto pt = pcl[i];
          if (pcl[i].curvature < max_offs_time_ms)
          {
            pt.curvature += (frame_header_time - meas.last_lio_update_time) * 1000.0f;
            meas.pcl_proc_cur->points.push_back(pt);// 当前处理帧
          }
          else
          {
            pt.curvature += (frame_header_time - m.lio_time) * 1000.0f;
            meas.pcl_proc_next->points.push_back(pt);// 下一帧
          }
        }
        lid_raw_data_buffer.pop_front();
        lid_header_time_buffer.pop_front();
      }

      meas.measures.push_back(m);
      meas.lio_vio_flg = LIO;// 标记为LIO处理在下面的case中调用了
      // meas.last_lio_update_time = m.lio_time;
      // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);
      // printf("[ Data Cut ] pcl_proc_cur number: %d \n", meas.pcl_proc_cur
      // ->points.size()); printf("[ Data Cut ] LIO process time: %lf \n",
      // omp_get_wtime() - t0);
      return true;
    }

    case LIO://将图像数据与时间对齐的IMU数据打包 
    {
      double img_capture_time = img_time_buffer.front() + exposure_time_init;
      meas.lio_vio_flg = VIO;//改成vio补齐
      // printf("[ Data Cut ] VIO \n");
      meas.measures.clear();
      double imu_time = imu_buffer.front()->header.stamp.toSec();

      // 取出图像数据
      struct MeasureGroup m;
      m.vio_time = img_capture_time;
      m.lio_time = meas.last_lio_update_time;//统计时间
      m.img = img_buffer.front();//
      mtx_buffer.lock();
      // while ((!imu_buffer.empty() && (imu_time < img_capture_time)))
      // {
      //   imu_time = imu_buffer.front()->header.stamp.toSec();
      //   if (imu_time > img_capture_time) break;
      //   m.imu.push_back(imu_buffer.front());
      //   imu_buffer.pop_front();
      //   printf("[ Data Cut ] imu time: %lf \n",
      //   imu_buffer.front()->header.stamp.toSec());
      // }
      // 弹出已处理图像
      img_buffer.pop_front();
      img_time_buffer.pop_front();//返回第一个元素
      mtx_buffer.unlock();
      sig_buffer.notify_all();
      // 标记为VIO处理
      meas.measures.push_back(m);
      lidar_pushed = false; // 处理vio的第一帧
      //after VIO update, the _lidar_frame_end_time will be refresh.
      // printf("[ Data Cut ] VIO process time: %lf \n", omp_get_wtime() - t0);
      return true;
    }

    default:
    {
      // printf("!! WRONG EKF STATE !!");
      return false;
    }
      // return false;
    }
    break;
  }
  case ONLY_LO:
  {
    if (!lidar_pushed) 
    { 
      // If not in lidar scan, need to generate new meas
      if (lid_raw_data_buffer.empty())  return false;
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic
      meas.lidar_frame_beg_time = lid_header_time_buffer.front(); // generate lidar_beg_time
      meas.lidar_frame_end_time  = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
      lidar_pushed = true;             
    }
    struct MeasureGroup m; // standard method to keep imu message.
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    lidar_pushed = false; // sync one whole lidar scan.
    meas.lio_vio_flg = LO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    return true;
    break;
  }
  default://如果识别错误
  {
    printf("!! WRONG SLAM TYPE !!");
    return false;
  }
  }
  ROS_ERROR("out sync");
}


/*
后面全是发布相关函数
将VIO管理器中的RGB图像转换为ROS消息并发布
*/
void LIVMapper::publish_img_rgb(const image_transport::Publisher &pubImage, VIOManagerPtr vio_manager)//发布img——rgb图像
{
  /*
  使用ros::Time::now()，需确保与传感器数据时间同步
  BGR8符合OpenCV默认格式，但ROS中某些节点可能期望RGB8，需确认下游节点兼容性
  out_msg.header.frame_id被注释，若需要坐标系信息需取消注释

  */
  cv::Mat img_rgb = vio_manager->img_cp;//vio_manager获取拷贝的RGB图像img_cp
  cv_bridge::CvImage out_msg;
  out_msg.header.stamp = ros::Time::now();
  // out_msg.header.frame_id = "camera_init";
  out_msg.encoding = sensor_msgs::image_encodings::BGR8;//使用cv_bridge封装图像数据，设置时间戳和BGR8编码格式
  out_msg.image = img_rgb;
  pubImage.publish(out_msg.toImageMsg());//通过image_transport::Publisher发布图像消息
}
/*
发布带RGB颜色的世界坐标系点云
融合点云与图像颜色信息，发布彩色点云，并可选保存为PCD文件
*/
void LIVMapper::publish_frame_world(const ros::Publisher &pubLaserCloudFullRes, VIOManagerPtr vio_manager)
{
  if (pcl_w_wait_pub->empty()) return;
  PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB());
  if (img_en)//静态变量pub_num控制发布间隔，累积pub_scan_num次点云后处理
  {
    static int pub_num = 1;
    *pcl_wait_pub += *pcl_w_wait_pub;
    if(pub_num == pub_scan_num)
    {
      pub_num = 1;
      size_t size = pcl_wait_pub->points.size();//通过pcl_wait_pub临时存储多帧点云数据
      laserCloudWorldRGB->reserve(size);
      // double inv_expo = _state.inv_expo_time;
      cv::Mat img_rgb = vio_manager->img_rgb;
      for (size_t i = 0; i < size; i++)//遍历点云，将每个世界坐标系点p_w转换到相机坐标系pf和像素坐标pc
      {
        PointTypeRGB pointRGB;
        pointRGB.x = pcl_wait_pub->points[i].x;
        pointRGB.y = pcl_wait_pub->points[i].y;
        pointRGB.z = pcl_wait_pub->points[i].z;

        V3D p_w(pcl_wait_pub->points[i].x, pcl_wait_pub->points[i].y, pcl_wait_pub->points[i].z);
        V3D pf(vio_manager->new_frame_->w2f(p_w)); if (pf[2] < 0) continue;
        V2D pc(vio_manager->new_frame_->w2c(p_w));

        if (vio_manager->new_frame_->cam_->isInFrame(pc.cast<int>(), 3)) // 100
        {
          V3F pixel = vio_manager->getInterpolatedPixel(img_rgb, pc);//使用getInterpolatedPixel进行双线性插值获取颜色，确保像素在图像范围内
          pointRGB.r = pixel[2];
          pointRGB.g = pixel[1];
          pointRGB.b = pixel[0];
          // pointRGB.r = pixel[2] * inv_expo; pointRGB.g = pixel[1] * inv_expo; pointRGB.b = pixel[0] * inv_expo;
          // if (pointRGB.r > 255) pointRGB.r = 255;
          // else if (pointRGB.r < 0) pointRGB.r = 0;
          // if (pointRGB.g > 255) pointRGB.g = 255;
          // else if (pointRGB.g < 0) pointRGB.g = 0;
          // if (pointRGB.b > 255) pointRGB.b = 255;
          // else if (pointRGB.b < 0) pointRGB.b = 0;
          if (pf.norm() > blind_rgb_points) laserCloudWorldRGB->push_back(pointRGB);//过滤近距离点（blind_rgb_points阈值）
        }
      }
    }
    else
    {
      pub_num++;
    }
  }

  /*** Publish Frame ***/
  sensor_msgs::PointCloud2 laserCloudmsg;
  if (img_en)//根据img_en标志选择发布彩色或原始点云
  {
    // cout << "RGB pointcloud size: " << laserCloudWorldRGB->size() << endl;
    pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
  }
  else 
  { 
    pcl::toROSMsg(*pcl_w_wait_pub, laserCloudmsg); 
  }
  laserCloudmsg.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
  laserCloudmsg.header.frame_id = "camera_init";//设置时间戳和camera_init坐标系
  pubLaserCloudFullRes.publish(laserCloudmsg);

  /**************** save map ****************/
  /* 1. make sure you have enough memories
  /* 2. noted that pcd save will influence the real-time performences **/

  if (pcd_save_en)//点云地图保存模块
  {
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
    static int scan_wait_num = 0;
    // *pcl_wait_save += *laserCloudWorldRGB;
    if (img_en)//如果出现图像
    {
      *pcl_wait_save += *laserCloudWorldRGB;
    }
    else
    {
      *pcl_wait_save_intensity += *pcl_w_wait_pub;
    }
    scan_wait_num++;
    //按间隔保存点云到指定路径，记录位姿信息
    if ((pcl_wait_save->size() > 0 || pcl_wait_save_intensity->size() > 0) && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
    {
      pcd_index++;
      string all_points_dir(string(string(ROOT_DIR) + "Log/PCD/") + to_string(pcd_index) + string(".pcd"));
      pcl::PCDWriter pcd_writer;
      if (pcd_save_en)
      {
        cout << "current scan saved to /PCD/" << all_points_dir << endl;//
        if (img_en)
        {
          pcd_writer.writeBinary(all_points_dir, *pcl_wait_save); // pcl::io::savePCDFileASCII(all_points_dir, *pcl_wait_save);
          PointCloudXYZRGB().swap(*pcl_wait_save);
        }
        else
        {
          pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_intensity);
          PointCloudXYZI().swap(*pcl_wait_save_intensity);
        }  
        // pcd_writer.writeBinary(all_points_dir, *pcl_wait_save); // pcl::io::savePCDFileASCII(all_points_dir, *pcl_wait_save);
        // PointCloudXYZRGB().swap(*pcl_wait_save);
        Eigen::Quaterniond q(_state.rot_end);
        fout_pcd_pos << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " " << q.w() << " " << q.x() << " " << q.y()
                     << " " << q.z() << " " << endl;
        scan_wait_num = 0;
      }
    }
  }
  if(laserCloudWorldRGB->size() > 0)  PointCloudXYZI().swap(*pcl_wait_pub); 
  PointCloudXYZI().swap(*pcl_w_wait_pub);
}
/*
发布可视化子地图
发布当前构建的局部子地图点云
*/
void LIVMapper::publish_visual_sub_map(const ros::Publisher &pubSubVisualMap)//发布子图像
{
  PointCloudXYZI::Ptr laserCloudFullRes(visual_sub_map);//从visual_sub_map获取点云数据
  int size = laserCloudFullRes->points.size(); if (size == 0) return;

  PointCloudXYZI::Ptr sub_pcl_visual_map_pub(new PointCloudXYZI());//复制到临时点云对象sub_pcl_visual_map_pub
  *sub_pcl_visual_map_pub = *laserCloudFullRes;//*sub_pcl_visual_map_pub = *laserCloudFullRes可能导致性能瓶颈（点云较大时），建议改用共享指针或move语义
  if (1)//转换为ROS消息并发布，设置时间戳和坐标系
  {
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*sub_pcl_visual_map_pub, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time::now();
    laserCloudmsg.header.frame_id = "camera_init";
    pubSubVisualMap.publish(laserCloudmsg);
  }
}
/*
将匹配后的点云数据转换为ROS消息并发布
*/
void LIVMapper::publish_effect_world(const ros::Publisher &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list)//发布有效点云数据 (publish_effect_world)
{
  int effect_feat_num = ptpl_list.size();//从输入参数ptpl_list（点-平面关联列表）获取有效点数量
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effect_feat_num, 1));
  for (int i = 0; i < effect_feat_num; i++)//创建PointCloudXYZI类型的点云对象，并遍历ptpl_list填充点云数据
  {
    laserCloudWorld->points[i].x = ptpl_list[i].point_w_[0];
    laserCloudWorld->points[i].y = ptpl_list[i].point_w_[1];
    laserCloudWorld->points[i].z = ptpl_list[i].point_w_[2];
  }
  sensor_msgs::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = ros::Time::now();
  laserCloudFullRes3.header.frame_id = "camera_init";
  //将点云转换为ROS消息，设置时间戳和坐标系（camera_init），通过指定发布器pubLaserCloudEffect发布
  pubLaserCloudEffect.publish(laserCloudFullRes3);
}
/*
通用模板函数，用于设置不同ROS消息的位姿字段
*/
template <typename T> void LIVMapper::set_posestamp(T &out)//设置位姿信息的模板函数 (set_posestamp)
{
  out.position.x = _state.pos_end(0);
  out.position.y = _state.pos_end(1);
  out.position.z = _state.pos_end(2);//将内部状态_state.pos_end的位置赋值给目标消息的position
  out.orientation.x = geoQuat.x;
  out.orientation.y = geoQuat.y;
  out.orientation.z = geoQuat.z;
  out.orientation.w = geoQuat.w;//将四元数geoQuat赋值给目标消息的orientation
}
/*
发布处理后的里程计数据和对应的TF变换
*/
void LIVMapper::publish_odometry(const ros::Publisher &pubOdomAftMapped)//发布里程计
{
  odomAftMapped.header.frame_id = "camera_init";//设置里程计消息odomAftMapped的帧ID和时间戳
  odomAftMapped.child_frame_id = "aft_mapped";
  odomAftMapped.header.stamp = ros::Time::now(); //.ros::Time()fromSec(last_timestamp_lidar);
  set_posestamp(odomAftMapped.pose.pose);//调用set_posestamp填充位姿信息

  static tf::TransformBroadcaster br;//使用tf::TransformBroadcaster发布从camera_init到aft_mapped的坐标变换
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(tf::Vector3(_state.pos_end(0), _state.pos_end(1), _state.pos_end(2)));
  q.setW(geoQuat.w);
  q.setX(geoQuat.x);
  q.setY(geoQuat.y);
  q.setZ(geoQuat.z);
  transform.setRotation(q);
  br.sendTransform( tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "aft_mapped") );


  pubOdomAftMapped.publish(odomAftMapped);//发布里程计消息到指定发布器pubOdomAftMapped
}
/*
发布MAVROS位姿 (publish_mavros) 向MAVROS系统发布当前机体位姿
*/
void LIVMapper::publish_mavros(const ros::Publisher &mavros_pose_publisher)
{
  msg_body_pose.header.stamp = ros::Time::now();//设置消息头的时间戳和帧ID
  msg_body_pose.header.frame_id = "camera_init";
  set_posestamp(msg_body_pose.pose);
  mavros_pose_publisher.publish(msg_body_pose);//使用set_posestamp填充位姿，通过mavros_pose_publisher发布
}
/*
发布路径信息 (publish_path) 更新并发布机器人的运动路径
*/
void LIVMapper::publish_path(const ros::Publisher pubPath)
{
  set_posestamp(msg_body_pose.pose);
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_init";//更新msg_body_pose的位姿和时间戳
  path.poses.push_back(msg_body_pose);//将当前位姿添加到path的轨迹数组中
  pubPath.publish(path);//  发布更新后的路径到指定发布器pubPath
}