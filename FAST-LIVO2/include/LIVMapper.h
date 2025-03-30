/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIV_MAPPER_H
#define LIV_MAPPER_H

#include "IMU_Processing.h"
#include "vio.h"
#include "preprocess.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Path.h>
#include <vikit/camera_loader.h>

class LIVMapper
{
public:
  LIVMapper(ros::NodeHandle &nh);//构造函数，初始化 ROS 节点和参数
  ~LIVMapper();
  void initializeSubscribersAndPublishers(ros::NodeHandle &nh, image_transport::ImageTransport &it);//初始化 ROS 订阅器和发布器
  void initializeComponents();//初始化 SLAM 系统的各个组件（如预处理、IMU 处理、VIO 管理等）
  void initializeFiles();//初始化文件输出（如日志文件、点云文件等）

  //主操作
  void run();//主循环函数
  void gravityAlignment();//重力对齐，用于初始化 IMU 的姿态
  void handleFirstFrame();//处理第一帧数据，初始化系统状态
  void stateEstimationAndMapping();//状态估计和地图构建的核心函数
  void handleVIO();//处理视觉-惯性里程计（VIO）数据
  void handleLIO();//处理 LiDAR-惯性里程计（LIO）数据
  void savePCD();
  void processImu();
  
  //数据处理
  bool sync_packages(LidarMeasureGroup &meas);//同步 LiDAR、IMU 和图像数据
  void prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr);//使用 IMU 数据进行状态传播
  void imu_prop_callback(const ros::TimerEvent &e);
  void transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud);//将 LiDAR 点云从机体坐标系转换到世界坐标系
  void pointBodyToWorld(const PointType &pi, PointType &po);//将点从机体坐标系转换到世界坐标系
  void RGBpointBodyToWorld(PointType const *const pi, PointType *const po);//将带 RGB 信息的点从机体坐标系转换到世界坐标系


  //回调函数
  void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg);//标准 LiDAR 点云回调函数
  void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg_in);//Livox LiDAR 点云回调函数
  void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in);//IMU 数据回调函数
  void img_cbk(const sensor_msgs::ImageConstPtr &msg_in);//图像数据回调函数

  //发布函数
  void publish_img_rgb(const image_transport::Publisher &pubImage, VIOManagerPtr vio_manager);//发布 RGB 图像
  void publish_frame_world(const ros::Publisher &pubLaserCloudFullRes, VIOManagerPtr vio_manager);//发布世界坐标系下的点云
  void publish_visual_sub_map(const ros::Publisher &pubSubVisualMap);
  void publish_effect_world(const ros::Publisher &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list);//发布有效的点云数据
  void publish_odometry(const ros::Publisher &pubOdomAftMapped);//发布里程计信息
  void publish_mavros(const ros::Publisher &mavros_pose_publisher);
  void publish_path(const ros::Publisher pubPath);//发布路径信息

  //工具函数
  void readParameters(ros::NodeHandle &nh);//从 ROS 参数服务器读取参数
  template <typename T> void set_posestamp(T &out);//设置位姿信息
  template <typename T> void pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi, Eigen::Matrix<T, 3, 1> &po);//从 ROS 图像消息中提取 OpenCV 图像
  template <typename T> Eigen::Matrix<T, 3, 1> pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi);
  cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);

  std::mutex mtx_buffer, mtx_buffer_imu_prop;//互斥锁，用于保护共享数据的访问 保护 LiDAR、IMU 和图像数据的缓冲区      保护 IMU 传播相关的数据
  std::condition_variable sig_buffer;//条件变量，用于线程间的同步，例如等待新数据到达

  SLAM_MODE slam_mode_;//SLAM 模式，可能包括 LiDAR-only、LiDAR-Inertial、LiDAR-Visual-Inertial 等
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map;//体素地图，使用哈希表存储体素位置到 VoxelOctoTree 指针的映射
  /*
  
  */


  string root_dir;//根目录，用于存储日志、点云等文件
  string lid_topic, imu_topic, seq_name, img_topic;//LiDAR、IMU 和图像数据的 ROS 话题名称
  V3D extT;
  M3D extR;//LiDAR 到 IMU 的外部平移和旋转矩阵

  int feats_down_size = 0, max_iterations = 0;//降采样后的点云数量      状态估计的最大迭代次数

  double res_mean_last = 0.05;//上一次的残差均值，用于状态估计的收敛判断
  double gyr_cov = 0, acc_cov = 0, inv_expo_cov = 0;//IMU 陀螺仪和加速度计的协方差，以及逆曝光协方差
  double blind_rgb_points = 0.0;//盲区 RGB 点的阈值
  double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0, last_timestamp_img = -1.0;//上一次 LiDAR、IMU 和图像数据的时间戳
  double filter_size_surf_min = 0;//点云降采样的最小体素大小
  double filter_size_pcd = 0;//点云保存时的体素大小
  double _first_lidar_time = 0.0;//第一帧 LiDAR 数据的时间戳
  double match_time = 0, solve_time = 0, solve_const_H_time = 0;//匹配、求解和求解常数 H 矩阵的时间

/*
标志与配置
lidar_map_inited：LiDAR 地图是否已初始化。
pcd_save_en：是否启用点云保存。
pub_effect_point_en：是否发布有效点云。
pose_output_en：是否输出位姿信息。
ros_driver_fix_en：是否启用 ROS 驱动修复
*/
  bool lidar_map_inited = false, pcd_save_en = false, pub_effect_point_en = false, pose_output_en = true, ros_driver_fix_en = false;
  int pcd_save_interval = -1, pcd_index = 0;//点云保存的间隔      点云保存的索引
  int pub_scan_num = 1;//发布的点云帧数


/*
imu相关的
*/
  StatesGroup imu_propagate, latest_ekf_state;//IMU 传播状态      最新的 EKF 状态

/*
new_imu：是否有新的 IMU 数据。
state_update_flg：状态更新标志。
imu_prop_enable：是否启用 IMU 传播。
ekf_finish_once：EKF 是否已完成一次
*/
  bool new_imu = false, state_update_flg = false, imu_prop_enable = true, ekf_finish_once = false;
  deque<sensor_msgs::Imu> prop_imu_buffer;//IMU 传播数据的缓冲区
  sensor_msgs::Imu newest_imu;//最新的 IMU 数据
  double latest_ekf_time;//最新的 EKF 时间戳
  nav_msgs::Odometry imu_prop_odom;//IMU 传播的里程计信息
  ros::Publisher pubImuPropOdom;//发布 IMU 传播的里程计信息
  double imu_time_offset = 0.0;//IMU 时间戳的偏移量

  bool gravity_align_en = false, gravity_align_finished = false;//gravity_align_en：是否启用重力对齐。gravity_align_finished：重力对齐是否已完成

  bool sync_jump_flag = false;//同步跳转标志


/*
lidar_pushed：LiDAR 数据是否已推送。
imu_en：是否启用 IMU。
gravity_est_en：是否启用重力估计。
flg_reset：是否重置系统。
ba_bg_est_en：是否启用加速度计和陀螺仪偏差估计
*/
  bool lidar_pushed = false, imu_en, gravity_est_en, flg_reset = false, ba_bg_est_en = true;
  bool dense_map_en = false;//是否启用稠密地图
  int img_en = 1, imu_int_frame = 3;//是否启用图像数据        IMU 数据插值的帧数
  bool normal_en = true;//是否启用法向量计算
  bool exposure_estimate_en = false;//是否启用曝光估计
  double exposure_time_init = 0.0;//初始曝光时间
  bool inverse_composition_en = false;//是否启用逆合成
  bool raycast_en = false;//是否启用射线投射
  int lidar_en = 1;//是否启用 LiDAR
  bool is_first_frame = false;//是否为第一帧数据


  /*
  图像与 LiDAR 数据
  */
  int grid_size, patch_size, grid_n_width, grid_n_height, patch_pyrimid_level;//图像网格和 patch 的大小
  double outlier_threshold;//异常值阈值
  double plot_time;//绘图时间
  int frame_cnt;//帧计数器
  double img_time_offset = 0.0;//图像时间戳的偏移量
  deque<PointCloudXYZI::Ptr> lid_raw_data_buffer;//原始 LiDAR 点云数据的缓冲区
  deque<double> lid_header_time_buffer;//LiDAR 数据的时间戳缓冲区
  deque<sensor_msgs::Imu::ConstPtr> imu_buffer;//IMU 数据的缓冲区
  deque<cv::Mat> img_buffer;//图像数据的缓冲区
  deque<double> img_time_buffer;//图像数据的时间戳缓冲区
  vector<pointWithVar> _pv_list;//带协方差的点列表
  vector<double> extrinT;
  vector<double> extrinR;
  vector<double> cameraextrinT;
  vector<double> cameraextrinR;//外部平移和旋转参数
  double IMG_POINT_COV;//图像点的协方差


/*
点云与地图
*/
  PointCloudXYZI::Ptr visual_sub_map;//可视化子地图
  PointCloudXYZI::Ptr feats_undistort;
  PointCloudXYZI::Ptr feats_down_body;
  PointCloudXYZI::Ptr feats_down_world;//去畸变、降采样后的点云
  PointCloudXYZI::Ptr pcl_w_wait_pub;
  PointCloudXYZI::Ptr pcl_wait_pub;//等待发布的点云
  PointCloudXYZRGB::Ptr pcl_wait_save;//等待保存的点云
  PointCloudXYZI::Ptr pcl_wait_save_intensity;

  ofstream fout_pre, fout_out, fout_pcd_pos, fout_points;//文件输出流，用于保存预处理数据、输出数据、点云位置和点云数据

/*
其他
*/
  pcl::VoxelGrid<PointType> downSizeFilterSurf;//点云降采样滤波器

  V3D euler_cur;//当前的欧拉角

  LidarMeasureGroup LidarMeasures;//LiDAR 测量数据组
  StatesGroup _state;
  StatesGroup  state_propagat;//当前状态和传播状态

  nav_msgs::Path path;//路径信息
  nav_msgs::Odometry odomAftMapped;//映射后的里程计信息
  geometry_msgs::Quaternion geoQuat;//当前姿态的四元数表示
  geometry_msgs::PoseStamped msg_body_pose;//机体位姿信息


/*
ros相关
*/
  PreprocessPtr p_pre;//点云预处理器
  ImuProcessPtr p_imu;//IMU 处理器
  VoxelMapManagerPtr voxelmap_manager;//体素地图管理器
  VIOManagerPtr vio_manager;//视觉-惯性里程计管理器

  ros::Publisher plane_pub;
  ros::Publisher voxel_pub;//发布平面和体素地图
  ros::Subscriber sub_pcl;
  ros::Subscriber sub_imu;
  ros::Subscriber sub_img;//订阅 LiDAR、IMU 和图像数据
  ros::Publisher pubLaserCloudFullRes;
  ros::Publisher pubNormal;
  ros::Publisher pubSubVisualMap;
  ros::Publisher pubLaserCloudEffect;
  ros::Publisher pubLaserCloudMap;
  ros::Publisher pubOdomAftMapped;
  ros::Publisher pubPath;
  ros::Publisher pubLaserCloudDyn;
  ros::Publisher pubLaserCloudDynRmed;
  ros::Publisher pubLaserCloudDynDbg;//发布点云、法向量、子地图、有效点云、地图、里程计、路径、动态点云等信息
  image_transport::Publisher pubImage;//发布图像数据
  ros::Publisher mavros_pose_publisher;//发布 MAVROS 位姿信息
  ros::Timer imu_prop_timer;//IMU 传播定时器


/*
性能统计
*/
  int frame_num = 0;//帧数计数器
  double aver_time_consu = 0;
  double aver_time_icp = 0;
  double aver_time_map_inre = 0;//平均时间消耗、ICP 时间和地图更新时间
  bool colmap_output_en = false;//是否启用 COLMAP 输出
};
#endif