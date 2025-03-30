/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VOXEL_MAP_H_
#define VOXEL_MAP_H_

#include "common_lib.h"
#include <Eigen/Dense>
#include <fstream>
#include <math.h>
#include <mutex>
#include <omp.h>
#include <pcl/common/io.h>
#include <ros/ros.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#define VOXELMAP_HASH_P 116101
#define VOXELMAP_MAX_N 10000000000

static int voxel_plane_id = 0;

typedef struct VoxelMapConfig
{
  double max_voxel_size_;
  int max_layer_;
  int max_iterations_;
  std::vector<int> layer_init_num_;
  int max_points_num_;
  double planner_threshold_;
  double beam_err_;
  double dept_err_;
  double sigma_num_;
  bool is_pub_plane_map_;

  // config of local map sliding
  double sliding_thresh;
  bool map_sliding_en;
  int half_map_size;
} VoxelMapConfig;

typedef struct PointToPlane
{
  Eigen::Vector3d point_b_;
  Eigen::Vector3d point_w_;
  Eigen::Vector3d normal_;
  Eigen::Vector3d center_;
  Eigen::Matrix<double, 6, 6> plane_var_;
  M3D body_cov_;
  int layer_;
  double d_;
  double eigen_value_;
  bool is_valid_;
  float dis_to_plane_;
} PointToPlane;

typedef struct VoxelPlane
{
  Eigen::Vector3d center_;
  Eigen::Vector3d normal_;
  Eigen::Vector3d y_normal_;
  Eigen::Vector3d x_normal_;
  Eigen::Matrix3d covariance_;
  Eigen::Matrix<double, 6, 6> plane_var_;
  float radius_ = 0;
  float min_eigen_value_ = 1;
  float mid_eigen_value_ = 1;
  float max_eigen_value_ = 1;
  float d_ = 0;
  int points_size_ = 0;
  bool is_plane_ = false;
  bool is_init_ = false;
  int id_ = 0;
  bool is_update_ = false;
  VoxelPlane()
  {
    plane_var_ = Eigen::Matrix<double, 6, 6>::Zero();
    covariance_ = Eigen::Matrix3d::Zero();
    center_ = Eigen::Vector3d::Zero();
    normal_ = Eigen::Vector3d::Zero();
  }
} VoxelPlane;

class VOXEL_LOCATION
{
public:
  int64_t x, y, z;

  VOXEL_LOCATION(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOCATION &other) const { return (x == other.x && y == other.y && z == other.z); }
};

// Hash value
namespace std
{
template <> struct hash<VOXEL_LOCATION>
{
  int64_t operator()(const VOXEL_LOCATION &s) const
  {
    using std::hash;
    using std::size_t;
    return ((((s.z) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.y)) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.x);
  }
};
} // namespace std

struct DS_POINT
{
  float xyz[3];
  float intensity;
  int count = 0;
};

void calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &cov);
/*
VoxelOctoTree类
*/
class VoxelOctoTree
{

public:
  VoxelOctoTree() = default;
  std::vector<pointWithVar> temp_points_;   //temp_points_:存储当前体素内点的临时列表，每个点的类型为 pointWithVar，可能包含点的坐标及其协方差或不确定性信息

  VoxelPlane *plane_ptr_;
  //指向 VoxelPlane 对象的指针，表示当前体素内拟合的平面。
  //平面通常由其法向量和平面上的一个点定义

  int layer_;//当前节点在八叉树中的深度。根节点的 layer_ = 0，随着树的细分，深度逐渐增加
  int octo_state_; // 0 is end of tree, 1 is not
  //节点状态的标志：
//0：当前节点是叶子节点（树的末端）。
//1：当前节点有子节点（非叶子节点）。
  VoxelOctoTree *leaves_[8];//leaves_[8]:包含 8 个子节点指针的数组，每个子节点对应当前体素的 8 个八分体之一
  double voxel_center_[3]; // x, y, zvoxel_center_[3]:当前体素中心的三维坐标（x, y, z）
  std::vector<int> layer_init_num_;//存储每一层八叉树节点初始化平面所需的最小点数
  float quater_length_;//当前体素边长的一半，用于确定子体素的大小
  float planer_threshold_;//用于判断体素内的点是否可以拟合为一个平面的阈值。如果拟合误差低于此阈值，则初始化平面
  int points_size_threshold_;//触发体素细分或平面拟合所需的最小点数
  int update_size_threshold_;//触发平面或体素更新所需的新点数
  int max_points_num_;//体素在细分前允许的最大点数
  int max_layer_;//八叉树的最大深度。达到此深度后，树停止细分
  int new_points_;//自上次更新以来添加到体素中的新点数量
  bool init_octo_;//标志位，表示八叉树节点是否已初始化
  bool update_enable_;//标志位，表示体素是否可以更新（例如，平面拟合或细分）

  VoxelOctoTree(int max_layer, int layer, int points_size_threshold, int max_points_num, float planer_threshold)
      : max_layer_(max_layer), layer_(layer), points_size_threshold_(points_size_threshold), max_points_num_(max_points_num),
        planer_threshold_(planer_threshold)
        /*
        使用参数（如 max_layer、layer、points_size_threshold、max_points_num 和 planer_threshold）初始化八叉树节点
        */
  {
    temp_points_.clear();//清空临时点列表 temp_points_，确保节点初始时没有点
    octo_state_ = 0;//将节点状态 octo_state_ 设置为 0，表示当前节点是叶子节点（尚未细分）
    new_points_ = 0;//将新点计数器 new_points_ 初始化为 0
    update_size_threshold_ = 5;//设置更新阈值 update_size_threshold_ 为 5，表示当新点数量达到 5 时，触发节点更新
    init_octo_ = false;//将初始化标志 init_octo_ 设置为 false，表示节点尚未初始化
    update_enable_ = true;//将更新使能标志 update_enable_ 设置为 true，表示节点可以更新
    for (int i = 0; i < 8; i++)
    {
      leaves_[i] = nullptr;//使用 for 循环将 8 个子节点指针 leaves_[i] 初始化为 nullptr，表示当前节点没有子节点
    }
    plane_ptr_ = new VoxelPlane;//动态分配一个新的 VoxelPlane 对象，并将其指针赋值给 plane_ptr_，用于存储当前节点内的平面信息
  }
/*
使用 for 循环遍历 8 个子节点指针 leaves_[i]。
如果子节点指针不为 nullptr，则调用 delete 释放其内存。
由于 VoxelOctoTree 是一个递归结构，子节点的析构函数也会被调用，从而递归释放整个子树的内存
tree
├── leaves_[0] -> VoxelOctoTree
├── leaves_[1] -> nullptr
├── leaves_[2] -> VoxelOctoTree
│   ├── leaves_[0] -> nullptr
│   ├── leaves_[1] -> nullptr
│   └── ...
├── leaves_[3] -> nullptr
└── ...
当 tree 被销毁时：
析构函数首先遍历 leaves_ 数组。对于非空的子节点（如 leaves_[0] 和 leaves_[2]），调用它们的析构函数。子节点的析构函数会继续递归释放它们的子节点最后，释放 plane_ptr_ 的内存
*/
  ~VoxelOctoTree()
  {
    for (int i = 0; i < 8; i++)
    {
      delete leaves_[i];
    }
    delete plane_ptr_;
  }
  //成员函数
  void init_plane(const std::vector<pointWithVar> &points, VoxelPlane *plane);//使用 temp_points_ 中的点初始化平面（VoxelPlane）
  void init_octo_tree();//初始化八叉树结构。如果当前体素中的点数超过 points_size_threshold_，则将其细分为 8 个子体素
  void cut_octo_tree();//将当前体素细分为 8 个子体素，并将点分配到相应的子体素中
  void UpdateOctoTree(const pointWithVar &pv);//向八叉树中插入一个新点（pv）。如果新点的数量超过 update_size_threshold_，则更新平面或体素
  /*
  指针类型的
  */
  VoxelOctoTree *find_correspond(Eigen::Vector3d pw);//查找与给定三维点（pw）对应的体素（或叶子节点）。用于定位插入或查询点的合适体素
  VoxelOctoTree *Insert(const pointWithVar &pv);//向八叉树中插入一个新点（pv）。如果体素中的点数超过 max_points_num_，则细分并重新分配点
};

void loadVoxelConfig(ros::NodeHandle &nh, VoxelMapConfig &voxel_config);




/*
管理 LiDAR 点云数据的体素地图系统的类
*/
class VoxelMapManager
{
public:
  VoxelMapManager() = default;
  VoxelMapConfig config_setting_;//体素地图的配置参数，类型为 VoxelMapConfig，包含体素大小、最大深度、平面拟合阈值等配置
  int current_frame_id_ = 0;//当前帧的 ID，用于跟踪处理的帧数
  ros::Publisher voxel_map_pub_;//ROS 发布器，用于发布体素地图数据
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map_;//体素地图的核心数据结构，使用 std::unordered_map 存储体素位置（VOXEL_LOCATION）到 VoxelOctoTree 指针的映射

  PointCloudXYZI::Ptr feats_undistort_;//去畸变后的点云数据，类型为 PointCloudXYZI::Ptr
  PointCloudXYZI::Ptr feats_down_body_;//降采样后的 LiDAR 点云数据（在机体坐标系下），类型为 PointCloudXYZI::Ptr
  PointCloudXYZI::Ptr feats_down_world_;//降采样后的 LiDAR 点云数据（在世界坐标系下），类型为 PointCloudXYZI::Ptr

  M3D extR_;
  V3D extT_;//外部旋转矩阵（extR_）和平移向量（extT_），用于将 LiDAR 点从 LiDAR 坐标系转换到机体坐标系
  float build_residual_time, ekf_time;//分别记录构建残差和扩展卡尔曼滤波（EKF）的时间
  float ave_build_residual_time = 0.0;
  float ave_ekf_time = 0.0;//分别记录构建残差和 EKF 的平均时间
  int scan_count = 0;//扫描次数计数器，用于统计处理的总帧数
  StatesGroup state_;//当前的状态估计，类型为 StatesGroup，可能包含位置、姿态、速度等信息
  V3D position_last_;//上一帧的位置，用于计算位移

  V3D last_slide_position = {0,0,0};//上一次滑动窗口的位置，用于地图滑动操作

  geometry_msgs::Quaternion geoQuat_;//当前姿态的四元数表示，用于发布可视化消息

  int feats_down_size_;//降采样后的点云数量
  int effct_feat_num_;//有效特征点的数量，用于状态估计
  std::vector<M3D> cross_mat_list_;//存储点的反对称矩阵列表
  std::vector<M3D> body_cov_list_;//存储每个点的协方差矩阵列表
  std::vector<pointWithVar> pv_list_;//存储带协方差的点列表
  std::vector<PointToPlane> ptpl_list_;//存储点-平面匹配结果的列表

  VoxelMapManager(VoxelMapConfig &config_setting, std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &voxel_map)
      : config_setting_(config_setting), voxel_map_(voxel_map)
      /*
      初始化 VoxelMapManager，接受 VoxelMapConfig 和 voxel_map_ 作为参数。
      初始化点云指针和当前帧 ID
      */
  {
    current_frame_id_ = 0;

    feats_undistort_.reset(new PointCloudXYZI());
    feats_down_body_.reset(new PointCloudXYZI());
    feats_down_world_.reset(new PointCloudXYZI());
  };

  void StateEstimation(StatesGroup &state_propagat);//状态估计函数，使用 LiDAR 点云和 EKF 进行状态估计
  void TransformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud);//将 LiDAR 点云从机体坐标系转换到世界坐标系

  void BuildVoxelMap();//构建体素地图，将点云数据分配到体素中
  V3F RGBFromVoxel(const V3D &input_point);//根据点的位置生成 RGB 颜色，用于可视化

  void UpdateVoxelMap(const std::vector<pointWithVar> &input_points);//更新体素地图，插入新的点并更新体素内的平面

  void BuildResidualListOMP(std::vector<pointWithVar> &pv_list, std::vector<PointToPlane> &ptpl_list);//使用 OpenMP 并行计算点-平面匹配的残差

  void build_single_residual(pointWithVar &pv, const VoxelOctoTree *current_octo, const int current_layer, bool &is_sucess, double &prob,
                             PointToPlane &single_ptpl);//计算单个点与体素内平面的残差

  void pubVoxelMap();//发布体素地图数据，用于可视化

  void mapSliding();//滑动地图窗口，移除超出范围的点云数据
  void clearMemOutOfMap(const int& x_max,const int& x_min,const int& y_max,const int& y_min,const int& z_max,const int& z_min );//清除超出地图范围的内存

private:
  void GetUpdatePlane(const VoxelOctoTree *current_octo, const int pub_max_voxel_layer, std::vector<VoxelPlane> &plane_list);//获取需要更新的平面列表

  void pubSinglePlane(visualization_msgs::MarkerArray &plane_pub, const std::string plane_ns, const VoxelPlane &single_plane, const float alpha,
                      const Eigen::Vector3d rgb);//发布单个平面的可视化消息
  void CalcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec, const Eigen::Vector3d &z_vec, geometry_msgs::Quaternion &q);//计算四元数，用于表示平面的姿态

  void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b);//根据值的大小生成 RGB 颜色，用于可视化
};
typedef std::shared_ptr<VoxelMapManager> VoxelMapManagerPtr;

#endif // VOXEL_MAP_H_