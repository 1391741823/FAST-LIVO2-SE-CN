/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "voxel_map.h"
/*
协方差计算函数
*/
void calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &cov)//计算点 pb 的协方差矩阵 cov，用于描述该点在三维空间中的不确定性
{
  if (pb[2] == 0) pb[2] = 0.0001;//如果点的 z 坐标为 0，将其设置为一个极小值（0.0001），避免计算错误

  float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);//计算点到原点的距离 range

  float range_var = range_inc * range_inc;//根据距离增量 range_inc 计算距离方差 range_var

  Eigen::Matrix2d direction_var;//根据角度增量 degree_inc 计算方向方差 direction_var

  direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0, pow(sin(DEG2RAD(degree_inc)), 2);

  Eigen::Vector3d direction(pb);//计算点的单位方向向量 direction
  direction.normalize();//法向量

  Eigen::Matrix3d direction_hat;//构造方向向量的反对称矩阵 direction_hat
  direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;

  Eigen::Vector3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));//计算两个正交基向量 base_vector1 和 base_vector2，用于描述方向的不确定性
  base_vector1.normalize();
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();

  Eigen::Matrix<double, 3, 2> N;//构造投影矩阵 N 和 A，用于将方向方差投影到三维空间
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2), base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  //最终协方差矩阵 cov 由距离方差和方向方差组成
  cov = direction * range_var * direction.transpose() + A * direction_var * A.transpose();
}
/*
发布相关配置参数
*/
void loadVoxelConfig(ros::NodeHandle &nh, VoxelMapConfig &voxel_config)//从 ROS 参数服务器加载体素地图的配置参数
{
  nh.param<bool>("publish/pub_plane_en", voxel_config.is_pub_plane_map_, false);//加载是否发布平面地图的参数  在LIVMapp中使用 看是否调用 这里设置的是0
  
  nh.param<int>("lio/max_layer", voxel_config.max_layer_, 1);
  nh.param<double>("lio/voxel_size", voxel_config.max_voxel_size_, 0.5);
  nh.param<double>("lio/min_eigen_value", voxel_config.planner_threshold_, 0.01);//加载体素地图的最大层数、体素大小、特征值阈值等参数
  
  nh.param<double>("lio/sigma_num", voxel_config.sigma_num_, 3);
  nh.param<double>("lio/beam_err", voxel_config.beam_err_, 0.02);
  nh.param<double>("lio/dept_err", voxel_config.dept_err_, 0.05);//加载光束误差和深度误差参数
  
  nh.param<vector<int>>("lio/layer_init_num", voxel_config.layer_init_num_, vector<int>{5,5,5,5,5});
  nh.param<int>("lio/max_points_num", voxel_config.max_points_num_, 50);
  nh.param<int>("lio/max_iterations", voxel_config.max_iterations_, 5);//加载每层的初始化点数和最大迭代次数

  nh.param<bool>("local_map/map_sliding_en", voxel_config.map_sliding_en, false);
  nh.param<int>("local_map/half_map_size", voxel_config.half_map_size, 100);
  nh.param<double>("local_map/sliding_thresh", voxel_config.sliding_thresh, 8);//加载局部地图的滑动窗口配置
}
/*
初始化体素平面
根据输入的点云数据初始化一个平面。它通过计算点云的协方差矩阵、中心点和法向量，判断这些点是否构成一个平面，并初始化平面相关的属性
初始输入的参数是平面点 和指针平面
*/
void VoxelOctoTree::init_plane(const std::vector<pointWithVar> &points, VoxelPlane *plane)//根据输入的点云数据初始化一个平面
{
  plane->plane_var_ = Eigen::Matrix<double, 6, 6>::Zero();
  plane->covariance_ = Eigen::Matrix3d::Zero();
  plane->center_ = Eigen::Vector3d::Zero();
  plane->normal_ = Eigen::Vector3d::Zero();//清零平面的协方差矩阵、中心点、法向量等变量

  plane->points_size_ = points.size();
  plane->radius_ = 0;
  for (auto pv : points)//输入每个点 协方差矩阵计算
  {
    plane->covariance_ += pv.point_w * pv.point_w.transpose();//协方差
    plane->center_ += pv.point_w;//中心点
  }
  plane->center_ = plane->center_ / plane->points_size_;
  plane->covariance_ = plane->covariance_ / plane->points_size_ - plane->center_ * plane->center_.transpose();//遍历所有点，计算中心点和协方差矩阵

  Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance_);
  Eigen::Matrix3cd evecs = es.eigenvectors();
  Eigen::Vector3cd evals = es.eigenvalues();//对协方差矩阵进行特征值分解，获取特征值和特征向量

  Eigen::Vector3d evalsReal;
  evalsReal = evals.real();//特征值排序逻辑
  //对特征值进行排序，找到最小、中间和最大的特征值及其对应的特征向量
  Eigen::Matrix3f::Index evalsMin, evalsMax;
  evalsReal.rowwise().sum().minCoeff(&evalsMin);
  evalsReal.rowwise().sum().maxCoeff(&evalsMax);
  int evalsMid = 3 - evalsMin - evalsMax;
  Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
  Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
  Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
  Eigen::Matrix3d J_Q;
  J_Q << 1.0 / plane->points_size_, 0, 0, 0, 1.0 / plane->points_size_, 0, 0, 0, 1.0 / plane->points_size_;
  // && evalsReal(evalsMid) > 0.05
  //&& evalsReal(evalsMid) > 0.01
  //判断是否为平面
  if (evalsReal(evalsMin) < planer_threshold_)//如果最小特征值小于阈值 planer_threshold_，则认为这些点构成一个平面 标记为平面
  {
    for (int i = 0; i < points.size(); i++)
    {//雅可比矩阵计算
      Eigen::Matrix<double, 6, 3> J;
      Eigen::Matrix3d F;
      for (int m = 0; m < 3; m++)
      {
        if (m != (int)evalsMin)
        {
          Eigen::Matrix<double, 1, 3> F_m =
              (points[i].point_w - plane->center_).transpose() / ((plane->points_size_) * (evalsReal[evalsMin] - evalsReal[m])) *
              (evecs.real().col(m) * evecs.real().col(evalsMin).transpose() + evecs.real().col(evalsMin) * evecs.real().col(m).transpose());
          F.row(m) = F_m;
        }
        else
        {
          Eigen::Matrix<double, 1, 3> F_m;
          F_m << 0, 0, 0;
          F.row(m) = F_m;
        }
      }// F 的填充逻辑
      J.block<3, 3>(0, 0) = evecs.real() * F;
      J.block<3, 3>(3, 0) = J_Q;
      plane->plane_var_ += J * points[i].var * J.transpose();
    }

    plane->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
    plane->y_normal_ << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid), evecs.real()(2, evalsMid);
    plane->x_normal_ << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax);
    plane->min_eigen_value_ = evalsReal(evalsMin);
    plane->mid_eigen_value_ = evalsReal(evalsMid);
    plane->max_eigen_value_ = evalsReal(evalsMax);
    plane->radius_ = sqrt(evalsReal(evalsMax));// 半径计算
    plane->d_ = -(plane->normal_(0) * plane->center_(0) + plane->normal_(1) * plane->center_(1) + plane->normal_(2) * plane->center_(2));
    plane->is_plane_ = true;
    plane->is_update_ = true;
    if (!plane->is_init_)
    {
      plane->id_ = voxel_plane_id;
      voxel_plane_id++;
      plane->is_init_ = true;
    }
  }
  else
  {
    plane->is_update_ = true;
    plane->is_plane_ = false;//不是平面
  }
}
/*
初始化八叉树节点，判断当前节点是否构成平面，并根据结果决定是否继续分割
八叉树构建和管理的核心部分，用于处理点云数据并将其组织成层次化的八叉树结构
无输入
*/
void VoxelOctoTree::init_octo_tree()//
{
  if (temp_points_.size() > points_size_threshold_)//如果当前节点的点数 temp_points_ 超过阈值 points_size_threshold_，则进行平面初始化  检查当前八叉树节点内的点云数量是否超过预设阈值
  {
    init_plane(temp_points_, plane_ptr_);//对当前节点的点云进行平面拟合，判断是否构成平面
    if (plane_ptr_->is_plane_ == true)
    {
      octo_state_ = 0;//设置节点状态为平面 (octo_state_ = 0)
      // new added
      if (temp_points_.size() > max_points_num_)//如果 plane_ptr_->is_plane_ 为 true，表示当前节点构成平面，设置节点状态为 0
      {//并检查点数是否超过最大点数 max_points_num_。如果超过，则禁用更新并清空点云
        update_enable_ = false;
        std::vector<pointWithVar>().swap(temp_points_);
        new_points_ = 0;
      }
    }
    else//如果当前节点不构成平面，设置节点状态为 1，并调用 cut_octo_tree 函数继续分割
    {
      octo_state_ = 1;//设置节点状态（是否分割）
      cut_octo_tree();//使用下一个构建的函数
    }
    init_octo_ = true;
    new_points_ = 0;//设置 init_octo_ 为 true，并重置 new_points_ 为 0
  }
}
/*
对当前八叉树节点进行分割，将点云数据分配到子节点中，并递归地对每个子节点进行平面检测和进一步分割。
它是八叉树构建和管理的核心部分，用于处理点云数据并将其组织成层次化的八叉树结构。
*/
void VoxelOctoTree::cut_octo_tree()//在上一个函数中引用的 当前节点不构成平面时 使用此函数
{
  if (layer_ >= max_layer_)
  {
    octo_state_ = 0;// 标记为叶子节点
    return;// 停止分割
  }//如果当前层数 layer_ 超过最大层数 max_layer_，则停止分割并返回
  for (size_t i = 0; i < temp_points_.size(); i++)//遍历所有点，根据点的坐标将其分配到对应的子节点
  //根据点的坐标将点分配到 8 个子节点中的一个
  { 
    // 计算点相对于体素中心的位置
    int xyz[3] = {0, 0, 0};
    if (temp_points_[i].point_w[0] > voxel_center_[0]) { xyz[0] = 1; }
    if (temp_points_[i].point_w[1] > voxel_center_[1]) { xyz[1] = 1; }
    if (temp_points_[i].point_w[2] > voxel_center_[2]) { xyz[2] = 1; }
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
/*
通过比较点坐标与体素中心 voxel_center_ 的位置，确定子节点索引 leafnum
若子节点未创建，初始化其中心坐标和边长（quater_length_ 为父节点边长的 1/4，子节点边长为父节点的 1/2）
*/
    if (leaves_[leafnum] == nullptr)//如果子节点不存在，则创建新的子节点，并设置其中心点和边长
    {
      leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
      leaves_[leafnum]->layer_init_num_ = layer_init_num_;
      leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
      leaves_[leafnum]->quater_length_ = quater_length_ / 2;
    }
    // 将点加入子节点
    leaves_[leafnum]->temp_points_.push_back(temp_points_[i]);
    leaves_[leafnum]->new_points_++;
  }
  //子节点平面检测与递归分割
  for (uint i = 0; i < 8; i++)
  {
    if (leaves_[i] != nullptr)//如果子节点的点数超过阈值，则初始化子节点并判断是否为平面
    {
      if (leaves_[i]->temp_points_.size() > leaves_[i]->points_size_threshold_)
      {
        init_plane(leaves_[i]->temp_points_, leaves_[i]->plane_ptr_);// 平面检测
        if (leaves_[i]->plane_ptr_->is_plane_)
        {
          leaves_[i]->octo_state_ = 0;// 标记为平面
          // new added
          if (leaves_[i]->temp_points_.size() > leaves_[i]->max_points_num_)
          {
            leaves_[i]->update_enable_ = false;// 禁用更新
            std::vector<pointWithVar>().swap(leaves_[i]->temp_points_);// 清空点云
            new_points_ = 0;
          }
        }
        else
        {
          leaves_[i]->octo_state_ = 1;// 标记需继续分割
          leaves_[i]->cut_octo_tree();// 递归分割 调用函数本身区执行
        }
        leaves_[i]->init_octo_ = true;// 标记已初始化
        leaves_[i]->new_points_ = 0;// 重置新点计数器
      }
    }
  }
}
/*
更新八叉树  结合上面两个函数
主要功能是更新八叉树节点，将新点插入到合适的节点中，并根据条件重新初始化或分割节点
const pointWithVar &pv: 输入的新点，包含点的三维坐标和协方差等信息
*/
void VoxelOctoTree::UpdateOctoTree(const pointWithVar &pv)
{
  if (!init_octo_)//如果节点未初始化，则将新点加入 temp_points_，并检查是否需要初始化
  {
    new_points_++;
    temp_points_.push_back(pv);
    if (temp_points_.size() > points_size_threshold_) { init_octo_tree(); }
  }
  else
  {
    if (plane_ptr_->is_plane_)//如果当前节点是平面且允许更新，则将新点加入 temp_points_，并检查是否需要重新初始化平面或清空点云
    {
      if (update_enable_)//是否更新
      {
        new_points_++;
        temp_points_.push_back(pv);
        if (new_points_ > update_size_threshold_)
        {
          init_plane(temp_points_, plane_ptr_);//先初始化
          new_points_ = 0;
        }
        if (temp_points_.size() >= max_points_num_)//点的个数超过最大个数存储此时去掉
        {
          update_enable_ = false;
          std::vector<pointWithVar>().swap(temp_points_);
          new_points_ = 0;
        }
      }
    }
    else//如果当前节点不是平面且未达到最大层数，则将新点分配到子节点
    {//处理非平面节点
      if (layer_ < max_layer_)//最大层数
      {
        int xyz[3] = {0, 0, 0};//找中心层数
        if (pv.point_w[0] > voxel_center_[0]) { xyz[0] = 1; }
        if (pv.point_w[1] > voxel_center_[1]) { xyz[1] = 1; }
        if (pv.point_w[2] > voxel_center_[2]) { xyz[2] = 1; }
        int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        if (leaves_[leafnum] != nullptr) { leaves_[leafnum]->UpdateOctoTree(pv); }
        else
        {//如果子节点不存在，则创建新的子节点，并设置其中心点和边长
          leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
          leaves_[leafnum]->layer_init_num_ = layer_init_num_;
          leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
          leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
          leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
          leaves_[leafnum]->quater_length_ = quater_length_ / 2;
          leaves_[leafnum]->UpdateOctoTree(pv);
        }
      }
      else
      {
        if (update_enable_)
        //如果达到最大层数且允许更新，则将新点加入 temp_points_，并检查是否需要重新初始化平面或清空点云
        {
          new_points_++;
          temp_points_.push_back(pv);
          if (new_points_ > update_size_threshold_)
          {
            init_plane(temp_points_, plane_ptr_);
            new_points_ = 0;
          }
          if (temp_points_.size() > max_points_num_)//如果多于最大点数 则清空点云
          {
            update_enable_ = false;
            std::vector<pointWithVar>().swap(temp_points_);
            new_points_ = 0;
          }
        }
      }
    }
  }
}
/*
寻找相应点
在vio中调用了，目的是寻找中心点
*/
VoxelOctoTree *VoxelOctoTree::find_correspond(Eigen::Vector3d pw)//根据输入点 pw 的坐标，在八叉树中找到对应的节点
{
  if (!init_octo_ || plane_ptr_->is_plane_ || (layer_ >= max_layer_)) return this;//如果当前节点未初始化、是平面节点或已达到最大层数，则返回当前节点

  int xyz[3] = {0, 0, 0};//根据点 pw 的坐标与当前节点中心点 voxel_center_ 的比较，确定点所在的子节点索引 leafnum
  xyz[0] = pw[0] > voxel_center_[0] ? 1 : 0;
  xyz[1] = pw[1] > voxel_center_[1] ? 1 : 0;
  xyz[2] = pw[2] > voxel_center_[2] ? 1 : 0;
  int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

  // printf("leafnum: %d. \n", leafnum);

  return (leaves_[leafnum] != nullptr) ? leaves_[leafnum]->find_correspond(pw) : this;//如果子节点存在，则递归调用 find_correspond；否则返回当前节点
}
/*
将点插入到八叉树中（没用到）
*/
VoxelOctoTree *VoxelOctoTree::Insert(const pointWithVar &pv)//将点 pv 插入到八叉树中，并返回插入的节
{
  if ((!init_octo_) || (init_octo_ && plane_ptr_->is_plane_) || (init_octo_ && (!plane_ptr_->is_plane_) && (layer_ >= max_layer_)))//如果当前节点未初始化、是平面节点或已达到最大层数，则将点加入当前节点的 temp_points_ 并返回当前节点
  {
    new_points_++;
    temp_points_.push_back(pv);
    return this;
  }
  //如果当前节点不是平面节点且未达到最大层数，则根据点 pv 的坐标与当前节点中心点 voxel_center_ 的比较，确定点所在的子节点索引 leafnum
  if (init_octo_ && (!plane_ptr_->is_plane_) && (layer_ < max_layer_))
  {
    int xyz[3] = {0, 0, 0};
    xyz[0] = pv.point_w[0] > voxel_center_[0] ? 1 : 0;
    xyz[1] = pv.point_w[1] > voxel_center_[1] ? 1 : 0;
    xyz[2] = pv.point_w[2] > voxel_center_[2] ? 1 : 0;
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] != nullptr) { return leaves_[leafnum]->Insert(pv); }//如果子节点存在，则递归调用 Insert
    else//如果子节点不存在，则创建新的子节点并插入点
    {
      leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
      leaves_[leafnum]->layer_init_num_ = layer_init_num_;
      leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
      leaves_[leafnum]->quater_length_ = quater_length_ / 2;
      return leaves_[leafnum]->Insert(pv);
    }
  }
  return nullptr;//如果以上条件均不满足，则返回 nullptr
}
/*
这两个函数共同实现了八叉树的动态更新和查询功能，适用于点云数据的存储和管理
基于扩展卡尔曼滤波（EKF）的状态估计过程包括了很多部分
VoxelMapManager类的相关函数
*/
void VoxelMapManager::StateEstimation(StatesGroup &state_propagat)
{
  cross_mat_list_.clear();//cross_mat_list_：存储点的反对称矩阵
  cross_mat_list_.reserve(feats_down_size_);
  body_cov_list_.clear();//body_cov_list_：存储每个点的协方差矩阵
  body_cov_list_.reserve(feats_down_size_);

  // build_residual_time = 0.0;
  // ekf_time = 0.0;
  // double t0 = omp_get_wtime();
/*处理每个 LiDAR 点
对每个 LiDAR 点：
提取点的坐标，并确保 z 坐标不为零。
计算点的协方差矩阵（var），并将其存入 body_cov_list_。
将点从 LiDAR 坐标系转换到机体坐标系（extR_ 和 extT_ 是外部旋转和平移矩阵）。
计算点的反对称矩阵（point_crossmat），并将其存入 cross_mat_list_。
*/
  for (size_t i = 0; i < feats_down_body_->size(); i++)
  {
    V3D point_this(feats_down_body_->points[i].x, feats_down_body_->points[i].y, feats_down_body_->points[i].z);
    if (point_this[2] == 0) { point_this[2] = 0.001; }
    M3D var;
    calcBodyCov(point_this, config_setting_.dept_err_, config_setting_.beam_err_, var);
    body_cov_list_.push_back(var);
    point_this = extR_ * point_this + extT_;
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);
    cross_mat_list_.push_back(point_crossmat);
  }
/*初始化状态估计变量
清空并调整 pv_list_ 的大小，用于存储带协方差的点。
初始化 EKF 相关矩阵：
G：卡尔曼增益矩阵。
H_T_H：测量雅可比矩阵的转置与测量协方差矩阵的乘积。
I_STATE：单位矩阵
*/
  vector<pointWithVar>().swap(pv_list_);
  pv_list_.resize(feats_down_size_);

  int rematch_num = 0;
  MD(DIM_STATE, DIM_STATE) G, H_T_H, I_STATE;
  G.setZero();//卡尔曼增益矩阵
  H_T_H.setZero();//测量雅可比矩阵的转置与测量协方差矩阵的乘积
  I_STATE.setIdentity();//单位矩阵
/*
迭代进行状态估计：
将 LiDAR 点从机体坐标系转换到世界坐标系。
计算每个点的协方差矩阵，并更新 pv_list_。
清空 ptpl_list_，用于存储点-平面匹配结果。
*/
  bool flg_EKF_inited, flg_EKF_converged, EKF_stop_flg = 0;//
  for (int iterCount = 0; iterCount < config_setting_.max_iterations_; iterCount++)
  {
    double total_residual = 0.0;//总残差初值设为0
    pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar(new pcl::PointCloud<pcl::PointXYZI>);//世界地图
    /*
    在下一个函数中进行了调用
    */
    TransformLidar(state_.rot_end, state_.pos_end, feats_down_body_, world_lidar);//转换函数在下面有说明
    M3D rot_var = state_.cov.block<3, 3>(0, 0);
    M3D t_var = state_.cov.block<3, 3>(3, 3);//外参设置初值
    for (size_t i = 0; i < feats_down_body_->size(); i++)//每个点
    {
      pointWithVar &pv = pv_list_[i];
      pv.point_b << feats_down_body_->points[i].x, feats_down_body_->points[i].y, feats_down_body_->points[i].z;
      pv.point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;

      M3D cov = body_cov_list_[i];
      M3D point_crossmat = cross_mat_list_[i];
      cov = state_.rot_end * cov * state_.rot_end.transpose() + (-point_crossmat) * rot_var * (-point_crossmat.transpose()) + t_var;
      pv.var = cov;
      pv.body_var = body_cov_list_[i];
    }
    ptpl_list_.clear();

    // double t1 = omp_get_wtime();
    //构建残差列表
    /*
    引用总结残差函数，在下面写了具体的函数构成
    */
    BuildResidualListOMP(pv_list_, ptpl_list_);//调用 BuildResidualListOMP 函数，使用 OpenMP 并行计算点-平面匹配的残差s
    //具体函数写在下面 里面调用了
    // build_residual_time += omp_get_wtime() - t1;
    //计算总残差
    for (int i = 0; i < ptpl_list_.size(); i++)
    {
      total_residual += fabs(ptpl_list_[i].dis_to_plane_);
    }
    effct_feat_num_ = ptpl_list_.size();
    cout << "[ LIO ] Raw feature num: " << feats_undistort_->size() << ", downsampled feature num:" << feats_down_size_ 
         << " effective feature num: " << effct_feat_num_ << " average residual: " << total_residual / effct_feat_num_ << endl;//计算总残差并输出有效特征点的数量及平均残差

    /*** Computation of Measuremnt Jacobian matrix H and measurents covarience    计算测量雅可比矩阵和测量协方差
     * ***/
    /*
    计算测量雅可比矩阵 Hsub 和测量协方差矩阵 R_inv。  
    更新测量向量 meas_vec
    */
    MatrixXd Hsub(effct_feat_num_, 6);
    MatrixXd Hsub_T_R_inv(6, effct_feat_num_);
    VectorXd R_inv(effct_feat_num_);
    VectorXd meas_vec(effct_feat_num_);
    meas_vec.setZero();
    for (int i = 0; i < effct_feat_num_; i++)
    {
      auto &ptpl = ptpl_list_[i];
      V3D point_this(ptpl.point_b_);
      point_this = extR_ * point_this + extT_;
      V3D point_body(ptpl.point_b_);
      M3D point_crossmat;
      point_crossmat << SKEW_SYM_MATRX(point_this);

      /*** get the normal vector of closest surface/corner ***/

      V3D point_world = state_propagat.rot_end * point_this + state_propagat.pos_end;
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = point_world - ptpl_list_[i].center_;
      J_nq.block<1, 3>(0, 3) = -ptpl_list_[i].normal_;

      M3D var;
      // V3D normal_b = state_.rot_end.inverse() * ptpl_list_[i].normal_;
      // V3D point_b = ptpl_list_[i].point_b_;
      // double cos_theta = fabs(normal_b.dot(point_b) / point_b.norm());
      // ptpl_list_[i].body_cov_ = ptpl_list_[i].body_cov_ * (1.0 / cos_theta) * (1.0 / cos_theta);

      // point_w cov
      // var = state_propagat.rot_end * extR_ * ptpl_list_[i].body_cov_ * (state_propagat.rot_end * extR_).transpose() +
      //       state_propagat.cov.block<3, 3>(3, 3) + (-point_crossmat) * state_propagat.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose();

      // point_w cov (another_version)
      // var = state_propagat.rot_end * extR_ * ptpl_list_[i].body_cov_ * (state_propagat.rot_end * extR_).transpose() +
      //       state_propagat.cov.block<3, 3>(3, 3) - point_crossmat * state_propagat.cov.block<3, 3>(0, 0) * point_crossmat;

      // point_body cov
      var = state_propagat.rot_end * extR_ * ptpl_list_[i].body_cov_ * (state_propagat.rot_end * extR_).transpose();

      double sigma_l = J_nq * ptpl_list_[i].plane_var_ * J_nq.transpose();

      R_inv(i) = 1.0 / (0.001 + sigma_l + ptpl_list_[i].normal_.transpose() * var * ptpl_list_[i].normal_);
      // R_inv(i) = 1.0 / (sigma_l + ptpl_list_[i].normal_.transpose() * var * ptpl_list_[i].normal_);

      /*** calculate the Measuremnt Jacobian matrix H ***/
      V3D A(point_crossmat * state_.rot_end.transpose() * ptpl_list_[i].normal_);
      Hsub.row(i) << VEC_FROM_ARRAY(A), ptpl_list_[i].normal_[0], ptpl_list_[i].normal_[1], ptpl_list_[i].normal_[2];
      Hsub_T_R_inv.col(i) << A[0] * R_inv(i), A[1] * R_inv(i), A[2] * R_inv(i), ptpl_list_[i].normal_[0] * R_inv(i),
          ptpl_list_[i].normal_[1] * R_inv(i), ptpl_list_[i].normal_[2] * R_inv(i);
      meas_vec(i) = -ptpl_list_[i].dis_to_plane_;
    }

    //扩展卡尔曼滤波更新  计算卡尔曼增益矩阵 K  更新状态向量 state_
    EKF_stop_flg = false;
    flg_EKF_converged = false;
    /*** Iterative Kalman Filter Update ***/
    MatrixXd K(DIM_STATE, effct_feat_num_);
    // auto &&Hsub_T = Hsub.transpose();
    auto &&HTz = Hsub_T_R_inv * meas_vec;
    // fout_dbg<<"HTz: "<<HTz<<endl;
    H_T_H.block<6, 6>(0, 0) = Hsub_T_R_inv * Hsub;
    // EigenSolver<Matrix<double, 6, 6>> es(H_T_H.block<6,6>(0,0));
    MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H.block<DIM_STATE, DIM_STATE>(0, 0) + state_.cov.block<DIM_STATE, DIM_STATE>(0, 0).inverse()).inverse();
    G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
    auto vec = state_propagat - state_;
    VD(DIM_STATE)
    solution = K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec.block<DIM_STATE, 1>(0, 0) - G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);
    int minRow, minCol;
    state_ += solution;
    auto rot_add = solution.block<3, 1>(0, 0);
    auto t_add = solution.block<3, 1>(3, 0);
    //收敛判断与协方差更新  判断 EKF 是否收敛。更新状态协方差矩阵
    if ((rot_add.norm() * 57.3 < 0.01) && (t_add.norm() * 100 < 0.015)) { flg_EKF_converged = true; }
    V3D euler_cur = state_.rot_end.eulerAngles(2, 1, 0);

    /*** Rematch Judgement ***/

    if (flg_EKF_converged || ((rematch_num == 0) && (iterCount == (config_setting_.max_iterations_ - 2)))) { rematch_num++; }

    /*** Convergence Judgements and Covariance Update ***/
    if (!EKF_stop_flg && (rematch_num >= 2 || (iterCount == config_setting_.max_iterations_ - 1)))
    {
      /*** Covariance Update ***/
      // _state.cov = (I_STATE - G) * _state.cov;
      state_.cov.block<DIM_STATE, DIM_STATE>(0, 0) =
          (I_STATE.block<DIM_STATE, DIM_STATE>(0, 0) - G.block<DIM_STATE, DIM_STATE>(0, 0)) * state_.cov.block<DIM_STATE, DIM_STATE>(0, 0);
      // total_distance += (_state.pos_end - position_last).norm();
      position_last_ = state_.pos_end;
      geoQuat_ = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));

      // VD(DIM_STATE) K_sum  = K.rowwise().sum();
      // VD(DIM_STATE) P_diag = _state.cov.diagonal();
      EKF_stop_flg = true;
    }
    if (EKF_stop_flg) break;
  }

  // double t2 = omp_get_wtime();
  // scan_count++;
  // ekf_time = t2 - t0 - build_residual_time;

  // ave_build_residual_time = ave_build_residual_time * (scan_count - 1) / scan_count + build_residual_time / scan_count;
  // ave_ekf_time = ave_ekf_time * (scan_count - 1) / scan_count + ekf_time / scan_count;

  // cout << "[ Mapping ] ekf_time: " << ekf_time << "s, build_residual_time: " << build_residual_time << "s" << endl;
  // cout << "[ Mapping ] ave_ekf_time: " << ave_ekf_time << "s, ave_build_residual_time: " << ave_build_residual_time << "s" << endl;
}
/*
将 LiDAR 点云从机体坐标系转换到世界坐标系
*/
void VoxelMapManager::TransformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud,
                                     pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud)
{
  pcl::PointCloud<pcl::PointXYZI>().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());//input_cloud：输入点云（机体坐标系） 输出点云（世界坐标系）
  for (size_t i = 0; i < input_cloud->size(); i++)//逐个点进行坐标变换
  {
    pcl::PointXYZINormal p_c = input_cloud->points[i];//清空并预留输出点云的空间
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR_ * p + extT_) + t);
    pcl::PointXYZI pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);//将变换后的点存入 trans_cloud
  }
}
/*
//构建体素地图
根据输入的点云数据构建体素地图，并初始化每个体素的八叉树结构。
*/
void VoxelMapManager::BuildVoxelMap()
{//读取配置参数
  float voxel_size = config_setting_.max_voxel_size_;//体素的大小
  float planer_threshold = config_setting_.planner_threshold_;//平面判断的阈值
  int max_layer = config_setting_.max_layer_;//八叉树的最大层数
  int max_points_num = config_setting_.max_points_num_;//每个体素的最大点数
  std::vector<int> layer_init_num = config_setting_.layer_init_num_;//每层的初始化点数
  //准备输入点云数据
  std::vector<pointWithVar> input_points;

  for (size_t i = 0; i < feats_down_world_->size(); i++)//对每个点计算其协方差矩阵，并将其存入 input_points
  {
    pointWithVar pv;
    //将点的三维坐标存储到 pointWithVar 类型中
    pv.point_w << feats_down_world_->points[i].x, feats_down_world_->points[i].y, feats_down_world_->points[i].z;
    V3D point_this(feats_down_body_->points[i].x, feats_down_body_->points[i].y, feats_down_body_->points[i].z);
    M3D var;
    //调用 calcBodyCov 函数计算点的协方差矩阵
    calcBodyCov(point_this, config_setting_.dept_err_, config_setting_.beam_err_, var);
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);
    var = (state_.rot_end * extR_) * var * (state_.rot_end * extR_).transpose() +
          (-point_crossmat) * state_.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + state_.cov.block<3, 3>(3, 3);
    pv.var = var;
    //将点的协方差矩阵和其他信息存储到 input_points 中
    input_points.push_back(pv);
  }

  uint plsize = input_points.size();

  //将点分配到体素中
  for (uint i = 0; i < plsize; i++)
  {
    const pointWithVar p_v = input_points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_v.point_w[j] / voxel_size;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = voxel_map_.find(position);
    if (iter != voxel_map_.end())//如果体素已存在，则将点加入该体素；否则，创建新的体素
    {
      voxel_map_[position]->temp_points_.push_back(p_v);
      voxel_map_[position]->new_points_++;
    }
    else
    {
      VoxelOctoTree *octo_tree = new VoxelOctoTree(max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
      voxel_map_[position] = octo_tree;
      voxel_map_[position]->quater_length_ = voxel_size / 4;
      voxel_map_[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      voxel_map_[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      voxel_map_[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      voxel_map_[position]->temp_points_.push_back(p_v);
      voxel_map_[position]->new_points_++;
      voxel_map_[position]->layer_init_num_ = layer_init_num;
    }
  }
  // 初始化八叉树
  for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); ++iter)
  {
    iter->second->init_octo_tree();//对每个体素调用 init_octo_tree，初始化八叉树结构
  }
}
/*
没用上这个函数 在后面调试的时候再次调用
*/
V3F VoxelMapManager::RGBFromVoxel(const V3D &input_point)//根据体素位置生成 RGB 颜色。
{
  int64_t loc_xyz[3];
  for (int j = 0; j < 3; j++)
  {
    loc_xyz[j] = floor(input_point[j] / config_setting_.max_voxel_size_);
  }

  VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);//计算输入点所在的体素位置
  int64_t ind = loc_xyz[0] + loc_xyz[1] + loc_xyz[2];
  uint k((ind + 100000) % 3);
  V3F RGB((k == 0) * 255.0, (k == 1) * 255.0, (k == 2) * 255.0);//根据体素位置的索引生成 RGB 颜色（简单的颜色分配逻辑）
  // cout<<"RGB: "<<RGB.transpose()<<endl;
  return RGB;
}
/*
更新体素地图
在LIO中计算出了雷达到世界坐标系的协方差 并带入了计算
*/
void VoxelMapManager::UpdateVoxelMap(const std::vector<pointWithVar> &input_points)
{
  float voxel_size = config_setting_.max_voxel_size_;
  float planer_threshold = config_setting_.planner_threshold_;
  int max_layer = config_setting_.max_layer_;
  int max_points_num = config_setting_.max_points_num_;
  std::vector<int> layer_init_num = config_setting_.layer_init_num_;
  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++)//对每个输入点，计算其所在的体素位置
  {
    const pointWithVar p_v = input_points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_v.point_w[j] / voxel_size;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = voxel_map_.find(position);
    if (iter != voxel_map_.end()) { voxel_map_[position]->UpdateOctoTree(p_v); }//如果体素已存在，则调用 UpdateOctoTree 更新体素；否则，创建新的体素并更新
    else
    {
      VoxelOctoTree *octo_tree = new VoxelOctoTree(max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
      voxel_map_[position] = octo_tree;
      voxel_map_[position]->layer_init_num_ = layer_init_num;
      voxel_map_[position]->quater_length_ = voxel_size / 4;
      voxel_map_[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      voxel_map_[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      voxel_map_[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      voxel_map_[position]->UpdateOctoTree(p_v);
    }
  }
}
/*
它从给定的 pointWithVar 点列表构建 PointToPlane 残差列表，并使用基于体素（Voxel）的查找方式 该函数支持 OpenMP 并行计算（如果定义了 MP_EN）。
它会在 体素地图 中查找每个点所属的体素，并使用 build_single_residual 计算 点到平面残差。
如果第一次查找失败，它会尝试在 邻近体素 重新查找。
使用 互斥锁（std::mutex） 确保线程安全
*/
void VoxelMapManager::BuildResidualListOMP(std::vector<pointWithVar> &pv_list, std::vector<PointToPlane> &ptpl_list)
{
  int max_layer = config_setting_.max_layer_;
  double voxel_size = config_setting_.max_voxel_size_;
  double sigma_num = config_setting_.sigma_num_;//从 config_setting_ 结构体中提取 体素参数（最大层级、体素大小、标准差阈值）
  std::mutex mylock;
  ptpl_list.clear();
  //清空 ptpl_list，并初始化：
  //all_ptpl_list：存储计算得到的残差点。
  //useful_ptpl：标记哪些点成功计算了残差。
  //index：记录点的索引
  std::vector<PointToPlane> all_ptpl_list(pv_list.size());
  std::vector<bool> useful_ptpl(pv_list.size());
  std::vector<size_t> index(pv_list.size());
  for (size_t i = 0; i < index.size(); ++i)
  {
    index[i] = i;
    useful_ptpl[i] = false;
  }
  #ifdef MP_EN//如果 MP_EN 被定义，则启用 OpenMP 并行加速
    omp_set_num_threads(MP_PROC_NUM);
    #pragma omp parallel for
  #endif
  for (int i = 0; i < index.size(); i++)//遍历 pv_list 中的所有点，将 世界坐标转换为体素索引
  {
    pointWithVar &pv = pv_list[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = pv.point_w[j] / voxel_size;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = voxel_map_.find(position);//在 体素地图 voxel_map_ 中查找点所属的体素
    if (iter != voxel_map_.end())
    {
      VoxelOctoTree *current_octo = iter->second;
      PointToPlane single_ptpl;
      bool is_sucess = false;
      double prob = 0;
      build_single_residual(pv, current_octo, 0, is_sucess, prob, single_ptpl);//调用 build_single_residual 计算点到平面的残差
      if (!is_sucess)//如果计算失败，尝试在 邻近体素 重新计算残差
      {
        VOXEL_LOCATION near_position = position;//标记体素位置
        if (loc_xyz[0] > (current_octo->voxel_center_[0] + current_octo->quater_length_)) { near_position.x = near_position.x + 1; }
        else if (loc_xyz[0] < (current_octo->voxel_center_[0] - current_octo->quater_length_)) { near_position.x = near_position.x - 1; }
        if (loc_xyz[1] > (current_octo->voxel_center_[1] + current_octo->quater_length_)) { near_position.y = near_position.y + 1; }
        else if (loc_xyz[1] < (current_octo->voxel_center_[1] - current_octo->quater_length_)) { near_position.y = near_position.y - 1; }
        if (loc_xyz[2] > (current_octo->voxel_center_[2] + current_octo->quater_length_)) { near_position.z = near_position.z + 1; }
        else if (loc_xyz[2] < (current_octo->voxel_center_[2] - current_octo->quater_length_)) { near_position.z = near_position.z - 1; }
        auto iter_near = voxel_map_.find(near_position);
        if (iter_near != voxel_map_.end()) { build_single_residual(pv, iter_near->second, 0, is_sucess, prob, single_ptpl); }
      }
      if (is_sucess)
      {
        mylock.lock();//使用 互斥锁（mylock） 确保线程安全地更新 useful_ptpl 和 all_ptpl_list
        useful_ptpl[i] = true;
        all_ptpl_list[i] = single_ptpl;
        mylock.unlock();
      }
      else
      {
        mylock.lock();
        useful_ptpl[i] = false;//标记 useful_ptpl[i] = false
        mylock.unlock();
      }
    }
  }
  for (size_t i = 0; i < useful_ptpl.size(); i++)//在并行部分结束后，将 所有成功计算的 PointToPlane 结果 存入 ptpl_list
  {
    if (useful_ptpl[i]) { ptpl_list.push_back(all_ptpl_list[i]); }
  }
}
/*
该函数用于计算给定点 pv 相对于 当前体素 current_octo 的点到平面（Point-to-Plane）残差，并更新 single_ptpl 结构
构建单个残差 计算函数在上面与后面的函数中进行调用
*/
void VoxelMapManager::build_single_residual(pointWithVar &pv, const VoxelOctoTree *current_octo, const int current_layer, bool &is_sucess,
                                            double &prob, PointToPlane &single_ptpl)//体素残差计算构建单个点的残差
{
  int max_layer = config_setting_.max_layer_;//最大层数，决定八叉树递归深度
  double sigma_num = config_setting_.sigma_num_;//标准差权重，用于判断点是否有效

  double radius_k = 3;//用于范围判定的半径因子
  Eigen::Vector3d p_w = pv.point_w;
  if (current_octo->plane_ptr_->is_plane_)//若 current_octo 具有平面 (is_plane_ = true)，则进行点到平面距离计算
  {
    VoxelPlane &plane = *current_octo->plane_ptr_;
    Eigen::Vector3d p_world_to_center = p_w - plane.center_;
    float dis_to_plane = fabs(plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_);//计算 点到平面距离 (dis_to_plane)
    float dis_to_center = (plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0)) + (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1)) +
                          (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2));
    float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);//计算 点到平面中心的投影半径 (range_dis)

    if (range_dis <= radius_k * plane.radius_)//如果 range_dis 在允许范围内 (≤ radius_k * plane.radius_)，则继续处理
    {
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
      J_nq.block<1, 3>(0, 3) = -plane.normal_;
      double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();//计算残差的 协方差 sigma_l
      sigma_l += plane.normal_.transpose() * pv.var * plane.normal_;
      if (dis_to_plane < sigma_num * sqrt(sigma_l))//若 dis_to_plane < sigma_num * sqrt(sigma_l)，则认为该点有效 计算残差概率 this_prob，选取概率最高的残差进行存储
      {
        is_sucess = true;
        double this_prob = 1.0 / (sqrt(sigma_l)) * exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
        if (this_prob > prob)
        {
          prob = this_prob;
          pv.normal = plane.normal_;
          single_ptpl.body_cov_ = pv.body_var;
          single_ptpl.point_b_ = pv.point_b;
          single_ptpl.point_w_ = pv.point_w;
          single_ptpl.plane_var_ = plane.plane_var_;
          single_ptpl.normal_ = plane.normal_;
          single_ptpl.center_ = plane.center_;
          single_ptpl.d_ = plane.d_;
          single_ptpl.layer_ = current_layer;
          single_ptpl.dis_to_plane_ = plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_;
        }
        return;
      }
      else
      {
        // is_sucess = false;
        return;
      }
    }
    else
    {
      // is_sucess = false;
      return;
    }
  }
  else
  {
    if (current_layer < max_layer)//递归搜索八叉树的子节点，直到达到 max_layer
    {
      for (size_t leafnum = 0; leafnum < 8; leafnum++)
      {
        if (current_octo->leaves_[leafnum] != nullptr)
        {

          VoxelOctoTree *leaf_octo = current_octo->leaves_[leafnum];
          build_single_residual(pv, leaf_octo, current_layer + 1, is_sucess, prob, single_ptpl);
        }
      }
      return;
    }
    else { return; }
  }
}
/*
该函数用于将 体素地图 可视化，并通过 ROS 话题发布 MarkerArray
*/
void VoxelMapManager::pubVoxelMap()
{
  double max_trace = 0.25;//用于归一化协方差的最大值
  double pow_num = 0.2;//用于颜色映射的指数权重
  ros::Rate loop(500);
  float use_alpha = 0.8;//平面透明度
  visualization_msgs::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);
  std::vector<VoxelPlane> pub_plane_list;
  for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); iter++)
  {
    GetUpdatePlane(iter->second, config_setting_.max_layer_, pub_plane_list);//调用 GetUpdatePlane() 获取所有 可视化平面数据
  }
  /*
  为每个平面计算颜色
提取 平面协方差 plane_var_ 的对角元素并计算 迹 trace。
归一化 trace 并 使用 mapJet() 生成 RGB 颜色
  */
  for (size_t i = 0; i < pub_plane_list.size(); i++)
  {
    V3D plane_cov = pub_plane_list[i].plane_var_.block<3, 3>(0, 0).diagonal();//计算 颜色 plane_rgb 和 透明度 alpha
    double trace = plane_cov.sum();
    if (trace >= max_trace) { trace = max_trace; }
    trace = trace * (1.0 / max_trace);
    trace = pow(trace, pow_num);
    uint8_t r, g, b;
    /*
    雅可比计算每个
    */
    mapJet(trace, 0, 1, r, g, b);
    Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
    double alpha;
    if (pub_plane_list[i].is_plane_) { alpha = use_alpha; }
    else { alpha = 0; }
    /*
    在下面的函数中使用 ，目的是使用单个平面的发布
    */
    pubSinglePlane(voxel_plane, "plane", pub_plane_list[i], alpha, plane_rgb);//通过 pubSinglePlane() 将数据转换为 visualization_msgs::MarkerArray
  }
  voxel_map_pub_.publish(voxel_plane);//发布 voxel_plane 话题
  loop.sleep();
}
//遍历八叉树，收集需要更新的平面 (is_update_ = true)
/*
获取新平面的各项数据
*/
void VoxelMapManager::GetUpdatePlane(const VoxelOctoTree *current_octo, const int pub_max_voxel_layer, std::vector<VoxelPlane> &plane_list)
{
  if (current_octo->layer_ > pub_max_voxel_layer) { return; }
  if (current_octo->plane_ptr_->is_update_) { plane_list.push_back(*current_octo->plane_ptr_); }
  if (current_octo->layer_ < current_octo->max_layer_)
  {
    if (!current_octo->plane_ptr_->is_plane_)
    {
      for (size_t i = 0; i < 8; i++)//8个部分的点云
      {
        if (current_octo->leaves_[i] != nullptr) { GetUpdatePlane(current_octo->leaves_[i], pub_max_voxel_layer, plane_list); }//迭代函数
      }
    }
  }
  return;
}

void VoxelMapManager::pubSinglePlane(visualization_msgs::MarkerArray &plane_pub, const std::string plane_ns, const VoxelPlane &single_plane,
                                     const float alpha, const Eigen::Vector3d rgb)//将平面 VoxelPlane 转换为 ROS Marker 并发布
{
  visualization_msgs::Marker plane;
  plane.header.frame_id = "camera_init";
  plane.header.stamp = ros::Time();
  plane.ns = plane_ns;
  plane.id = single_plane.id_;
  plane.type = visualization_msgs::Marker::CYLINDER;
  plane.action = visualization_msgs::Marker::ADD;
  plane.pose.position.x = single_plane.center_[0];
  plane.pose.position.y = single_plane.center_[1];
  plane.pose.position.z = single_plane.center_[2];
  geometry_msgs::Quaternion q;
  /*
  调用下面使用函数
  
  */
  CalcVectQuation(single_plane.x_normal_, single_plane.y_normal_, single_plane.normal_, q);
  plane.pose.orientation = q;
  plane.scale.x = 3 * sqrt(single_plane.max_eigen_value_);
  plane.scale.y = 3 * sqrt(single_plane.mid_eigen_value_);
  plane.scale.z = 2 * sqrt(single_plane.min_eigen_value_);
  plane.color.a = alpha;
  plane.color.r = rgb(0);
  plane.color.g = rgb(1);
  plane.color.b = rgb(2);
  plane.lifetime = ros::Duration();
  plane_pub.markers.push_back(plane);
}

void VoxelMapManager::CalcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec, const Eigen::Vector3d &z_vec,
                                      geometry_msgs::Quaternion &q)//计算四元数方向
{
  Eigen::Matrix3d rot;
  rot << x_vec(0), x_vec(1), x_vec(2), y_vec(0), y_vec(1), y_vec(2), z_vec(0), z_vec(1), z_vec(2);
  Eigen::Matrix3d rotation = rot.transpose();
  Eigen::Quaterniond eq(rotation);
  q.w = eq.w();
  q.x = eq.x();
  q.y = eq.y();
  q.z = eq.z();
}

void VoxelMapManager::mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b)//计算 r, g, b 颜色值（基于数值 v 的范围映射）
{
  r = 255;
  g = 255;
  b = 255;

  if (v < vmin) { v = vmin; }

  if (v > vmax) { v = vmax; }

  double dr, dg, db;

  if (v < 0.1242)
  {
    db = 0.504 + ((1. - 0.504) / 0.1242) * v;
    dg = dr = 0.;
  }
  else if (v < 0.3747)
  {
    db = 1.;
    dr = 0.;
    dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
  }
  else if (v < 0.6253)
  {
    db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
    dg = 1.;
    dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
  }
  else if (v < 0.8758)
  {
    db = 0.;
    dr = 1.;
    dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
  }
  else
  {
    db = 0.;
    dg = 0.;
    dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
  }

  r = (uint8_t)(255 * dr);
  g = (uint8_t)(255 * dg);
  b = (uint8_t)(255 * db);
}

void VoxelMapManager::mapSliding()//判断当前位置 position_last_ 是否移动足够远，如果是，则更新地图
{
  if((position_last_ - last_slide_position).norm() < config_setting_.sliding_thresh)
  {
    std::cout<<RED<<"[DEBUG]: Last sliding length "<<(position_last_ - last_slide_position).norm()<<RESET<<"\n";
    return;
  }

  //get global id now
  last_slide_position = position_last_;
  double t_sliding_start = omp_get_wtime();
  float loc_xyz[3];
  for (int j = 0; j < 3; j++)
  {
    loc_xyz[j] = position_last_[j] / config_setting_.max_voxel_size_;
    if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
  }
  // VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);//discrete global
  /*
  调用下面的函数删除掉超出范围的体素信息
  */
  clearMemOutOfMap((int64_t)loc_xyz[0] + config_setting_.half_map_size, (int64_t)loc_xyz[0] - config_setting_.half_map_size,
                    (int64_t)loc_xyz[1] + config_setting_.half_map_size, (int64_t)loc_xyz[1] - config_setting_.half_map_size,
                    (int64_t)loc_xyz[2] + config_setting_.half_map_size, (int64_t)loc_xyz[2] - config_setting_.half_map_size);
  double t_sliding_end = omp_get_wtime();
  std::cout<<RED<<"[DEBUG]: Map sliding using "<<t_sliding_end - t_sliding_start<<" secs"<<RESET<<"\n";
  return;
}
/*
遍历 voxel_map_，删除超出范围的体素
*/
void VoxelMapManager::clearMemOutOfMap(const int& x_max,const int& x_min,const int& y_max,const int& y_min,const int& z_max,const int& z_min )
{
  int delete_voxel_cout = 0;
  // double delete_time = 0;
  // double last_delete_time = 0;
  for (auto it = voxel_map_.begin(); it != voxel_map_.end(); )
  {
    const VOXEL_LOCATION& loc = it->first;
    bool should_remove = loc.x > x_max || loc.x < x_min || loc.y > y_max || loc.y < y_min || loc.z > z_max || loc.z < z_min;
    if (should_remove){
      // last_delete_time = omp_get_wtime();
      delete it->second;
      it = voxel_map_.erase(it);
      // delete_time += omp_get_wtime() - last_delete_time;
      delete_voxel_cout++;
    } else {
      ++it;
    }
  }
  std::cout<<RED<<"[DEBUG]: Delete "<<delete_voxel_cout<<" root voxels"<<RESET<<"\n";
  // std::cout<<RED<<"[DEBUG]: Delete "<<delete_voxel_cout<<" voxels using "<<delete_time<<" s"<<RESET<<"\n";
}