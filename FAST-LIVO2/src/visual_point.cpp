/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "visual_point.h"
#include "feature.h"
#include <stdexcept>
#include <vikit/math_utils.h>
/*
构造函数VisualPoint接收一个三维向量pos，用来初始化pos_，也就是这个点的三维位置。
其他成员变量如previous_normal_和normal_初始化为零向量，is_converged_、is_normal_initialized_和has_ref_patch_都初始化为false。
这些变量可能用于记录法线方向、收敛状态、法线是否初始化以及是否有参考图像块（ref_patch）
*/

VisualPoint::VisualPoint(const Vector3d &pos)
    : pos_(pos), previous_normal_(Vector3d::Zero()), normal_(Vector3d::Zero()),
      is_converged_(false), is_normal_initialized_(false), has_ref_patch_(false)
      /*
      pos_：3D 空间中的位置坐标
normal_ & previous_normal_：存储法向量信息
is_converged_：该点是否已经收敛（用于优化）
is_normal_initialized_：法向量是否初始化
has_ref_patch_：是否有参考特征点（Patch）
      */
{
}

VisualPoint::~VisualPoint() 
{
  for (auto it = obs_.begin(), ite = obs_.end(); it != ite; ++it)
  {
    delete(*it);
  }
  obs_.clear();
  ref_patch = nullptr;
}

void VisualPoint::addFrameRef(Feature *ftr)//添加观测
{
  obs_.push_front(ftr);
}
/*
用于删除特定的Feature引用。首先检查是否是ref_patch，如果是的话，将ref_patch置空，并设置has_ref_patch_为false。然后遍历obs_，找到对应的Feature指针，删除并移除该元素。
这里需要注意的是，如果Feature被删除，那么对应的内存会被释放，所以在其他地方不能再使用这个指针
*/
void VisualPoint::deleteFeatureRef(Feature *ftr)//删除观测点
{
  if (ref_patch == ftr)
  {
    ref_patch = nullptr;
    has_ref_patch_ = false;
  }
  for (auto it = obs_.begin(), ite = obs_.end(); it != ite; ++it)
  {
    if ((*it) == ftr)
    {
      delete((*it));
      obs_.erase(it);
      return;
    }
  }
}
/*
作用是找到与当前帧位置framepos视角最接近的观测点。函数参数中的ftr是一个输出参数，用于返回找到的Feature指针。函数首先检查obs_是否为空，如果为空则返回false。
然后计算当前帧位置到该三维点的方向向量obs_dir，并归一化。接着遍历所有观测，对于每个观测，计算该观测对应的相机位置（通过T_f_w_的逆变换得到平移部分，即相机的位置）到三维点的方向向量dir，
并计算与obs_dir的点积，即夹角的余弦值。找到最大的余弦值对应的观测，即最接近当前视角的观测。如果最大余弦值小于0.5（对应角度大于60度），则返回false。否则，将找到的Feature指针赋给ftr，并返回true。
这里可能有一个问题，如果当前帧的位置和某个观测的相机位置相同，可能会导致dir为零向量，归一化时出现除以零的错误。
不过这种情况可能在系统中被其他部分处理，比如不会添加相同位置的观测

*/
bool VisualPoint::getCloseViewObs(const Vector3d &framepos, Feature *&ftr, const Vector2d &cur_px) const//找到与当前相机位置视角最接近的观测点
{
  // TODO: get frame with same point of view AND same pyramid level!
  if (obs_.size() <= 0) return false;

  Vector3d obs_dir(framepos - pos_);
  obs_dir.normalize();
  auto min_it = obs_.begin();
  double min_cos_angle = 0;
  for (auto it = obs_.begin(), ite = obs_.end(); it != ite; ++it)
  {
    Vector3d dir((*it)->T_f_w_.inverse().translation() - pos_);
    dir.normalize();
    double cos_angle = obs_dir.dot(dir);
    if (cos_angle > min_cos_angle)
    {
      min_cos_angle = cos_angle;
      min_it = it;
    }
  }
  ftr = *min_it;

  // Vector2d ftr_px = ftr->px_;
  // double pixel_dist = (cur_px-ftr_px).norm();

  // if(pixel_dist > 200)
  // {
  //   ROS_ERROR("The pixel dist exceeds 200.");
  //   return false;
  // }

  if (min_cos_angle < 0.5) // assume that observations larger than 60° are useless 0.5
  {
    // ROS_ERROR("The obseved angle is larger than 60°.");
    return false;
  }

  return true;
}
/*
用于找到得分最小的Feature，这可能是因为得分低的特征点更稳定或者更适合作为参考。遍历所有观测，找到score_最小的Feature，将其赋给ftr参数
*/
void VisualPoint::findMinScoreFeature(const Vector3d &framepos, Feature *&ftr) const//找到得分最低的观测点（可能表示最稳定）
{
  auto min_it = obs_.begin();
  float min_score = std::numeric_limits<float>::max();

  for (auto it = obs_.begin(), ite = obs_.end(); it != ite; ++it)
  {
    if ((*it)->score_ < min_score)
    {
      min_score = (*it)->score_;
      min_it = it;
    }
  }
  ftr = *min_it;
}
/*
删除所有非ref_patch的Feature。遍历obs_，如果当前元素不是ref_patch，就删除该Feature，并从链表中移除。
这样，最后只剩下ref_patch对应的Feature。这可能用于在优化后，只保留参考关键帧的特征点，减少内存占用
*/
void VisualPoint::deleteNonRefPatchFeatures()//删除所有 非参考 观测的 Feature*
{
  for (auto it = obs_.begin(); it != obs_.end();)
  {
    if (*it != ref_patch)
    {
      delete *it;
      it = obs_.erase(it);
    }
    else
    {
      ++it;
    }
  }
}