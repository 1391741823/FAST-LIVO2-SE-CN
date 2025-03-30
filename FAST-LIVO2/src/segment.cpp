#include <include/segment.hpp>

Segment::~Segment() {}

Segment::Segment(): 
    extT(0, 0, 0),//平移
    extR(M3D::Identity())//旋转
{
    extrinT.assign(3, 0.0);
    extrinR.assign(9, 0.0);
    cameraextrinT.assign(3, 0.0);
    cameraextrinR.assign(9, 0.0);//协助外参

    p_pre.reset(new Preprocess());//IMU数据
    p_imu.reset(new ImuProcess());//LiDAR预处理

    // readParameters(nh);
    // VoxelMapConfig voxel_config;
    // loadVoxelConfig(nh, voxel_config);

    visual_sub_map.reset(new PointCloudXYZI());
    feats_undistort.reset(new PointCloudXYZI());
    feats_down_body.reset(new PointCloudXYZI());
    feats_down_world.reset(new PointCloudXYZI());
    pcl_w_wait_pub.reset(new PointCloudXYZI());
    pcl_wait_pub.reset(new PointCloudXYZI());
    pcl_wait_save.reset(new PointCloudXYZRGB());
    voxelmap_manager.reset(new VoxelMapManager(voxel_config, voxel_map));
    vio_manager.reset(new VIOManager());//重置

    root_dir = ROOT_DIR;
    initializeFiles();//初始化文件
    initializeComponents();//初始化组成
    path.header.stamp = ros::Time::now();
    path.header.frame_id = "camera_init";
    
}

