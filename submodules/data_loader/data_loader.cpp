#include <fstream>
#include <iostream>
#include <memory>

#include "pcl/io/pcd_io.h"
#include "submodules/data_loader/data_loader.h"
#include "submodules/utils/bin_utils/endian.h"
#include "submodules/utils/ray_utils/ray_utils.h"
#include "submodules/data_loader/data_parsers/kitti_parser.hpp"
#include "submodules/data_loader/data_parsers/oxford_spires_parser.hpp"
#include "submodules/utils/ply_utils/ply_utils_pcl.h"
#include "submodules/utils/ply_utils/ply_utils_torch.h"
#include <pcl/visualization/pcl_visualizer.h>
enum DatasetType
{
  Replica = 0,
  R3live = 1,
  NeuralRGBD = 2,
  Kitti = 3,
  Fastlivo = 4,
  Spires = 5,
};

namespace dataloader
{
  DataLoader::DataLoader(const std::string &dataset_path,
                         const int &_dataset_type, const torch::Device &_device,
                         const bool &_preload, const float &_res_scale,
                         const sensor::Sensors &_sensor)
      : dataset_type_(_dataset_type), device_(_device)
  {
    switch (dataset_type_)
    {
    case DatasetType::Kitti:
      dataparser_ptr_ = std::make_shared<dataparser::Kitti>(dataset_path, device_, _preload, _res_scale);
      break;
    case DatasetType::Spires:
      dataparser_ptr_ = std::make_shared<dataparser::Spires>(dataset_path, device_, _preload, _res_scale);
      break;
    default:
      throw std::runtime_error("Unsupported dataset type");
    }
  }

  bool DataLoader::get_item(int idx, torch::Tensor &lidar_pose, torch::Tensor &cam_pose, pcl::PointCloud<pcl::PointXYZRGB> &colored_points, cv::Mat &image)
  {
    if (idx >= dataparser_ptr_->raw_depth_filelists_.size())
    {
      std::cout << "\nEnd of the data!\n";
      return false;
    }

    // std::cout << "\rData idx: " << idx << ", Depth file:" << dataparser_ptr_->raw_depth_filelists_[idx];
    // 这个位姿是相机在世界坐标系下的位姿
    lidar_pose = get_pose(idx, dataparser::DataType::RawDepth).to(device_);
    cam_pose = get_pose(idx, dataparser::DataType::RawColor).to(device_);
    // std::cout << "Pose: " << _pose << std::endl;

    // 点云文件路径
    std::string infile = dataparser_ptr_->dataset_path_ / dataparser_ptr_->get_file(idx, dataparser::DataType::RawDepth);

    // 读取点云数据(.bin, .ply, .pcd)
    if (infile.find(".bin") != std::string::npos)
    {
      std::ifstream input(infile.c_str(), std::ios::in | std::ios::binary);
      if (!input)
      {
        std::cerr << "Could not read file: " << infile << "\n";
        return false;
      }

      const size_t kMaxNumberOfPoints = 1e6; // From the Readme of raw files.
      colored_points.clear();
      colored_points.reserve(kMaxNumberOfPoints);

      while (input.is_open() && !input.eof())
      {
        // PointT point;
        pcl::PointXYZRGB point;

        input.read((char *)&point.x, 3 * sizeof(float));
        // pcl::PointXYZI
        float intensity;
        input.read((char *)&intensity, sizeof(float));
        // input.read((char *)&point.intensity, sizeof(float));
        colored_points.push_back(point);
      }
      input.close();
    }
    // else if (infile.find(".ply") != std::string::npos)
    // {
    //   ply_utils::read_ply_file(infile, colored_points);
    // }
    // else if (infile.find(".pcd") != std::string::npos)
    // {
    //   pcl::io::loadPCDFile<PointT>(infile, colored_points);
    // }

    // 读取图像数据
    std::string imgfile = dataparser_ptr_->dataset_path_ / dataparser_ptr_->get_file(idx, dataparser::DataType::RawColor);
    image = cv::imread(imgfile, cv::IMREAD_UNCHANGED);
    if (image.empty())
    {
      std::cerr << "Could not read image: " << imgfile << "\n";
      return false;
    }
    // std::cout << "Colored points before colorization: " << colored_points.size() << std::endl;
    colorize_pointcloud(colored_points, image, dataparser_ptr_->P, dataparser_ptr_->Tr, lidar_pose);
    // std::cout << "Colored points: " << colored_points.size() << std::endl;
    return true;
  }

  bool DataLoader::get_item(
      int idx,
      torch::Tensor &cam_pose,
      torch::Tensor &point_cloud,
      torch::Tensor &color,
      cv::Mat &image)
  {
    if (idx >= dataparser_ptr_->raw_depth_filelists_.size())
    {
      std::cout << "\nEnd of the data!\n";
      return false;
    }

    cam_pose = get_pose(idx, dataparser::DataType::RawColor).to(device_);
    std::string imgfile = dataparser_ptr_->dataset_path_ / dataparser_ptr_->get_file(idx, dataparser::DataType::RawColor);
    // std::cout << "Image file: " << imgfile << std::endl;
    image = cv::imread(imgfile, cv::IMREAD_COLOR); // 直接读取为 RGB 格式
    if (image.empty())
    {
      std::cerr << "Could not read image: " << imgfile << "\n";
      return false;
    }
    image.convertTo(image, CV_32FC3, 1.0f / 255.0f);
    torch::Tensor image_tensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kFloat32).to(device_);

    torch::Tensor lidar_pose = get_pose(idx, dataparser::DataType::RawDepth).to(device_);
    std::string infile = dataparser_ptr_->dataset_path_ / dataparser_ptr_->get_file(idx, dataparser::DataType::RawDepth);
    // std::cout << "Point cloud file: " << infile << std::endl;
    pcl::PointCloud<pcl::PointXYZ> points;
    // 读取点云数据 (支持 .bin, .ply, .pcd)
    if (infile.find(".bin") != std::string::npos)
    {
      std::ifstream input(infile.c_str(), std::ios::in | std::ios::binary);
      if (!input)
      {
        std::cerr << "Could not read file: " << infile << "\n";
        return false;
      }

      const size_t kMaxNumberOfPoints = 1e6; // 最大点数
      points.reserve(kMaxNumberOfPoints);
      pcl::PointXYZ point;
      while (input.read(reinterpret_cast<char *>(&point.x), 3 * sizeof(float)) && !input.eof())
      {
        float intensity;
        input.read(reinterpret_cast<char *>(&intensity), sizeof(float));
        points.push_back(point);
      }
      input.close();
    }
    else if (infile.find(".ply") != std::string::npos)
    {
      ply_utils::read_ply_file(infile, points);
    }
    else if (infile.find(".pcd") != std::string::npos)
    {
      pcl::io::loadPCDFile<pcl::PointXYZ>(infile, points);
    }

    point_cloud = torch::zeros({points.size(), 3}, torch::kFloat32).to(device_);
    color = torch::zeros({points.size(), 3}, torch::kFloat32).to(device_);
    for (size_t i = 0; i < points.size(); ++i)
    {
      point_cloud[i][0] = points[i].x;
      point_cloud[i][1] = points[i].y;
      point_cloud[i][2] = points[i].z;
    }

    // 投影矩阵和变换矩阵直接传递
    Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Tr = dataparser_ptr_->Tr;
    Eigen::Matrix<float, 3, 4, Eigen::RowMajor> P = dataparser_ptr_->P;
    // std::cout << "Tr: " << Tr << std::endl;
    // std::cout << "P: " << P << std::endl;
    torch::Tensor proj_mat = torch::from_blob(P.data(), {12}, torch::kFloat32).to(device_);
    torch::Tensor Tr_velo_to_cam = torch::from_blob(Tr.data(), {16}, torch::kFloat32).to(device_);
    // 点云着色（并行化，批量处理）
    colorize_launcher(point_cloud, color, proj_mat, Tr_velo_to_cam, lidar_pose, image_tensor);

    // torch::Tensor cam_inv = cam_pose.inverse();
    // std::cout << "相机位姿的逆为:\n " << cam_inv << std::endl;
    // 用以显示点云，查看染色结果是否正确
    if (false)
    {
      pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Viewer"));
      viewer->setBackgroundColor(0, 0, 0);
      viewer->initCameraParameters();
      viewer->addCoordinateSystem(1.0);

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud_raw(new pcl::PointCloud<pcl::PointXYZRGB>());
      pcl_cloud_raw->points.reserve(points.size());
      for (const auto &pt : points)
      {
        pcl::PointXYZRGB p;
        p.x = pt.x;
        p.y = pt.y;
        p.z = pt.z;
        p.r = 255;
        p.g = 0;
        p.b = 0;
        pcl_cloud_raw->points.push_back(p);
      }
      pcl_cloud_raw->width = pcl_cloud_raw->points.size();
      pcl_cloud_raw->height = 1;
      pcl_cloud_raw->is_dense = false;
      viewer->addPointCloud<pcl::PointXYZRGB>(pcl_cloud_raw, "raw_cloud");
      viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "raw_cloud");

      if (point_cloud.device().is_cuda())
        point_cloud = point_cloud.cpu();
      if (color.device().is_cuda())
        color = color.cpu();
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
      pcl_cloud->points.reserve(point_cloud.size(0));
      for (int i = 0; i < point_cloud.size(0); ++i)
      {
        pcl::PointXYZRGB pt;
        pt.x = point_cloud[i][0].item<float>();
        pt.y = point_cloud[i][1].item<float>();
        pt.z = point_cloud[i][2].item<float>();
        pt.r = static_cast<uint8_t>(color[i][0].item<float>() * 255.0f);
        pt.g = static_cast<uint8_t>(color[i][1].item<float>() * 255.0f);
        pt.b = static_cast<uint8_t>(color[i][2].item<float>() * 255.0f);
        pcl_cloud->points.push_back(pt);
      }
      pcl_cloud->width = pcl_cloud->points.size();
      pcl_cloud->height = 1;
      pcl_cloud->is_dense = false;
      viewer->addPointCloud<pcl::PointXYZRGB>(pcl_cloud, "colored_cloud");
      viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "colored_cloud");

      torch::Tensor rotation = cam_pose.narrow(0, 0, 3).narrow(1, 0, 3);    // 3x3 旋转矩阵
      torch::Tensor translation = cam_pose.narrow(0, 0, 3).narrow(1, 3, 1); // 3x1 位移向量

      // 将相机位姿转换为 Eigen 类型用于 PCL 可视化
      Eigen::Matrix4f T_camera = Eigen::Matrix4f::Identity();
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
        {
          T_camera(i, j) = rotation[i][j].item<float>();
        }
        T_camera(i, 3) = translation[i].item<float>();
      }
      Eigen::Vector3f camera_position = T_camera.block<3, 1>(0, 3); // 相机位置
      Eigen::Vector3f x_axis = camera_position + 0.5f * T_camera.block<3, 1>(0, 0);
      Eigen::Vector3f y_axis = camera_position + 0.5f * T_camera.block<3, 1>(0, 1);
      Eigen::Vector3f z_axis = camera_position + 0.5f * T_camera.block<3, 1>(0, 2);

      torch::Tensor rotation_ = lidar_pose.narrow(0, 0, 3).narrow(1, 0, 3);    // 3x3 旋转矩阵
      torch::Tensor translation_ = lidar_pose.narrow(0, 0, 3).narrow(1, 3, 1); // 3x1 位移向量
      Eigen::Matrix4f T_lidar = Eigen::Matrix4f::Identity();
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
        {
          T_lidar(i, j) = rotation_[i][j].item<float>();
        }
        T_lidar(i, 3) = translation_[i].item<float>();
      }
      Eigen::Vector3f lidar_position = T_lidar.block<3, 1>(0, 3); // 相机位置
      Eigen::Vector3f x_axis_ = lidar_position + 0.5f * T_lidar.block<3, 1>(0, 0);
      Eigen::Vector3f y_axis_ = lidar_position + 0.5f * T_lidar.block<3, 1>(0, 1);
      Eigen::Vector3f z_axis_ = lidar_position + 0.5f * T_lidar.block<3, 1>(0, 2);

      viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(camera_position.x(), camera_position.y(), camera_position.z()),
                                     pcl::PointXYZ(x_axis.x(), x_axis.y(), x_axis.z()), 1.0, 0.0, 0.0, "x_axis");
      viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(camera_position.x(), camera_position.y(), camera_position.z()),
                                     pcl::PointXYZ(y_axis.x(), y_axis.y(), y_axis.z()), 0.0, 1.0, 0.0, "y_axis");
      viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(camera_position.x(), camera_position.y(), camera_position.z()),
                                     pcl::PointXYZ(z_axis.x(), z_axis.y(), z_axis.z()), 0.0, 0.0, 1.0, "z_axis");

      viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(lidar_position.x(), lidar_position.y(), lidar_position.z()),
                                     pcl::PointXYZ(x_axis_.x(), x_axis_.y(), x_axis_.z()), 1.0, 0.0, 0.0, "x_axis_");
      viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(lidar_position.x(), lidar_position.y(), lidar_position.z()),
                                     pcl::PointXYZ(y_axis_.x(), y_axis_.y(), y_axis_.z()), 0.0, 1.0, 0.0, "y_axis_");
      viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(lidar_position.x(), lidar_position.y(), lidar_position.z()),
                                     pcl::PointXYZ(z_axis_.x(), z_axis_.y(), z_axis_.z()), 0.0, 0.0, 1.0, "z_axis_");

      // 把染色后的点云转到相机坐标系下
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud_in_cam(new pcl::PointCloud<pcl::PointXYZRGB>());
      pcl_cloud_in_cam->points.reserve(pcl_cloud->points.size());

      Eigen::Matrix4f T_world_to_cam = T_camera.inverse(); // 世界坐标系 -> 相机坐标系变换

      for (const auto &pt : pcl_cloud->points)
      {
        Eigen::Vector4f p_world(pt.x, pt.y, pt.z, 1.0f);
        Eigen::Vector4f p_cam = T_world_to_cam * p_world;

        pcl::PointXYZRGB p_cam_pt;
        p_cam_pt.x = p_cam.x();
        p_cam_pt.y = p_cam.y();
        p_cam_pt.z = p_cam.z();
        p_cam_pt.r = pt.r;
        p_cam_pt.g = pt.g;
        p_cam_pt.b = pt.b;

        pcl_cloud_in_cam->points.push_back(p_cam_pt);
      }

      pcl_cloud_in_cam->width = pcl_cloud_in_cam->points.size();
      pcl_cloud_in_cam->height = 1;
      pcl_cloud_in_cam->is_dense = false;

      viewer->addPointCloud<pcl::PointXYZRGB>(pcl_cloud_in_cam, "cloud_in_cam");
      viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_in_cam");

      viewer->spin();
    }

    return true;
  }

  void DataLoader::colorize_pointcloud(
      pcl::PointCloud<pcl::PointXYZRGB> &cloud,
      const cv::Mat &image,
      const Eigen::Matrix<float, 3, 4> &proj_mat,       // 相机投影矩阵
      const Eigen::Matrix<float, 4, 4> &Tr_velo_to_cam, // 从 Velodyne 到相机的变换
      const torch::Tensor &lidar_pose)
  {
    if (cloud.empty() || image.empty())
    {
      return;
    }

    // 验证 lidar_pose
    if (!lidar_pose.is_contiguous() || lidar_pose.dim() != 2 ||
        lidar_pose.size(0) != 4 || lidar_pose.size(1) != 4)
    {
      throw std::runtime_error("lidar_pose must be a 4x4 tensor");
    }
    if (!lidar_pose.dtype().Match<float>())
    {
      throw std::runtime_error("lidar_pose must be float tensor");
    }

    // 将 lidar_pose 转换为 Eigen 矩阵
    auto lidar_pose_cpu = lidar_pose.cpu().contiguous();
    Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> pose(lidar_pose_cpu.data_ptr<float>());

    // 创建新点云存储有效点
    pcl::PointCloud<pcl::PointXYZRGB> filtered_cloud;
    for (size_t i = 0; i < cloud.points.size(); ++i)
    {
      auto &point = cloud.points[i];
      Eigen::Vector4f pt_velo(point.x, point.y, point.z, 1.0);
      Eigen::Vector3f pt_cam = proj_mat * Tr_velo_to_cam * pt_velo;

      if (pt_cam(2) >= 0.1)
      {
        int u = static_cast<int>(pt_cam(0) / pt_cam(2));
        int v = static_cast<int>(pt_cam(1) / pt_cam(2));

        if (u >= 0 && u < image.cols && v >= 0 && v < image.rows)
        {
          auto color = image.at<cv::Vec3b>(v, u);
          pt_velo = pose * pt_velo; // 转换到全局系

          // 创建新点
          pcl::PointXYZRGB new_point;
          new_point.x = pt_velo(0);
          new_point.y = pt_velo(1);
          new_point.z = pt_velo(2);
          new_point.r = color[2];
          new_point.g = color[1];
          new_point.b = color[0];

          // 添加到新点云
          filtered_cloud.points.push_back(new_point);
        }
      }
    }

    // 更新原始点云
    cloud = std::move(filtered_cloud);
    cloud.width = cloud.points.size();
    cloud.height = 1;
    cloud.is_dense = true;
  }
  torch::Tensor DataLoader::get_pose(int idx, const int &pose_type)
  {
    // return dataparser_ptr_->get_pose(idx, pose_type).slice(0, 0, 3);      // return pose matrix at idx with shape [3, 4]
    return dataparser_ptr_->get_pose(idx, pose_type); // 4x4
  }
} // namespace dataloader