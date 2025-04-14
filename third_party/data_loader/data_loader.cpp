#include "data_loader.h"
#include <fstream>
#include <iostream>
#include <memory>

#include "utils/bin_utils/endian.h"
#include "utils/ray_utils/ray_utils.h"
#include "data_loader/data_parsers/kitti_parser.hpp"
#include "pcl/io/pcd_io.h"
#include "utils/ply_utils/ply_utils_pcl.h"
#include "utils/ply_utils/ply_utils_torch.h"

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
    case DatasetType::NeuralRGBD:
      dataparser_ptr_ = std::make_shared<dataparser::NeuralRGBD>(
          dataset_path, device_, _preload, _res_scale, 0);
      break;
    case DatasetType::Replica:
      dataparser_ptr_ = std::make_shared<dataparser::Replica>(
          dataset_path, device_, _preload, _res_scale);
      break;
    case DatasetType::R3live:
      dataparser_ptr_ = std::make_shared<dataparser::R3live>(
          dataset_path, device_, _preload, _res_scale);
      break;
    case DatasetType::Kitti:
      dataparser_ptr_ = std::make_shared<dataparser::Kitti>(dataset_path, device_,
                                                            _preload, _res_scale);
      break;
    case DatasetType::Fastlivo:
      dataparser_ptr_ = std::make_shared<dataparser::Fastlivo>(
          dataset_path, device_, _preload, _res_scale, _sensor);
      break;
    case DatasetType::Spires:
      dataparser_ptr_ = std::make_shared<dataparser::Spires>(
          dataset_path, device_, _preload, _res_scale, _sensor);
      break;
    default:
      throw std::runtime_error("Unsupported dataset type");
    }
  }

  bool get_item(int idx, torch::Tensor &_pose, PointCloudT &colored_points, cv::Mat &image)
  {
    if (idx >= dataparser_ptr_->raw_depth_filelists_.size())
    {
      std::cout << "\nEnd of the data!\n";
      return false;
    }

    // std::cout << "\rData idx: " << idx << ", Depth file:" << dataparser_ptr_->raw_depth_filelists_[idx];
    _pose = get_pose(idx, dataparser::DataType::RawDepth).to(device_);

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
      _points.clear();
      _points.reserve(kMaxNumberOfPoints);

      while (input.is_open() && !input.eof())
      {
        PointT point;

        input.read((char *)&point.x, 3 * sizeof(float));
        // pcl::PointXYZI
        float intensity;
        input.read((char *)&intensity, sizeof(float));
        // input.read((char *)&point.intensity, sizeof(float));
        _points.push_back(point);
      }
      input.close();
    }
    else if (infile.find(".ply") != std::string::npos)
    {
      ply_utils::read_ply_file(infile, _points);
    }
    else if (infile.find(".pcd") != std::string::npos)
    {
      pcl::io::loadPCDFile<PointT>(infile, _points);
    }

    // 读取图像数据
    std::string infile = dataparser_ptr_->dataset_path_ / dataparser_ptr_->get_file(idx, dataparser::DataType::RawColor);
    image = cv::imread(infile, cv::IMREAD_UNCHANGED);
    if (image.empty())
    {
      std::cerr << "Could not read image: " << infile << "\n";
      return false;
    }

    return true;
  }

  // // 获取下一个带位姿的点云数据
  // bool DataLoader::get_item(int idx, torch::Tensor &_pose, PointCloudT &_points)
  // {
  //   if (idx >= dataparser_ptr_->raw_depth_filelists_.size())
  //   {
  //     std::cout << "\nEnd of the data!\n";
  //     return false;
  //   }

  //   // std::cout << "\rData idx: " << idx << ", Depth file:" << dataparser_ptr_->raw_depth_filelists_[idx];
  //   _pose = get_pose(idx, dataparser::DataType::RawDepth).to(device_);

  //   // 点云文件路径
  //   std::string infile = dataparser_ptr_->dataset_path_ / dataparser_ptr_->get_file(idx, dataparser::DataType::RawDepth);

  //   // 读取点云数据(.bin, .ply, .pcd)
  //   if (infile.find(".bin") != std::string::npos)
  //   {
  //     std::ifstream input(infile.c_str(), std::ios::in | std::ios::binary);
  //     if (!input)
  //     {
  //       std::cerr << "Could not read file: " << infile << "\n";
  //       return false;
  //     }

  //     const size_t kMaxNumberOfPoints = 1e6; // From the Readme of raw files.
  //     _points.clear();
  //     _points.reserve(kMaxNumberOfPoints);

  //     while (input.is_open() && !input.eof())
  //     {
  //       PointT point;

  //       input.read((char *)&point.x, 3 * sizeof(float));
  //       // pcl::PointXYZI
  //       float intensity;
  //       input.read((char *)&intensity, sizeof(float));
  //       // input.read((char *)&point.intensity, sizeof(float));
  //       _points.push_back(point);
  //     }
  //     input.close();
  //   }
  //   else if (infile.find(".ply") != std::string::npos)
  //   {
  //     ply_utils::read_ply_file(infile, _points);
  //   }
  //   else if (infile.find(".pcd") != std::string::npos)
  //   {
  //     pcl::io::loadPCDFile<PointT>(infile, _points);
  //   }

  //   return true;
  // }

  // bool DataLoader::get_item(int idx, torch::Tensor &_pose,
  //                           DepthSamples &_depth_rays,
  //                           ColorSamples &_color_rays,
  //                           const torch::Device &_device)
  // {
  //   if (idx >= dataparser_ptr_->size(1))
  //   {
  //     std::cout << "\nEnd of the data!\n";
  //     return false;
  //   }

  //   get_item(idx, _pose, _depth_rays, device_);
  //   torch::Tensor color_pose;
  //   get_item(idx, color_pose, _color_rays, device_);
  //   return true;
  // }

  // bool DataLoader::get_item(int idx, torch::Tensor &_pose,
  //                           DepthSamples &_depth_rays,
  //                           const torch::Device &_device)
  // {
  //   if (idx >= dataparser_ptr_->size(1))
  //   {
  //     std::cout << "\nEnd of the data!\n";
  //     return false;
  //   }

  //   // std::cout << "\rData idx: " << idx << ", Depth file:" << dataparser_ptr_->raw_depth_filelists_[idx];

  //   auto points_ndir_dirn = dataparser_ptr_->get_distance_ndir_zdirn(idx);
  //   DepthSamples depth_rays;
  //   depth_rays.depth = points_ndir_dirn[0].view({-1, 1}).to(_device);

  //   depth_rays.direction = points_ndir_dirn[1].view({-1, 3}).to(_device);

  //   _pose = get_pose(idx, dataparser::DataType::RawDepth).to(_device);
  //   auto rot = _pose.slice(1, 0, 3);
  //   auto pos = _pose.slice(1, 3, 4).squeeze();
  //   //[n,3]
  //   depth_rays.origin = pos.expand({depth_rays.depth.size(0), 3});
  //   depth_rays.direction = depth_rays.direction.mm(rot.t());
  //   depth_rays.xyz = depth_rays.direction * depth_rays.depth + pos;
  //   _depth_rays = depth_rays;
  //   return true;
  // }

  // // 获取下一个带位姿的图像数据并生成每个像素的光线
  // bool DataLoader::get_item(int idx, torch::Tensor &_pose,
  //                           ColorSamples &_color_rays,
  //                           const torch::Device &_device)
  // {
  //   if (idx >= dataparser_ptr_->size(0))
  //   {
  //     std::cout << "\nEnd of the data!\n";
  //     return false;
  //   }
  //   // std::cout << "\rData idx: " << idx << ", Color file:" << dataparser_ptr_->color_filelists_[idx];

  //   auto rgb = dataparser_ptr_->get_image(idx, dataparser::DataType::RawColor);
  //   if (rgb.numel() == 0)
  //   {
  //     _color_rays = ColorSamples();
  //   }
  //   else
  //   {
  //     _color_rays.rgb = rgb.view({-1, 3}).to(_device); // 颜色值

  //     _pose = get_pose(idx, dataparser::DataType::RawColor).to(_device);

  //     auto camera_ray_results =
  //         dataparser_ptr_->sensor_.camera.generate_rays(_pose); // 获取相机射线(包含光心、光线方向)
  //     _color_rays.origin = camera_ray_results[0];               // 光心
  //     _color_rays.direction = camera_ray_results[1];            // 光线方向
  //   }
  //   return true;
  // }

  torch::Tensor DataLoader::get_pose(int idx, const int &pose_type)
  {
    // return pose matrix at idx with shape [3, 4]
    return dataparser_ptr_->get_pose(idx, pose_type).slice(0, 0, 3);
  }

  void DataLoader::colorPointCloudFromImage(
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
      const cv::Mat &image,
      const Eigen::Matrix<double, 3, 4> &proj_mat,      // 投影矩阵
      const Eigen::Matrix<double, 4, 4> &Tr_velo_to_cam // 变换矩阵从 Velodyne 到 Camera
  )
  {
    if (!cloud || image.empty())
      return;

#pragma omp parallel for
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
      auto &point = cloud->points[i];
      Eigen::Vector4d pt_velo(point.x, point.y, point.z, 1.0);
      Eigen::Vector3d pt_cam = proj_mat * Tr_velo_to_cam * pt_velo;

      if (pt_cam(2) <= 0.1)
        continue;

      int u = static_cast<int>(pt_cam(0) / pt_cam(2));
      int v = static_cast<int>(pt_cam(1) / pt_cam(2));

      if (u >= 0 && u < image.cols && v >= 0 && v < image.rows)
      {
        auto color = image.at<cv::Vec3b>(v, u);
        point.r = color[2];
        point.g = color[1];
        point.b = color[0];
      }
    }
  }
} // namespace dataloader