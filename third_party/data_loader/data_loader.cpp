#include <fstream>
#include <iostream>
#include <memory>

#include "pcl/io/pcd_io.h"
#include "third_party/data_loader/data_loader.h"
#include "third_party/utils/bin_utils/endian.h"
#include "third_party/utils/ray_utils/ray_utils.h"
#include "third_party/data_loader/data_parsers/kitti_parser.hpp"
#include "third_party/utils/ply_utils/ply_utils_pcl.h"
#include "third_party/utils/ply_utils/ply_utils_torch.h"

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
    default:
      throw std::runtime_error("Unsupported dataset type");
    }
  }

  bool DataLoader::get_item(int idx, torch::Tensor &_pose, pcl::PointCloud<pcl::PointXYZRGB> &colored_points, cv::Mat &image)
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
    colorize_pointcloud(colored_points, image, dataparser_ptr_->P, dataparser_ptr_->Tr);
    return true;
  }

  torch::Tensor DataLoader::get_pose(int idx, const int &pose_type)
  {
    // return pose matrix at idx with shape [3, 4]
    return dataparser_ptr_->get_pose(idx, pose_type).slice(0, 0, 3);
  }

  void DataLoader::colorize_pointcloud(
      pcl::PointCloud<pcl::PointXYZRGB> &cloud,
      const cv::Mat &image,
      const Eigen::Matrix<double, 3, 4> &proj_mat,      // 投影矩阵
      const Eigen::Matrix<double, 4, 4> &Tr_velo_to_cam // 变换矩阵从 Velodyne 到 Camera
  )
  {
    if (cloud.empty() || image.empty())
      return;

#pragma omp parallel for
    for (size_t i = 0; i < cloud.points.size(); ++i)
    {
      auto &point = cloud.points[i];
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