#pragma once

#include <torch/torch.h>

#include "data_parsers/base_parser.h"
#include "utils/sensor_utils/sensors.hpp"

namespace dataloader {
class DataLoader {
public:
  typedef std::shared_ptr<DataLoader> Ptr;
  explicit DataLoader(const std::string &dataset_path,
                      const int &_dataset_type = 0,
                      const torch::Device &_device = torch::kCPU,
                      const bool &_preload = false,
                      const float &_res_scale = 1.0,
                      const sensor::Sensors &_sensor = sensor::Sensors());

  torch::Device device_ = torch::kCPU;

  dataparser::DataParser::Ptr dataparser_ptr_;

  bool get_item(int idx, torch::Tensor &_pose,
                    PointCloudT &colored_points,
                    cv::Mat &image);

  bool get_item(int idx, torch::Tensor &_pose, PointCloudT &_points);
  bool get_item(int idx, torch::Tensor &_pose, DepthSamples &_depth_rays,
                     ColorSamples &_color_rays,
                     const torch::Device &_device = torch::Device(torch::kCPU));

  bool get_item(int idx, torch::Tensor &_pose, DepthSamples &_depth_rays,
                     const torch::Device &_device = torch::Device(torch::kCPU));
  bool get_item(int idx, torch::Tensor &_pose, ColorSamples &_color_rays,
                     const torch::Device &_device = torch::Device(torch::kCPU));

  torch::Tensor get_pose(int idx, const int &pose_type = 0);

  void DataLoader::colorPointCloudFromImage(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    const cv::Mat &image,
    const Eigen::Matrix<double, 3, 4> &proj_mat,      // 投影矩阵
    const Eigen::Matrix<double, 4, 4> &Tr_velo_to_cam // 变换矩阵从 Velodyne 到 Camera
  );

private:
  int dataset_type_;  // 数据集类型
};
} // namespace dataloader