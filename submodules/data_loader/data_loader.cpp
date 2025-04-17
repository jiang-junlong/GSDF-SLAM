#include <fstream>
#include <iostream>
#include <memory>

#include "pcl/io/pcd_io.h"
#include "submodules/data_loader/data_loader.h"
#include "submodules/utils/bin_utils/endian.h"
#include "submodules/utils/ray_utils/ray_utils.h"
#include "submodules/data_loader/data_parsers/kitti_parser.hpp"
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
    // 读取图像并归一化
    std::string imgfile = dataparser_ptr_->dataset_path_ / dataparser_ptr_->get_file(idx, dataparser::DataType::RawColor);
    image = cv::imread(imgfile, cv::IMREAD_COLOR); // 直接读取为 RGB 格式
    if (image.empty())
    {
      std::cerr << "Could not read image: " << imgfile << "\n";
      return false;
    }
    image.convertTo(image, CV_32FC3, 1.0f / 255.0f);
    torch::Tensor image_tensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kFloat32).to(device_);

    // 读取点云数据 (.bin 文件)
    torch::Tensor lidar_pose = get_pose(idx, dataparser::DataType::RawDepth).to(device_);
    std::string infile = dataparser_ptr_->dataset_path_ / dataparser_ptr_->get_file(idx, dataparser::DataType::RawDepth);
    std::ifstream input(infile.c_str(), std::ios::in | std::ios::binary);
    if (!input)
    {
      std::cerr << "Could not read file: " << infile << "\n";
      return false;
    }

    const size_t kMaxNumberOfPoints = 1e6; // 最大点数
    std::vector<pcl::PointXYZ> points;
    points.reserve(kMaxNumberOfPoints);
    pcl::PointXYZ point;
    while (input.read(reinterpret_cast<char *>(&point.x), 3 * sizeof(float)) && !input.eof())
    {
      float intensity;
      input.read(reinterpret_cast<char *>(&intensity), sizeof(float));
      points.push_back(point);
    }
    input.close();

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
    torch::Tensor proj_mat = torch::from_blob(P.data(), {12}, torch::kFloat32).to(device_);
    torch::Tensor Tr_velo_to_cam = torch::from_blob(Tr.data(), {16}, torch::kFloat32).to(device_);
    // 处理点云着色（并行化，批量处理）
    colorize_launcher(point_cloud, color, proj_mat, Tr_velo_to_cam, lidar_pose, image_tensor);

    // 用以显示点云，查看染色结果是否正确
    // if (point_cloud.device().is_cuda())
    //   point_cloud = point_cloud.cpu();
    // if (color.device().is_cuda())
    //   color = color.cpu();

    // const int N = point_cloud.size(0);
    // auto pcl_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    // pcl_cloud->points.resize(N);
    // pcl_cloud->width = N;
    // pcl_cloud->height = 1;
    // pcl_cloud->is_dense = false;
    // pcl::visualization::PCLVisualizer viewer("Viewer");
    // viewer.setBackgroundColor(0, 0, 0);
    // viewer.initCameraParameters();
    // viewer.addCoordinateSystem(1.0);
    // for (int i = 0; i < N; ++i)
    // {
    //   pcl::PointXYZRGB pt;
    //   pt.x = point_cloud[i][0].item<float>();
    //   pt.y = point_cloud[i][1].item<float>();
    //   pt.z = point_cloud[i][2].item<float>();

    //   pt.r = static_cast<uint8_t>(color[i][0].item<float>() * 255.0f);
    //   pt.g = static_cast<uint8_t>(color[i][1].item<float>() * 255.0f);
    //   pt.b = static_cast<uint8_t>(color[i][2].item<float>() * 255.0f);

    //   pcl_cloud->points[i] = pt;
    // }
    // if (!viewer.updatePointCloud(pcl_cloud, "colored_cloud"))
    // {
    //   viewer.addPointCloud(pcl_cloud, "colored_cloud");
    // }

    // viewer.spin();
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