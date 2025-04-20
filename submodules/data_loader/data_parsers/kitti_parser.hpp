#pragma once
#include "base_parser.h"
#include "submodules/utils/coordinates.h"
#include "submodules/utils/sensor_utils/cameras.hpp"
#include <pcl/io/ply_io.h>

namespace dataparser
{
  struct Kitti : DataParser
  {
    explicit Kitti(const std::filesystem::path &_dataset_lidar_path,
                   const torch::Device &_device = torch::kCPU,
                   const bool &_preload = true, const float &_res_scale = 1.0)
        : DataParser(_dataset_lidar_path, _device, _preload, _res_scale,
                     coords::SystemType::OpenCV)
    {
      depth_type_ = DepthType::BIN;
      auto base_path = (dataset_path_).lexically_normal();
      calib_path_ = base_path / "calib.txt"; // 标定文件路径
      pose_path_ = base_path / "poses.txt";  // 位姿文件路径
      color_path_ = base_path / "image_2";   // 图片文件路径
      depth_path_ = base_path / "velodyne";  // 激光点云文件路径

      load_intrinsics(); // 加载相机内参
      load_calib();      // 加载标定文件
      load_data();       // 加载数据
    }

    torch::Tensor T_C0_L, T_C0_C2;
    void load_data() override
    {

      auto T_W_C0 = load_poses(pose_path_, false, 2)[0];                              // 加载位姿
      // std::cout << "T_W_C0: " << T_W_C0 << std::endl;
      TORCH_CHECK(T_W_C0.size(0) > 0);
      // auto T_W_C0 = coords::change_world_system(T_C0_C0, coords::SystemType::Kitti);
      color_poses_ = T_W_C0.matmul(T_C0_C2);
      depth_poses_ = T_W_C0.matmul(T_C0_L);

      load_colors(".png", "", false, true);                                            // 加载图片
      TORCH_CHECK(color_poses_.size(0) == raw_color_filelists_.size());                
      load_depths(".bin", "", false, true);                                            // 加载激光点云                      
      TORCH_CHECK(depth_poses_.size(0) == raw_depth_filelists_.size());
    }

    void load_calib() override
    {
      std::ifstream file(calib_path_);
      if (!file.is_open())
      {
        std::cerr << "Failed to open calibration file: " << calib_path_ << std::endl;
        return;
      }

      std::string line;
      while (std::getline(file, line))
      {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;

        std::vector<float> values((std::istream_iterator<float>(ss)), std::istream_iterator<float>()); // 读取一行数据

        if (values.size() != 12)
          continue;

        if (tag == "Tr:")
        {
          Tr = Eigen::Matrix<float, 4, 4>::Identity();
          Tr.block<3, 4>(0, 0) = Eigen::Map<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>(values.data());
        }
        else
        {
          Eigen::Matrix<float, 3, 4> mat = Eigen::Map<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>(values.data());
          if (tag == "P0:")
            continue;
          else if (tag == "P1:")
            continue;
          else if (tag == "P2:")
            P = mat;
          else if (tag == "P3:")
            continue;
        }
      }
    }
    void load_intrinsics() override
    {
      // open calibration file
      std::ifstream calib_file(calib_path_);
      if (!calib_file)
      {
        throw std::runtime_error("Could not open calibration file: " + calib_path_.string()); // 标定文件打开失败
      }

      std::string line;
      T_C0_L = torch::eye(4);   // 变换矩阵从 Velodyne 到 Camera
      T_C0_C2 = torch::eye(4);  // 变换矩阵从 Camera2(左边彩色相机) 到 Camera0(左边灰度相机)
      while (std::getline(calib_file, line))
      {
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token == "P2:")
        { // 右相机的投影矩阵
          int index = 0;
          torch::Tensor baseline = torch::zeros(3);
          while (iss >> token)
          {
            if (index == 0)
            {
              sensor_.camera.fx = std::stof(token);
            }
            else if (index == 2)
            {
              sensor_.camera.cx = std::stof(token);
            }
            else if (index == 3)
            {
              T_C0_C2[0][3] = -std::stof(token) / sensor_.camera.fx;
            }
            else if (index == 5)
            {
              sensor_.camera.fy = std::stof(token);
            }
            else if (index == 6)
            {
              sensor_.camera.cy = std::stof(token);
            }
            else if (index == 7)
            {
              T_C0_C2[1][3] = -std::stof(token) / sensor_.camera.fy;
            }
            else if (index == 11)
            {
              T_C0_C2[2][3] = -std::stof(token);
            }
            index++;
          }
        }
        else if (token == "Tr:")
        {
          int i = 0;
          int j = 0;
          float value;
          while (iss >> value)
          {
            T_C0_L[i][j] = value;
            j++;
            if ((j % 4) == 0)
            {
              i++;
              j = 0;
            }
          }
        }
      }
      sensor_.camera.width = 1226;
      sensor_.camera.height = 370;
      depth_scale_inv_ = 1.0;
      // print out cameras
      // std::cout << "fx: " << sensor_.camera.fx << ", fy: " << sensor_.camera.fy
      //           << ", cx: " << sensor_.camera.cx << ", cy: " << sensor_.camera.cy
      //           << "\n";
      // std::cout << "T_C0_L:\n"
      //           << T_C0_L << "\n";
    }

    std::vector<at::Tensor> get_distance_ndir_zdirn(const int &idx) override
    {
      /**
       * @description:
       * @return {distance, ndir, dir_norm}, where ndir.norm = 1;
                 {[height width 1], [height width 3], [height width 1]}
       */

      auto pointcloud = get_depth_image(idx);
      // [height width 1]
      auto distance = pointcloud.norm(2, -1, true);
      auto ndir = pointcloud / distance;
      return {distance, ndir, distance};
    }
  };
} // namespace dataparser