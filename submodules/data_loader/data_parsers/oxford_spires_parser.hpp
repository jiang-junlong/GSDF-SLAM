#pragma once
#include "base_parser.h"
#include "submodules/utils/coordinates.h"
#include "submodules/utils/sensor_utils/cameras.hpp"
#include <pcl/io/ply_io.h>
#include <yaml-cpp/yaml.h>

namespace dataparser
{
  struct Spires : DataParser
  {
    explicit Spires(const std::filesystem::path &_dataset_path,
                    const torch::Device &_device = torch::kCPU,
                    const bool &_preload = true, const float &_res_scale = 1.0)
        : DataParser(_dataset_path, _device, _preload, _res_scale,
                     coords::SystemType::OpenCV)
    {
      // export undistorted images
      calib_path_ = dataset_path_ / "calibration/cam-lidar-imu.yaml"; // 标定文件路径
      pose_path_ = dataset_path_ / "color_poses.txt";
      depth_pose_path_ = dataset_path_ / "depth_poses.txt";
      color_path_ = dataset_path_ / "undistorted_images/cam0";
      depth_path_ = dataset_path_ / "lidar-clouds"; // origin data
      // /media/chrisliu/T9/Datasets/Oxford_Spires_Dataset/bodleian_library/02
      // └── 02
      //     ├── gt-tum.txt
      //     ├── images
      //     │   ├── cam0
      //     │   ├── cam1
      //     │   └── cam2
      //     └── lidar-clouds
      dataset_name_ = dataset_path_.filename();

      // pose_path_ = dataset_path_ / "gt-tum.txt";
      // color_path_ = dataset_path_ / "images" / "cam0";
      depth_type_ = DepthType::PCD;
      load_calib();
      load_intrinsics();
      load_data();
    }

    torch::Tensor T_B_L, T_C_L;
    torch::Tensor T_C0_L, T_C0_C2;

    std::filesystem::path depth_pose_path_;
    void load_data() override
    {
      // if (!std::filesystem::exists(pose_path_) ||
      //     !std::filesystem::exists(depth_pose_path_) ||
      //     !std::filesystem::exists(color_path_) ||
      //     !std::filesystem::exists(depth_path_))
      // {
      //   pose_path_ = dataset_path_ / "gt-tum.txt";
      //   color_path_ = dataset_path_ / "images" / "cam0";
      //   depth_path_ = dataset_path_ / "lidar-clouds";

      //   auto pose_data = load_poses(pose_path_, false, 3);
      //   auto T_W_B = pose_data[0];
      //   auto T_W_L = T_W_B.matmul(T_B_L);
      //   time_stamps_ = pose_data[1];
      //   TORCH_CHECK(T_W_L.size(0) > 0);
      //   // auto T_W_C0 =
      //   //     coords::change_world_system(T_C0_C0, coords::SystemType::Kitti);
      //   color_poses_ = T_W_L.matmul(T_C_L.inverse());
      //   depth_poses_ = T_W_L;

      //   load_colors(".jpg", "", false, false);
      //   std::cout << "color_poses: " << color_poses_.size(0) << " color_files: "
      //             << raw_color_filelists_.size() << std::endl;
      //   load_depths(".pcd", "", false, false);
      //   std::cout << "depth_poses: " << depth_poses_.size(0)
      //             << " depth_files: " << raw_depth_filelists_.size() << std::endl;

      //   // export undistorted images
      //   // color_path_ = dataset_path_ / "undistorted_images";
      //   // std::filesystem::create_directories(color_path_);
      //   pose_path_ = dataset_path_ / "color_poses.txt";
      //   std::ofstream color_pose_file(pose_path_);
      //   for (int i = 0; i < raw_color_filelists_.size(); i++)
      //   {
      //     // auto color_image = get_image_cv_mat(i);
      //     // auto undistorted_img = sensor_.camera.undistort(color_image);
      //     // auto undistorted_img_path =
      //     //     color_path_ / raw_color_filelists_[i].filename();
      //     // cv::imwrite(undistorted_img_path, undistorted_img);

      //     auto T_W_C = color_poses_[i];
      //     for (int i = 0; i < 4; i++)
      //     {
      //       for (int j = 0; j < 4; j++)
      //       {
      //         color_pose_file << T_W_C[i][j].item<float>() << " ";
      //       }
      //       color_pose_file << "\n";
      //     }
      //   }
      //   // export align pose depth
      //   depth_path_ = dataset_path_ / "depths";
      //   std::filesystem::create_directories(depth_path_);
      //   std::ofstream depth_pose_file(depth_pose_path_);
      //   for (int i = 0; i < raw_depth_filelists_.size(); i++)
      //   {
      //     // copy depth file to undistorted_images
      //     std::filesystem::copy_file(
      //         raw_depth_filelists_[i],
      //         depth_path_ / raw_depth_filelists_[i].filename(),
      //         std::filesystem::copy_options::overwrite_existing);

      //     auto T_W_L = depth_poses_[i];
      //     for (int i = 0; i < 4; i++)
      //     {
      //       for (int j = 0; j < 4; j++)
      //       {
      //         depth_pose_file << T_W_L[i][j].item<float>() << " ";
      //       }
      //       depth_pose_file << "\n";
      //     }
      //   }
      // }

      time_stamps_ = torch::Tensor(); // reset time_stamps
      auto T_B_C = T_B_L.matmul(T_C_L.inverse());
      color_poses_ = load_poses(pose_path_, false, 3)[0];
      color_poses_ = color_poses_.matmul(T_B_C);
      TORCH_CHECK(color_poses_.size(0) > 0);
      
      depth_poses_ = load_poses(depth_pose_path_, false, 3)[0];
      depth_poses_ = depth_poses_.matmul(T_B_L);
      TORCH_CHECK(depth_poses_.size(0) > 0);

      load_colors(".jpg", "", false, true);
      TORCH_CHECK(color_poses_.size(0) == raw_color_filelists_.size());
      load_depths(".pcd", "", false, true);
      TORCH_CHECK(depth_poses_.size(0) == raw_depth_filelists_.size());
    }

    // 将xyz和四元数(q_xyzw)转换为4x4变换矩阵
    torch::Tensor xyz_q_xyzw_to_matrix(const std::vector<double> &t_xyz_q_xyzw)
    {
      Eigen::Quaterniond q(t_xyz_q_xyzw[6], t_xyz_q_xyzw[3], t_xyz_q_xyzw[4], t_xyz_q_xyzw[5]);
      q.normalize();
      Eigen::Matrix3d R = q.toRotationMatrix();
      Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T = Eigen::Matrix4d::Identity();
      T.block<3, 3>(0, 0) = R;
      T.block<3, 1>(0, 3) << t_xyz_q_xyzw[0], t_xyz_q_xyzw[1], t_xyz_q_xyzw[2];
      return torch::from_blob(T.data(), {4, 4}, torch::kFloat64).clone().to(torch::kFloat32);
    }

    void load_calib() override
    {
      YAML::Node config = YAML::LoadFile(calib_path_);
      auto T_base_lidar_t_xyz_q_xyzw = config["T_base_lidar_t_xyz_q_xyzw"].as<std::vector<double>>();
      T_B_L = xyz_q_xyzw_to_matrix(T_base_lidar_t_xyz_q_xyzw);
      auto T_cam_lidar_t_xyz_q_xyzw_overwrite = config["cam0"]["T_cam_lidar_t_xyz_q_xyzw_overwrite"].as<std::vector<double>>();
      T_C_L = xyz_q_xyzw_to_matrix(T_cam_lidar_t_xyz_q_xyzw_overwrite);
      for (int i = 0; i < 4; ++i)
      {
        for (int j = 0; j < 4; ++j)
        {
          Tr(i, j) = T_C_L[i][j].item<float>();
        }
      }
    }
    void load_intrinsics() override
    {
      YAML::Node config = YAML::LoadFile(calib_path_);
      YAML::Node k_rect = config["cam0"]["K_rect"];
      Eigen::Matrix3f K;
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
        {
          K(i, j) = k_rect[i][j].as<float>();
        }
      }
      P.block<3, 3>(0, 0) = K;
      P.col(3).setZero();
      sensor_.camera.width = 1440;
      sensor_.camera.height = 1080;
      sensor_.camera.fx = K(0, 0);
      sensor_.camera.fy = K(1, 1);
      sensor_.camera.cx = K(0, 2);
      sensor_.camera.cy = K(1, 2);
      depth_scale_inv_ = 1.0;
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