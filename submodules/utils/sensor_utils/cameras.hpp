#pragma once
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

namespace sensor {

static torch::Tensor get_image_coords(const int &height, const int &width,
                                      const float &pixel_offset = 0.5) {
  // [height width 2], stored as (y, x) coordinates
  auto image_coords =
      torch::meshgrid({torch::arange(height), torch::arange(width)}, "ij");
  return torch::stack(image_coords, -1) + pixel_offset;
}

static torch::Tensor get_image_coords_zdir(const int &height, const int &width,
                                           const float &fx, const float &fy,
                                           const float &cx, const float &cy,
                                           torch::Device device = torch::kCPU,
                                           const float &pixel_offset = 0.5) {
  // [height width 2]
  auto vu = get_image_coords(height, width).to(device);
  // [height width]
  auto pt_x = (vu.select(2, 1) - cx) / fx;
  auto pt_y = (vu.select(2, 0) - cy) / fy;
  auto pt_z = torch::ones_like(pt_x);
  // [height width 3]
  return torch::stack({pt_x, pt_y, pt_z}, -1);
}

static std::vector<torch::Tensor>
get_image_coords_ndir(const int &height, const int &width, const float &fx,
                      const float &fy, const float &cx, const float &cy,
                      torch::Device device = torch::kCPU,
                      const float &pixel_offset = 0.5) {
  // [height width 3]
  auto zdir = get_image_coords_zdir(height, width, fx, fy, cx, cy, device);
  // [height width 1]
  auto zdir_norm = zdir.norm(2, -1, true);
  auto ndir = zdir / zdir_norm;
  return {ndir, zdir_norm};
}

// 相机传感器模型：相机参数和一些相关函数
struct Cameras {
  float fx;
  float fy;
  float cx;
  float cy;
  int width;
  int height;

  // please set distortion parameters by set_distortion
  int model = 0; // 0: pinhole; 1:equidistant;
  float d0 = 0;
  float d1 = 0;
  float d2 = 0;
  float d3 = 0;
  float d4 = 0;

  float focal_length_;

  void set_distortion(const float &_d0, const float &_d1 = 0,
                      const float &_d2 = 0, const float &_d3 = 0,
                      const float &_d4 = 0) {
    d0 = _d0;
    d1 = _d1;
    d2 = _d2;
    d3 = _d3;
    d4 = _d4;

    cv::Mat K =
        (cv::Mat_<float>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);

    cv::Mat new_K;

    if (model == 0) {
      cv::Mat D = (cv::Mat_<float>(1, 5) << d0, d1, d2, d3, d4);

      new_K = cv::getOptimalNewCameraMatrix(K, D, cv::Size(width, height), 0,
                                            cv::Size(width, height), 0, true);
      std::cout << "origin K:\n" << K << "\n";
      std::cout << "undistort K:\n" << new_K << "\n";
      cv::initUndistortRectifyMap(
          K,                           // Intrinsics (distorted image)
          D,                           // Distortion (distorted image)
          cv::Mat_<double>::eye(3, 3), // Rectification (distorted image)
          new_K,                       // New camera matrix (undistorted image)
          cv::Size(width, height),     // Image resolution (distorted image)
          CV_16SC2,                    // Map type
          undist_map_x_,               // Undistortion map for X axis
          undist_map_y_                // Undistortion map for Y axis
      );
    } else if (model == 1) {
      cv::Mat D = (cv::Mat_<float>(1, 4) << d0, d1, d2, d3);

      new_K = cv::getOptimalNewCameraMatrix(K, D, cv::Size(width, height), 0,
                                            cv::Size(width, height), 0, true);
      std::cout << "origin K:\n" << K << "\n";
      std::cout << "undistort K:\n" << new_K << "\n";
      cv::fisheye::initUndistortRectifyMap(
          K,                           // Intrinsics (distorted image)
          D,                           // Distortion (distorted image)
          cv::Mat_<double>::eye(3, 3), // Rectification (distorted image)
          new_K,                       // New camera matrix (undistorted image)
          cv::Size(width, height),     // Image resolution (distorted image)
          CV_16SC2,                    // Map type
          undist_map_x_,               // Undistortion map for X axis
          undist_map_y_                // Undistortion map for Y axis
      );
    } else {
      throw std::runtime_error("Invalid camera model");
    }
    distortion_ = true;

    fx = new_K.at<float>(0, 0);
    fy = new_K.at<float>(1, 1);
    cx = new_K.at<float>(0, 2);
    cy = new_K.at<float>(1, 2);
    d0 = d1 = d2 = d3 = d4 = 0;
  }

  cv::Mat undistort(const cv::Mat &_img) const {
    cv::Mat undist_img;
    if (distortion_)
      cv::remap(_img, undist_img, undist_map_x_, undist_map_y_,
                cv::INTER_LINEAR);
    else
      undist_img = _img.clone();
    return undist_img;
  }

  std::vector<torch::Tensor>
  generate_rays_from_coords(const torch::Tensor &_coords,
                            const float &_scale = 1.0) const {
    TORCH_CHECK(_coords.dtype() == torch::kFloat32,
                "Input coordinates must be float32");
    auto y = _coords.select(-1, 0);
    auto x = _coords.select(-1, 1);
    // [height width 3]
    auto zrays =
        torch::stack({(x - _scale * cx) / (_scale * fx),
                      (y - _scale * cy) / (_scale * fy), torch::ones_like(x)},
                     -1);
    auto rays_norm = torch::norm(zrays, 2, -1, true);
    auto nrays = zrays / rays_norm;
    return {nrays, rays_norm};
  }

  std::vector<torch::Tensor> generate_rays(const torch::Tensor &_pose,   // 相机位姿
                                           const float &_scale = 1.0) {
    if (_scale != res_scale_) { // 检查分辨率是否变化
      camera_ray_results_ = generate_rays_from_coords(
          get_image_coords(_scale * height, _scale * width), _scale);
      camera_rays_ = camera_ray_results_[0].view({-1, 3});
      camera_ray_norms_ = camera_ray_results_[1].view({-1, 1});
      res_scale_ = _scale;
    }

    auto pos = _pose.slice(1, 3, 4).view({1, 3});                 // 相机位置
    auto ray_o = pos.expand({camera_rays_.size(0), 3});           // 复制相机位置
    auto rot = _pose.slice(1, 0, 3);                              // 相机旋转矩阵
    auto ray_d = camera_rays_.to(_pose.device()).matmul(rot.t()); // 将相机坐标系下的射线变换到世界坐标系

    return {ray_o, ray_d, camera_ray_norms_}; // 返回光心、方向和归一化的射线
  }

private:
  float res_scale_ = 0.0;
  std::vector<torch::Tensor> camera_ray_results_;
  torch::Tensor camera_rays_, camera_ray_norms_;

  bool distortion_ = false;
  cv::Mat undist_map_x_, undist_map_y_;
};
} // namespace sensor