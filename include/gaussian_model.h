/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 *
 * This file is Derivative Works of Gaussian Splatting,
 * created by Longwei Li, Huajian Huang, Hui Cheng and Sai-Kit Yeung in 2023,
 * as part of Photo-SLAM.
 */

#pragma once

#include <memory>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>

#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "types.h"
#include "general_utils.h"
#include "sh_utils.h"
#include "tensor_utils.h"
#include "gaussian_parameters.h"

#include "submodules/Sophus/sophus/se3.hpp"
#include "submodules/simple-knn/spatial.h"
#include "submodules/utils/ply_utils/tinyply.h"

#define GAUSSIAN_MODEL_TENSORS_TO_VEC                        \
    this->Tensor_vec_xyz_ = {this->xyz_};                    \
    this->Tensor_vec_feature_dc_ = {this->features_dc_};     \
    this->Tensor_vec_feature_rest_ = {this->features_rest_}; \
    this->Tensor_vec_opacity_ = {this->opacity_};            \
    this->Tensor_vec_scaling_ = {this->scaling_};            \
    this->Tensor_vec_rotation_ = {this->rotation_};

#define GAUSSIAN_MODEL_INIT_TENSORS(device_type)                                             \
    this->xyz_ = torch::empty(0, torch::TensorOptions().device(device_type));                \
    this->features_dc_ = torch::empty(0, torch::TensorOptions().device(device_type));        \
    this->features_rest_ = torch::empty(0, torch::TensorOptions().device(device_type));      \
    this->scaling_ = torch::empty(0, torch::TensorOptions().device(device_type));            \
    this->rotation_ = torch::empty(0, torch::TensorOptions().device(device_type));           \
    this->opacity_ = torch::empty(0, torch::TensorOptions().device(device_type));            \
    this->max_radii2D_ = torch::empty(0, torch::TensorOptions().device(device_type));        \
    this->xyz_gradient_accum_ = torch::empty(0, torch::TensorOptions().device(device_type)); \
    this->denom_ = torch::empty(0, torch::TensorOptions().device(device_type));              \
    GAUSSIAN_MODEL_TENSORS_TO_VEC

class GaussianModel
{
public:
    GaussianModel(const int sh_degree);
    GaussianModel(const GaussianModelParams &model_params);

    torch::Tensor getScalingActivation();
    torch::Tensor getRotationActivation();
    torch::Tensor getXYZ();
    torch::Tensor getFeatures();
    torch::Tensor getOpacityActivation();
    torch::Tensor getCovarianceActivation(int scaling_modifier = 1);

    void oneUpShDegree();
    void setShDegree(const int sh);
    void createFromPcd(
        torch::Tensor &point_cloud,
        torch::Tensor &color,
        const float spatial_lr_scale);

    void increasePcd(std::vector<float> points, std::vector<float> colors, const int iteration);
    void increasePcd(torch::Tensor &new_point_cloud, torch::Tensor &new_colors, const int iteration);
    void increasePcd(torch::Tensor &new_point_cloud, torch::Tensor &new_colors, const int iteration, const float spatial_lr_scale);

    void trainingSetup(const GaussianOptimizationParams &training_args);
    float updateLearningRate(int step);
    void setPositionLearningRate(float position_lr);
    void setFeatureLearningRate(float feature_lr);
    void setOpacityLearningRate(float opacity_lr);
    void setScalingLearningRate(float scaling_lr);
    void setRotationLearningRate(float rot_lr);

    void resetOpacity();
    torch::Tensor replaceTensorToOptimizer(torch::Tensor &t, int tensor_idx);

    void prunePoints(torch::Tensor &mask);

    void densificationPostfix(
        torch::Tensor &new_xyz,
        torch::Tensor &new_features_dc,
        torch::Tensor &new_features_rest,
        torch::Tensor &new_opacities,
        torch::Tensor &new_scaling,
        torch::Tensor &new_rotation,
        torch::Tensor &new_exist_since_iter);

    void densifyAndSplit(
        torch::Tensor &grads,
        float grad_threshold,
        float scene_extent,
        int N = 2);

    void densifyAndClone(
        torch::Tensor &grads,
        float grad_threshold,
        float scene_extent);

    void densifyAndPrune(
        float max_grad,
        float min_opacity,
        float extent,
        int max_screen_size);

    void addDensificationStats(
        torch::Tensor &viewspace_point_tensor,
        torch::Tensor &update_filter);

    // void increasePointsIterationsOfExistence(const int i = 1);

    void loadPly(std::filesystem::path ply_path);
    void savePly(std::filesystem::path result_path);
    void saveSparsePointsPly(std::filesystem::path result_path);

    float percentDense();
    void setPercentDense(const float percent_dense);

protected:
    float exponLrFunc(int step);

public:
    torch::DeviceType device_type_;

    int active_sh_degree_;
    int max_sh_degree_;

    torch::Tensor xyz_;           // 位置
    torch::Tensor features_dc_;   // 球谐系数（常数颜色部分）
    torch::Tensor features_rest_; // 球谐系数（其它颜色部分）
    torch::Tensor scaling_;       // 缩放
    torch::Tensor rotation_;      // 旋转
    torch::Tensor opacity_;       // 不透明度
    torch::Tensor max_radii2D_;   // 最大半径2D
    torch::Tensor xyz_gradient_accum_;
    torch::Tensor denom_;
    torch::Tensor exist_since_iter_;

    std::vector<torch::Tensor> Tensor_vec_xyz_,
        Tensor_vec_feature_dc_,
        Tensor_vec_feature_rest_,
        Tensor_vec_opacity_,
        Tensor_vec_scaling_,
        Tensor_vec_rotation_;

    std::shared_ptr<torch::optim::Adam> optimizer_;
    float percent_dense_;
    float spatial_lr_scale_;

    torch::Tensor sparse_points_xyz_;
    torch::Tensor sparse_points_color_;

protected:
    float lr_init_;
    float lr_final_;
    int lr_delay_steps_;
    float lr_delay_mult_;
    int max_steps_;

    std::mutex mutex_settings_;
};