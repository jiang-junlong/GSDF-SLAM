/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Photo-SLAM.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <unordered_map>
#include <filesystem>
#include <fstream>

#include <torch/torch.h>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include "third_party/colmap/utils/endian.h"
#include "include/gaussian_mapper.h"
#include "viewer/imgui_viewer.h"


int main(int argc, char** argv)
{
    if (argc != 4 && argc != 5)
    {
        std::cerr << std::endl
                  << "Usage: " << argv[0]
                  << " path_to_gaussian_mapping_settings"    /*1*/
                  << " path_to_colmap_data_directory/"       /*2*/
                  << " path_to_output_directory/"            /*3*/
                  << " (optional)no_viewer"                  /*4*/
                  << std::endl;
        return 1;
    }
    bool use_viewer = true;
    if (argc == 5)
        use_viewer = (std::string(argv[4]) == "no_viewer" ? false : true);

    std::string output_directory = std::string(argv[3]);
    if (output_directory.back() != '/')
        output_directory += "/";
    std::filesystem::path output_dir(output_directory);

    // Device
    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }

    // 读取配置文件
    cv::FileStorage fsSettings(_config_path, cv::FileStorage::READ);                
    if (!fsSettings.isOpened()) {
        std::cerr << "ERROR: Wrong path to settings: " << _config_path << "\n";
        exit(-1);
    }
    std::filesystem::path dataset_path = fsSettings["data_path"];  // 数据集路径
    int dataset_type = fsSettings["dataset_type"];
    float res_scale = 1.0f;
    sensor::Sensors sensor;
    if (!fsSettings["camera"].isNone()) {
        sensor.camera.model = fsSettings["camera"]["model"];
        sensor.camera.width = fsSettings["camera"]["width"];
        sensor.camera.height = fsSettings["camera"]["height"];
        sensor.camera.fx = fsSettings["camera"]["fx"];
        sensor.camera.fy = fsSettings["camera"]["fy"];
        sensor.camera.cx = fsSettings["camera"]["cx"];
        sensor.camera.cy = fsSettings["camera"]["cy"];
    
        sensor.camera.set_distortion(
            fsSettings["camera"]["d0"], fsSettings["camera"]["d1"],
            fsSettings["camera"]["d2"], fsSettings["camera"]["d3"],
            fsSettings["camera"]["d4"]);
      }
      if (!fsSettings["extrinsic"].isNone()) {
        cv::Mat cv_T_C_L;
        fsSettings["extrinsic"]["T_C_L"] >> cv_T_C_L;
        cv_T_C_L.convertTo(cv_T_C_L, CV_32FC1);
        sensor.T_C_L =
            torch::from_blob(cv_T_C_L.data, {4, 4}, torch::kFloat32).clone();
    
        cv::Mat cv_T_B_L;
        fsSettings["extrinsic"]["T_B_L"] >> cv_T_B_L;
        cv_T_B_L.convertTo(cv_T_B_L, CV_32FC1);
        sensor.T_B_L =
            torch::from_blob(cv_T_B_L.data, {4, 4}, torch::kFloat32).clone();
    
        sensor.T_B_C = sensor.T_B_L.matmul(sensor.T_C_L.inverse());
      }

    // 创建数据加载器
    data_loader_ptr = std::make_unique<dataloader::DataLoader>(
        dataset_path, dataset_type, device_type, false, res_scale, sensor);

    // Create GaussianMapper
    std::filesystem::path gaussian_cfg_path(argv[1]);
    std::shared_ptr<GaussianMapper> pGausMapper =
        std::make_shared<GaussianMapper>(gaussian_cfg_path, output_dir, 0, device_type);

    // Read the colmap scene
    pGausMapper->setSensorType(MONOCULAR);
    pGausMapper->setColmapDataPath(argv[2]);
    readColmapScene(pGausMapper);

    // Create Gaussian Viewer
    std::thread viewer_thd;
    std::shared_ptr<ImGuiViewer> pViewer;
    if (use_viewer)
    {
        pViewer = std::make_shared<ImGuiViewer>(pGausMapper);
        viewer_thd = std::thread(&ImGuiViewer::run, pViewer.get());
    }

    // Train and save results
    pGausMapper->trainColmap();

    if (use_viewer)
        viewer_thd.join();
    return 0;
}