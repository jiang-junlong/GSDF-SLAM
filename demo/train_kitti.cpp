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
#include "include/gaussian_mapper.h"
#include "viewer/imgui_viewer.h"
#include <pcl/visualization/pcl_visualizer.h>

int main(int argc, char **argv)
{
    // if (argc != 4 && argc != 5)
    // {
    //     std::cerr << std::endl
    //               << "Usage: " << argv[0]
    //               << " path_to_gaussian_mapping_settings"    /*1*/
    //               << " path_to_colmap_data_directory/"       /*2*/
    //               << " path_to_output_directory/"            /*3*/
    //               << " (optional)no_viewer"                  /*4*/
    //               << std::endl;
    //     return 1;
    // }
    bool use_viewer = true;
    // if (argc == 5)
    //     use_viewer = (std::string(argv[4]) == "no_viewer" ? false : true);

    // std::string output_directory = std::string(argv[3]);
    // if (output_directory.back() != '/')
    //     output_directory += "/";
    // std::filesystem::path output_dir(output_directory);

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

    // std::string dataset_path = "/home/jiang/myproject/GSDF-SLAM/dataset/kitti_04/04";
    // std::string output_dir = "/home/jiang/myproject/GSDF-SLAM/result/kitti_04/04";
    std::string dataset_path = "/home/jiang/myproject/GSDF-SLAM/dataset/Oxford-Spires-Dataset/2024-03-12-keble-college-02";
    std::string output_dir = "/home/jiang/myproject/GSDF-SLAM/result/Oxford-Spires-Dataset/2024-03-12-keble-college-02";
    std::string gaussian_cfg_path = "/home/jiang/myproject/GSDF-SLAM/cfg/colmap/gaussian_splatting.yaml";

    // Create GaussianMapper
    // std::filesystem::path gaussian_cfg_path(argv[1]);
    std::shared_ptr<GaussianMapper> pGausMapper = std::make_shared<GaussianMapper>(dataset_path, gaussian_cfg_path, output_dir, 0, device_type);
    std::thread training_thd(&GaussianMapper::run, pGausMapper.get());

    // Create Gaussian Viewer
    std::thread viewer_thd;
    std::shared_ptr<ImGuiViewer> pViewer;
    if (use_viewer)
    {
        pViewer = std::make_shared<ImGuiViewer>(pGausMapper);
        viewer_thd = std::thread(&ImGuiViewer::run, pViewer.get());
    }

    training_thd.join();
    if (use_viewer)
        viewer_thd.join();
    return 0;
}