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

int main(int argc, char** argv)
{
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
    std::string dataset_path = "/home/uav/myproject/GSDF-SLAM/dataset/kitti_04/04";
    dataloader::DataLoader test=dataloader::DataLoader(dataset_path, 3, device_type);;
    
    pcl::visualization::PCLVisualizer viewer("Viewer");
    viewer.setBackgroundColor(0, 0, 0);
    viewer.initCameraParameters();
    viewer.addCoordinateSystem(1.0);
    
    for (int i = 0; i < test.dataparser_ptr_->raw_depth_filelists_.size(); ++i) {
        pcl::PointCloud<pcl::PointXYZRGB> colored_points;
        cv::Mat image;

        torch::Tensor lidar_pose, cam_pose;

        test.get_item(i, lidar_pose, cam_pose, colored_points, image);
    
        if (!viewer.updatePointCloud(colored_points.makeShared(), "colored_cloud")) {
            viewer.addPointCloud(colored_points.makeShared(), "colored_cloud");
        }
        cv::imshow("image", image);
        cv::waitKey(1);
        viewer.spinOnce(10);  // 更流畅
        std::this_thread::sleep_for(std::chrono::milliseconds(30));  // 控帧率
    }
    return 0;
}