#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <fstream>

void colorPointCloudFromImage(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    const cv::Mat& image,
    const cv::Mat& camera_matrix)
{
    // 遍历点云中的每个点，将其投影到图像平面并赋值颜色
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        pcl::PointXYZRGB& point = cloud->points[i];

        // 获取点云中的每个点的 3D 坐标
        float x = point.x;
        float y = point.y;
        float z = point.z;

        // 将 3D 点通过相机内参矩阵投影到图像平面
        float u = (camera_matrix.at<double>(0, 0) * x + camera_matrix.at<double>(0, 2) * z) / z;
        float v = (camera_matrix.at<double>(1, 1) * y + camera_matrix.at<double>(1, 2) * z) / z;

        // 确保投影点在图像的有效范围内
        if (u >= 0 && u < image.cols && v >= 0 && v < image.rows) {
            // 获取图像中对应位置的像素值（假设是 BGR 图像）
            cv::Vec3b color = image.at<cv::Vec3b>(cv::Point(u, v));
            
            // 将图像像素颜色赋给点云中的点
            point.r = color[2];  // Red
            point.g = color[1];  // Green
            point.b = color[0];  // Blue
        }
    }
}

// bool loadPointCloudFromBin(const std::string& filename, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
// {
//     std::ifstream input(filename, std::ios::binary);
//     if (!input) {
//         std::cerr << "Failed to load point cloud from: " << filename << std::endl;
//         return false;
//     }

//     cloud->clear();

//     // 每个点由3个float和1个float的强度组成
//     while (!input.eof()) {
//         pcl::PointXYZ point;
//         float intensity;
//         input.read(reinterpret_cast<char*>(&point.x), sizeof(float));
//         input.read(reinterpret_cast<char*>(&point.y), sizeof(float));
//         input.read(reinterpret_cast<char*>(&point.z), sizeof(float));
//         input.read(reinterpret_cast<char*>(&intensity), sizeof(float));
        
//         if (input.gcount() < sizeof(float) * 4) break;

//         cloud->push_back(point);
//     }

//     input.close();
//     return true;
// }

bool loadPointCloudFromBin(const std::string& filename, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    std::ifstream input(filename, std::ios::in | std::ios::binary);
    if (!input) {
        std::cerr << "Could not read file: " << filename << std::endl;
        return false;
    }

    cloud->clear();
    size_t point_count = 0;  // 计数读取的点数
    while (input.is_open() && !input.eof()) {
        pcl::PointXYZ point;
        float intensity;

        // 逐个读取 x, y, z 和 intensity
        input.read(reinterpret_cast<char*>(&point.x), sizeof(float));
        input.read(reinterpret_cast<char*>(&point.y), sizeof(float));
        input.read(reinterpret_cast<char*>(&point.z), sizeof(float));
        input.read(reinterpret_cast<char*>(&intensity), sizeof(float));

        // 如果读取的字节少于预期，说明文件读取出了问题
        if (input.gcount() < sizeof(float) * 4) {
            break;
        }

        cloud->push_back(point);
        point_count++;

        // 打印调试信息，确认每个点都被正确读取
        if (point_count % 1000 == 0) {
            std::cout << "Read " << point_count << " points...\n";
        }
    }

    input.close();
    
    // 如果没有读取到任何点，输出错误信息
    if (point_count == 0) {
        std::cerr << "No valid points were read from the file." << std::endl;
        return false;
    }

    std::cout << "Successfully read " << point_count << " points from " << filename << std::endl;
    return true;
}


int main()
{
    // 读取点云数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

    // 假设点云文件是 .bin 格式
    std::string pointCloudFile = "/home/uav/myproject/GSDF-SLAM/dataset/kitti_04/04/velodyne/000000.bin";
    if (!loadPointCloudFromBin(pointCloudFile, cloud)) {
        return -1;
    }
    std::cout << "Loaded " << cloud->points.size() << " points from " << pointCloudFile << std::endl;
    // 创建一个新的点云用于存储带颜色的点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::copyPointCloud(*cloud, *colored_cloud);

    // 读取图像
    cv::Mat image = cv::imread("./dataset/kitti_04/04/image_2/000000.png");

    // 相机内参矩阵（假设已经知道）
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 7.070912000000e+02, 0.0, 6.018873000000e+02, 0.0, 7.070912000000e+02, 1.831104000000e+02, 0.0, 0.0, 1.0);

    // 将点云与图像结合，给点云上色
    colorPointCloudFromImage(colored_cloud, image, camera_matrix);

    // 可视化带颜色的点云
    pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
    viewer.addPointCloud(cloud, "colored_cloud");
    viewer.spin();

    // 保存为 .ply 文件
    pcl::io::savePLYFile("colored_point_cloud.ply", *colored_cloud);

    return 0;
}
