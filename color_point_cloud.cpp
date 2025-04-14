#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <fstream>
#include <Eigen/Dense>

void loadCalibrationFile(const std::string &calib_file,
                         Eigen::Matrix<double, 3, 4> &P0,
                         Eigen::Matrix<double, 3, 4> &P1,
                         Eigen::Matrix<double, 3, 4> &P2,
                         Eigen::Matrix<double, 3, 4> &P3,
                         Eigen::Matrix<double, 4, 4> &Tr)
{
    std::ifstream file(calib_file);
    if (!file.is_open())
    {
        std::cerr << "Failed to open calibration file: " << calib_file << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;

        std::vector<double> values((std::istream_iterator<double>(ss)), std::istream_iterator<double>()); // 读取一行数据

        if (values.size() != 12)
            continue;

        if (tag == "Tr:")
        {
            Tr = Eigen::Matrix<double, 4, 4>::Identity();
            Tr.block<3, 4>(0, 0) = Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(values.data());
        }
        else
        {
            Eigen::Matrix<double, 3, 4> mat = Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(values.data());
            if (tag == "P0:")
                P0 = mat;
            else if (tag == "P1:")
                P1 = mat;
            else if (tag == "P2:")
                P2 = mat;
            else if (tag == "P3:")
                P3 = mat;
        }
    }
}

void colorPointCloudFromImage(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    const cv::Mat &image,
    const Eigen::Matrix<double, 3, 4> &proj_mat,      // 投影矩阵
    const Eigen::Matrix<double, 4, 4> &Tr_velo_to_cam // 变换矩阵从 Velodyne 到 Camera
)
{
    if (!cloud || image.empty())
        return;

#pragma omp parallel for
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        auto &point = cloud->points[i];
        Eigen::Vector4d pt_velo(point.x, point.y, point.z, 1.0);
        Eigen::Vector3d pt_cam = proj_mat * Tr_velo_to_cam * pt_velo;

        if (pt_cam(2) <= 0.1)
        {
            // point.r = 0;
            // point.g = 0;
            // point.b = 255; // 用蓝色表示非法点
            continue;
        }

        int u = static_cast<int>(pt_cam(0) / pt_cam(2));
        int v = static_cast<int>(pt_cam(1) / pt_cam(2));

        if (u >= 0 && u < image.cols && v >= 0 && v < image.rows)
        {
            auto color = image.at<cv::Vec3b>(v, u);
            point.r = color[2];
            point.g = color[1];
            point.b = color[0];
        }
        // else
        // {
        //     // 不在图像视野内的点标红
        //     point.r = 255;
        //     point.g = 0;
        //     point.b = 0;
        // }
    }
}

bool loadKITTIBinCloud(const std::string &filename, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    std::ifstream input(filename, std::ios::binary);
    if (!input)
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    cloud->clear();

    struct PointXYZI
    {
        float x, y, z, intensity;
    };

    PointXYZI point;
    size_t point_index = 0;
    while (input.read(reinterpret_cast<char *>(&point), sizeof(PointXYZI)))
    {
        pcl::PointXYZ p;
        p.x = point.x;
        p.y = point.y;
        p.z = point.z;
        cloud->push_back(p);

        ++point_index;
    }

    input.close();

    if (cloud->empty())
    {
        std::cerr << "No points read from file: " << filename << std::endl;
        return false;
    }
    return true;
}

int main()
{
    std::string pointCloudFile = "/home/uav/myproject/GSDF-SLAM/dataset/kitti_04/04/velodyne/000000.bin"; // 点云文件，Kitti 数据集，.bin 格式
    std::string calib_file = "/home/uav/myproject/GSDF-SLAM/dataset/kitti_04/04/calib.txt"; // 标定文件
    cv::Mat imageL = cv::imread("./dataset/kitti_04/04/image_2/000000.png"); // 左图像
    cv::Mat imageR = cv::imread("./dataset/kitti_04/04/image_3/000000.png"); // 右图像
    if (imageL.empty() || imageR.empty())
    {
        std::cerr << "Failed to load images." << std::endl;
        return -1;
    }

    // 读取点云数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

    if (!loadKITTIBinCloud(pointCloudFile, cloud))
    {
        return -1;
    }
    std::cout << "Loaded " << cloud->points.size() << " points from " << pointCloudFile << std::endl;

    // 创建一个新的点云用于存储带颜色的点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::copyPointCloud(*cloud, *colored_cloud);

    // 读取标定文件

    Eigen::Matrix<double, 3, 4> P0, P1, P2, P3;
    Eigen::Matrix<double, 4, 4> Tr;
    loadCalibrationFile(calib_file, P0, P1, P2, P3, Tr);
    // std::cout << "P0: " << P0 << std::endl;
    // std::cout << "P1: " << P1 << std::endl;
    // std::cout << "P2: " << P2 << std::endl;
    // std::cout << "P3: " << P3 << std::endl;
    // std::cout << "Tr: " << Tr << std::endl;

    // 将点云与图像结合，给点云上色
    colorPointCloudFromImage(colored_cloud, imageL, P2, Tr);
    colorPointCloudFromImage(colored_cloud, imageR, P3, Tr);

    // 可视化带颜色的点云
    pcl::visualization::PCLVisualizer viewer("Viewer");
    viewer.addPointCloud(colored_cloud, "colored_cloud");
    viewer.spin();

    // 保存为 .ply 文件
    // pcl::io::savePLYFile("colored_point_cloud.ply", *colored_cloud);

    return 0;
}
