#include "cuda_colorize/colorize_pointcloud.h"

__forceinline__ __device__ float3 transformPoint(const float3 &p,
                                                    const float *matrix)
{
    float3 transformed = {
        matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3],
        matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7],
        matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11],
    };
    return transformed;
}

__global__ void colorize_kernel(
    float3 *points,              // 输入点云数据 (N, 3)
    float3 *colors,              // 输出颜色数据 (N, 3)
    bool *valid_flags,           // 点是否有效的标记 (N)
    int num_points,              // 点云数量
    const float *lidar_pose,     // LiDAR位姿 (行优先存储 4x4)          
    const float *proj_mat,       // 相机投影矩阵 (行优先存储 3x4)
    const float *Tr_velo_to_cam, // 激光到相机的变换矩阵 (行优先存储 4x4)
    const float3 *image,         // 图像数据 (H, W, 3)
    int img_width,               // 图像宽度
    int img_height               // 图像高度
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_points)
        return;

    float3 p = points[idx];                              // 读取点云位置
    float3 p_cam = transformPoint(p, Tr_velo_to_cam);    // LiDAR系 -> 相机坐标系
    float3 p_proj = transformPoint(p_cam, proj_mat);     // 投影到像素坐标
    if (p_proj.z <= 0.1f)
    {
        valid_flags[idx] = false;
        return;
    }

    // 投影到像素坐标
    int px = static_cast<int>(p_proj.x / p_proj.z);
    int py = static_cast<int>(p_proj.y / p_proj.z);

    if (px < 0 || px >= img_width || py < 0 || py >= img_height)
    {
        valid_flags[idx] = false;
        return;
    }

    float3 color = image[py * img_width + px];         // 获取图像颜色
    float3 p_world = transformPoint(p, lidar_pose);    // LiDAR系 -> 世界坐标系
    points[idx] = p_world;                             // 更新点云数据
    colors[idx] = color;                               // 更新颜色
    valid_flags[idx] = true;                           // 标记为有效
}

void colorize_launcher(
    torch::Tensor &points, // float32, (N, 3) - 原地修改
    torch::Tensor &colors, // uint8, (H, W, 3)
    torch::Tensor &proj,   // float32, (12,) 列优先 3x4
    torch::Tensor &Tr,     // float32, (16,) 列优先 4x4
    torch::Tensor &pose,   // float32, (16,) 列优先 4x4
    torch::Tensor &image   // uint8, (N, 3) - 原地修改
)
{
    const int N = points.size(0);
    const int H = image.size(0);
    const int W = image.size(1);

    // 确保输入张量在CUDA设备上
    if(points.is_cuda() == false) points = points.to(torch::kCUDA);
    if(colors.is_cuda() == false) colors = colors.to(torch::kCUDA);
    torch::Tensor proj_mat = proj.to(torch::kCUDA);
    torch::Tensor Tr_velo_to_cam = Tr.to(torch::kCUDA);
    torch::Tensor lidar_pose = pose.to(torch::kCUDA);
    

    // 创建valid_flags用于标记有效的点
    torch::Tensor valid_flags = torch::zeros({N}, torch::kBool).to(torch::kCUDA);


    // CUDA内核的线程配置
    const dim3 threads(256);
    const dim3 blocks((N + threads.x - 1) / threads.x);

    // 调用CUDA内核进行点云染色和标记有效点
    colorize_kernel<<<blocks, threads>>>(
        reinterpret_cast<float3 *>(points.data_ptr<float>()),
        reinterpret_cast<float3 *>(colors.data_ptr<float>()),
        valid_flags.data_ptr<bool>(),
        N,
        lidar_pose.data_ptr<float>(),
        proj_mat.data_ptr<float>(),
        Tr_velo_to_cam.data_ptr<float>(),
        reinterpret_cast<float3 *>(image.data_ptr<float>()),
        W,
        H);

    // 同步CUDA设备，确保内核执行完成
    cudaDeviceSynchronize();

    // 根据valid_flags筛选有效的点
    torch::Tensor valid_indices = valid_flags.nonzero().squeeze();
    points = points.index_select(0, valid_indices).contiguous();
    colors = colors.index_select(0, valid_indices).contiguous();
}
