#include <cuda_runtime.h>
#include <torch/torch.h>

void colorize_launcher(
    torch::Tensor& points,   // 按引用传递，确保原地修改
    torch::Tensor& colors,    // 图像数据（按引用传递）
    torch::Tensor& proj,     // 相机投影矩阵（按引用传递）
    torch::Tensor& Tr,       // 激光雷达到相机的变换矩阵（按引用传递）
    torch::Tensor& pose,     // Lidar pose（按引用传递）
    torch::Tensor& image    // 输出颜色（按引用传递）
);
