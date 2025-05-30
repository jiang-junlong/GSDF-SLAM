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
 * All the performance optimizations are released under the MIT License with
 * {Mallick and Goel} and Kerbl, Bernhard and Vicente Carrasco, Francisco and
 * Steinberger, Markus and De La Torre, Fernando
 */

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>

#include <algorithm>
#include <cub/cub.cuh>

#include "auxiliary.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "forward.h"
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
// 该函数将每个高斯的球谐函数系数转换为简单的RGB颜色值
__device__ glm::vec3 computeColorFromSH(int idx,                // 当前高斯的索引
                                        int deg,                // 最大的球谐函数阶数
                                        int max_coeffs,         // 每个高斯的最大球谐函数系数数目
                                        const glm::vec3 *means, // 高斯的中心位置
                                        glm::vec3 campos,       // 相机的位置
                                        const float *dc,        // 直接颜色的系数（例如，来自光源的颜色或直接反射的光照）
                                        const float *shs,
                                        bool *clamped)
{
  // The implementation is loosely based on code for
  // "Differentiable Point-Based Radiance Fields for
  // Efficient View Synthesis" by Zhang et al. (2022)

  // 1. 获取当前高斯的中心位置，并计算从相机到高斯中心的方向向量
  glm::vec3 pos = means[idx];   // 获取当前高斯的中心位置
  glm::vec3 dir = pos - campos; // 计算从相机到高斯中心的方向向量
  dir = dir / glm::length(dir); // 归一化方向向量

  // 2. 获取当前高斯的直接颜色和球谐函数系数
  glm::vec3 *direct_color = ((glm::vec3 *)dc) + idx;     // 当前高斯的直接颜色值
  glm::vec3 *sh = ((glm::vec3 *)shs) + idx * max_coeffs; // 当前高斯的球谐函数系数

  // 3. 计算球谐函数的零阶项
  glm::vec3 result = SH_C0 * direct_color[0]; // 零阶球谐函数项与直接颜色的乘积

  // 4. 如果球谐函数的阶数大于0，加入一阶球谐函数项
  if (deg > 0)
  {
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    // 一阶球谐函数项
    result = result - SH_C1 * y * sh[0] + SH_C1 * z * sh[1] - SH_C1 * x * sh[2];

    // 5. 如果球谐函数的阶数大于1，加入二阶球谐函数项
    if (deg > 1)
    {
      float xx = x * x, yy = y * y, zz = z * z;
      float xy = x * y, yz = y * z, xz = x * z;
      // 二阶球谐函数项
      result = result + SH_C2[0] * xy * sh[3] + SH_C2[1] * yz * sh[4] +
               SH_C2[2] * (2.0f * zz - xx - yy) * sh[5] +
               SH_C2[3] * xz * sh[6] + SH_C2[4] * (xx - yy) * sh[7];

      // 6. 如果球谐函数的阶数大于2，加入三阶球谐函数项
      if (deg > 2)
      {
        result = result + SH_C3[0] * y * (3.0f * xx - yy) * sh[8] +
                 SH_C3[1] * xy * z * sh[9] +
                 SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[10] +
                 SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[11] +
                 SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[12] +
                 SH_C3[5] * z * (xx - yy) * sh[13] +
                 SH_C3[6] * x * (xx - 3.0f * yy) * sh[14];
      }
    }
  }
  // 7. 增加一个常数偏移量，防止颜色值出现负值
  result += 0.5f;

  // RGB colors are clamped to positive values. If values are
  // clamped, we need to keep track of this for the backward pass.
  // 8. 如果颜色值小于零，说明颜色值被钳制，需要记录这个状态
  clamped[3 * idx + 0] = (result.x < 0); // 记录是否发生了对红色通道的钳制
  clamped[3 * idx + 1] = (result.y < 0); // 记录是否发生了对绿色通道的钳制
  clamped[3 * idx + 2] = (result.z < 0); // 记录是否发生了对蓝色通道的钳制

  // 9. 将颜色值钳制到0和1之间，防止出现负值
  return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3 &mean,
                               float focal_x,
                               float focal_y,
                               float tan_fovx,
                               float tan_fovy,
                               const float *cov3D,
                               const float *viewmatrix)
{
  // The following models the steps outlined by equations 29
  // and 31 in "EWA Splatting" (Zwicker et al., 2002).
  // Additionally considers aspect / scaling of viewport.
  // Transposes used to account for row-/column-major conventions.
  // 将高斯中心点位置从世界坐标系转换到相机坐标系
  float3 t = transformPoint4x3(mean, viewmatrix);

  // 限制投影的x和y分量，避免超出视锥体范围
  const float limx = 1.3f * tan_fovx;
  const float limy = 1.3f * tan_fovy;
  const float txtz = t.x / t.z;
  const float tytz = t.y / t.z;
  t.x = min(limx, max(-limx, txtz)) * t.z;
  t.y = min(limy, max(-limy, tytz)) * t.z;

  glm::mat3 J =
      glm::mat3(focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
                0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
                0, 0, 0);

  glm::mat3 W = glm::mat3(viewmatrix[0], viewmatrix[4], viewmatrix[8],
                          viewmatrix[1], viewmatrix[5], viewmatrix[9],
                          viewmatrix[2], viewmatrix[6], viewmatrix[10]);

  glm::mat3 T = W * J;

  glm::mat3 Vrk = glm::mat3(cov3D[0], cov3D[1], cov3D[2],
                            cov3D[1], cov3D[3], cov3D[4],
                            cov3D[2], cov3D[4], cov3D[5]);

  glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

  // Apply low-pass filter: every Gaussian should be at least
  // one pixel wide/high. Discard 3rd row and column.
  cov[0][0] += 0.3f;
  cov[1][1] += 0.3f;
  return {float(cov[0][0]), float(cov[0][1]), float(cov[1][1])};
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale,
                             float mod,
                             const glm::vec4 rot,
                             float *cov3D)
{
  // Create scaling matrix
  // 创建缩放矩阵
  glm::mat3 S = glm::mat3(1.0f); // 创建一个3*3的单位矩阵
  S[0][0] = mod * scale.x;
  S[1][1] = mod * scale.y;
  S[2][2] = mod * scale.z;

  // Normalize quaternion to get valid rotation
  // 对四元数进行归一化，确保得到有效的旋转矩阵
  glm::vec4 q = rot; // / glm::length(rot);
  float r = q.x;
  float x = q.y;
  float y = q.z;
  float z = q.w;

  // Compute rotation matrix from quaternion
  glm::mat3 R = glm::mat3(1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
                          2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
                          2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

  glm::mat3 M = S * R;

  // Compute 3D world covariance matrix Sigma
  // 计算3D世界协方差矩阵Sigma
  glm::mat3 Sigma = glm::transpose(M) * M;

  // Covariance is symmetric, only store upper right
  // 由于协方差矩阵是对称的，只存储上三角部分
  cov3D[0] = Sigma[0][0];
  cov3D[1] = Sigma[0][1];
  cov3D[2] = Sigma[0][2];
  cov3D[3] = Sigma[1][1];
  cov3D[4] = Sigma[1][2];
  cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
// 在光栅化之前，对每个高斯进行初始化
template <int C>
__global__ void preprocessCUDA(int P, // 三维高斯的总数
                               int D, //
                               int M,
                               const float *orig_points,
                               const glm::vec3 *scales,
                               const float scale_modifier,
                               const glm::vec4 *rotations,
                               const float *opacities,
                               const float *dc,
                               const float *shs,
                               bool *clamped,
                               const float *cov3D_precomp,
                               const float *colors_precomp,
                               const float *viewmatrix,  // 相机位姿
                               const float *projmatrix,  // 投影矩阵
                               const glm::vec3 *cam_pos, // 相机位置
                               const int W,              // 图像宽度
                               int H,                    // 图像高度
                               const float tan_fovx,     // 水平视场的正切值
                               float tan_fovy,           // 垂直视场的正切值
                               const float focal_x,      // 水平焦距
                               float focal_y,            // 垂直焦距
                               int *radii,               // 半径
                               float2 *points_xy_image,  // 投影到图像平面的二维点
                               float *depths,            // 深度值
                               float *cov3Ds,            // 三维高斯的协方差
                               float *rgb,               // 颜色值
                               float4 *conic_opacity,
                               const dim3 grid, // 网格维度
                               uint32_t *tiles_touched,
                               bool prefiltered)
{
  auto idx = cg::this_grid().thread_rank(); // 获取每个线程的全局索引
  if (idx >= P)
    return;

  // Initialize radius and touched tiles to 0. If this isn't changed,
  // this Gaussian will not be processed further.
  radii[idx] = 0;
  tiles_touched[idx] = 0;

  // Perform near culling, quit if outside.
  // 判断高斯是否在视椎体内，如果不在就返回
  float3 p_view;
  if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered,
                  p_view))
    // printf("Gaussian %d is outside of frustum\n", idx);
    return;

  // Transform point by projecting
  float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]};
  float4 p_hom = transformPoint4x4(p_orig, projmatrix);                                       
  float p_w = 1.0f / (p_hom.w + 0.0000001f);
  float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};

  // If 3D covariance matrix is precomputed, use it, otherwise compute
  // from scaling and rotation parameters.
  const float *cov3D;
  if (cov3D_precomp != nullptr)
  {
    cov3D = cov3D_precomp + idx * 6;
  }
  else
  {
    computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
    cov3D = cov3Ds + idx * 6;
  }

  // Compute 2D screen-space covariance matrix
  // 根据每个三维高斯的协方差计算投影到图像空间的二维高斯的协方差
  float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

  // Invert covariance (EWA algorithm)
  // 计算二维高斯协方差矩阵的逆
  float det = (cov.x * cov.z - cov.y * cov.y); // 计算二维高斯协方差的行列式
  if (det == 0.0f)                             // 非满秩，直接返回
    return;
  float det_inv = 1.f / det;
  float3 conic = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv}; // 二维协方差矩阵的逆

  // Compute extent in screen space (by finding eigenvalues of
  // 2D covariance matrix). Use extent to compute a bounding rectangle
  // of screen-space tiles that this Gaussian overlaps with. Quit if
  // rectangle covers 0 tiles.
  float mid = 0.5f * (cov.x + cov.z);
  // 计算二维协方差矩阵的特征值
  float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
  float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
  float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));         // 椭圆的边界圆，3sigma原则
  float2 point_image = {ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)}; // 将归一化设备坐标 (NDC, Normalized Device Coordinates) 转换为像素坐标
  uint2 rect_min, rect_max;
  getRect(point_image, my_radius, rect_min, rect_max, grid); // 求tile grid和边界圆的交，rect_min，rect_max 代表在 tile_grid 中的坐标
  if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
    return;

  // If colors have been precomputed, use them, otherwise convert
  // spherical harmonics coefficients to RGB color.
  if (colors_precomp == nullptr)
  {
    glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3 *)orig_points,
                                          *cam_pos, dc, shs, clamped);
    rgb[idx * C + 0] = result.x;
    rgb[idx * C + 1] = result.y;
    rgb[idx * C + 2] = result.z;
  }

  // Store some useful helper data for the next steps.
  depths[idx] = p_view.z;
  radii[idx] = my_radius; // 存储二维高斯椭圆的最大半径
  points_xy_image[idx] = point_image;

  // Inverse 2D covariance and opacity neatly pack into one float4
  conic_opacity[idx] = {conic.x, conic.y, conic.z, opacities[idx]}; // 二维协方差的逆和透明度
  tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
    renderCUDA(const uint2 *__restrict__ ranges,
               const uint32_t *__restrict__ point_list,
               const uint32_t *__restrict__ per_tile_bucket_offset,
               uint32_t *__restrict__ bucket_to_tile,
               float *__restrict__ sampled_T,
               float *__restrict__ sampled_ar,
               int W,
               int H,
               const float2 *__restrict__ points_xy_image,
               const float *__restrict__ features,
               const float4 *__restrict__ conic_opacity,
               float *__restrict__ final_T,
               uint32_t *__restrict__ n_contrib,
               uint32_t *__restrict__ max_contrib,
               const float *__restrict__ bg_color,
               float *__restrict__ out_color)
{
  // Identify current tile and associated min/max pixel range.
  auto block = cg::this_thread_block();
  uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  uint2 pix_min = {block.group_index().x * BLOCK_X,
                   block.group_index().y * BLOCK_Y};
  uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
  uint2 pix = {pix_min.x + block.thread_index().x,
               pix_min.y + block.thread_index().y};
  uint32_t pix_id = W * pix.y + pix.x;
  float2 pixf = {(float)pix.x, (float)pix.y};

  // Check if this thread is associated with a valid pixel or outside.
  bool inside = pix.x < W && pix.y < H;
  // Done threads can help with fetching, but don't rasterize
  bool done = !inside;

  // Load start/end range of IDs to process in bit sorted list.
  uint32_t tile_id =
      block.group_index().y * horizontal_blocks + block.group_index().x;
  uint2 range = ranges[tile_id];
  const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
  int toDo = range.y - range.x;

  // what is the number of buckets before me? what is my offset?
  uint32_t bbm = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
  // let's first quickly also write the bucket-to-tile mapping
  int num_buckets = (toDo + 31) / 32;
  for (int i = 0; i < (num_buckets + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i)
  {
    int bucket_idx = i * BLOCK_SIZE + block.thread_rank();
    if (bucket_idx < num_buckets)
    {
      bucket_to_tile[bbm + bucket_idx] = tile_id;
    }
  }

  // Allocate storage for batches of collectively fetched data.
  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ float2 collected_xy[BLOCK_SIZE];
  __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

  // Initialize helper variables
  float T = 1.0f;
  uint32_t contributor = 0;
  uint32_t last_contributor = 0;
  float C[CHANNELS] = {0};

  // Iterate over batches until all done or range is complete
  for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
  {
    // End if entire block votes that it is done rasterizing
    int num_done = __syncthreads_count(done);
    if (num_done == BLOCK_SIZE)
      break;

    // Collectively fetch per-Gaussian data from global to shared
    int progress = i * BLOCK_SIZE + block.thread_rank();
    if (range.x + progress < range.y)
    {
      int coll_id = point_list[range.x + progress];
      collected_id[block.thread_rank()] = coll_id;
      collected_xy[block.thread_rank()] = points_xy_image[coll_id];
      collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
    }
    block.sync();

    // Iterate over current batch
    for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
    {
      // add incoming T value for every 32nd gaussian
      if (j % 32 == 0)
      {
        sampled_T[(bbm * BLOCK_SIZE) + block.thread_rank()] = T;
        for (int ch = 0; ch < CHANNELS; ++ch)
        {
          sampled_ar[(bbm * BLOCK_SIZE * CHANNELS) + ch * BLOCK_SIZE +
                     block.thread_rank()] = C[ch];
        }
        ++bbm;
      }

      // Keep track of current position in range
      contributor++;

      // Resample using conic matrix (cf. "Surface
      // Splatting" by Zwicker et al., 2001)
      float2 xy = collected_xy[j];
      float2 d = {xy.x - pixf.x, xy.y - pixf.y};
      float4 con_o = collected_conic_opacity[j];
      float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) -
                    con_o.y * d.x * d.y;
      if (power > 0.0f)
        continue;

      // Eq. (2) from 3D Gaussian splatting paper.
      // Obtain alpha by multiplying with Gaussian opacity
      // and its exponential falloff from mean.
      // Avoid numerical instabilities (see paper appendix).
      float alpha = min(0.99f, con_o.w * exp(power));
      if (alpha < 1.0f / 255.0f)
        continue;
      float test_T = T * (1 - alpha);
      if (test_T < 0.0001f)
      {
        done = true;
        continue;
      }

      // Eq. (3) from 3D Gaussian splatting paper.
      for (int ch = 0; ch < CHANNELS; ch++)
        C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

      T = test_T;

      // Keep track of last range entry to update this
      // pixel.
      last_contributor = contributor;
    }
  }

  // All threads that treat valid pixel write out their final
  // rendering data to the frame and auxiliary buffers.
  if (inside)
  {
    final_T[pix_id] = T;
    n_contrib[pix_id] = last_contributor;
    for (int ch = 0; ch < CHANNELS; ch++)
      out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
  }

  // max reduce the last contributor
  typedef cub::BlockReduce<uint32_t, BLOCK_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                           BLOCK_Y>
      BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  last_contributor =
      BlockReduce(temp_storage).Reduce(last_contributor, cub::Max());
  if (block.thread_rank() == 0)
  {
    max_contrib[tile_id] = last_contributor;
  }
}

void FORWARD::render(const dim3 grid,
                     dim3 block,
                     const uint2 *ranges,
                     const uint32_t *point_list,
                     const uint32_t *per_tile_bucket_offset,
                     uint32_t *bucket_to_tile,
                     float *sampled_T,
                     float *sampled_ar,
                     int W,
                     int H,
                     const float2 *means2D,
                     const float *colors,
                     const float4 *conic_opacity,
                     float *final_T,
                     uint32_t *n_contrib,
                     uint32_t *max_contrib,
                     const float *bg_color,
                     float *out_color)
{
  renderCUDA<NUM_CHAFFELS><<<grid, block>>>(
      ranges, point_list, per_tile_bucket_offset, bucket_to_tile, sampled_T,
      sampled_ar, W, H, means2D, colors, conic_opacity, final_T, n_contrib,
      max_contrib, bg_color, out_color);
}

void FORWARD::preprocess(int P, // 三维高斯的总数
                         int D,
                         int M,
                         const float *means3D,
                         const glm::vec3 *scales,
                         const float scale_modifier,
                         const glm::vec4 *rotations,
                         const float *opacities,
                         const float *dc,
                         const float *shs,
                         bool *clamped,
                         const float *cov3D_precomp,
                         const float *colors_precomp,
                         const float *viewmatrix,
                         const float *projmatrix,
                         const glm::vec3 *cam_pos,
                         const int W,
                         int H,
                         const float focal_x,
                         float focal_y,
                         const float tan_fovx,
                         float tan_fovy,
                         int *radii,
                         float2 *means2D,
                         float *depths,
                         float *cov3Ds,
                         float *rgb,
                         float4 *conic_opacity,
                         const dim3 grid,
                         uint32_t *tiles_touched,
                         bool prefiltered)
{
  // 封装preprocessCUDA核函数以处理三维高斯渲染的预处理步骤
  // 根据P（三维高斯的总数）和块大小256动态确定CUDA网格(grid)和块(block)数量
  // 调用CUDA核函数预处理三维高斯分布的每个参数, 为渲染做好准备
  preprocessCUDA<NUM_CHAFFELS><<<(P + 255) / 256, 256>>>(
      P, D, M, means3D, scales, scale_modifier, rotations, opacities, dc, shs,
      clamped, cov3D_precomp, colors_precomp, viewmatrix, projmatrix, cam_pos,
      W, H, tan_fovx, tan_fovy, focal_x, focal_y, radii, means2D, depths,
      cov3Ds, rgb, conic_opacity, grid, tiles_touched, prefiltered);
}
