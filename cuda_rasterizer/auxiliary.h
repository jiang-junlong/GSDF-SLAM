/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once

#include <algorithm>
#include <cstdio>

#include "config.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE / 32)

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {1.0925484305920792f, -1.0925484305920792f,
                                  0.31539156525252005f, -1.0925484305920792f,
                                  0.5462742152960396f};
__device__ const float SH_C3[] = {-0.5900435899266435f, 2.890611442640554f,
                                  -0.4570457994644658f, 0.3731763325901154f,
                                  -0.4570457994644658f, 1.445305721320277f,
                                  -0.5900435899266435f};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
  return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p,
                                        int max_radius,
                                        uint2 &rect_min,
                                        uint2 &rect_max,
                                        dim3 grid)
{
  rect_min = {min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
              min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))};
  rect_max = {
      min(grid.x,
          max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
      min(grid.y,
          max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))};
}

__forceinline__ __device__ void getRect(const float2 p,
                                        int2 ext_rect,
                                        uint2 &rect_min,
                                        uint2 &rect_max,
                                        dim3 grid)
{
  rect_min = {min(grid.x, max((int)0, (int)((p.x - ext_rect.x) / BLOCK_X))),
              min(grid.y, max((int)0, (int)((p.y - ext_rect.y) / BLOCK_Y)))};
  rect_max = {
      min(grid.x,
          max((int)0, (int)((p.x + ext_rect.x + BLOCK_X - 1) / BLOCK_X))),
      min(grid.y,
          max((int)0, (int)((p.y + ext_rect.y + BLOCK_Y - 1) / BLOCK_Y)))};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3 &p,
                                                    const float *matrix)
{
  float3 transformed = {
      matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
      matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
      matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
  };
  return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3 &p,
                                                    const float *matrix)
{
  float4 transformed = {
      matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
      matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
      matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
      matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]};
  return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3 &p,
                                                  const float *matrix)
{
  float3 transformed = {
      matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
      matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
      matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
  };
  return transformed;
}

__forceinline__ __device__ float3
transformVec4x3Transpose(const float3 &p, const float *matrix)
{
  float3 transformed = {
      matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
      matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
      matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
  };
  return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
  float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
  float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
  float dnormvdz =
      (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) *
      invsum32;
  return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
  float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
  float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

  float3 dnormvdv;
  dnormvdv.x =
      ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) *
      invsum32;
  dnormvdv.y =
      (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) *
      invsum32;
  dnormvdv.z =
      (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) *
      invsum32;
  return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
  float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

  float4 vdv = {v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w};
  float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
  float4 dnormvdv;
  dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
  dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
  dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
  dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
  return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
  return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
                                           const float *orig_points,
                                           const float *viewmatrix, // 位姿
                                           const float *projmatrix,
                                           bool prefiltered,
                                           float3 &p_view)
{
  float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1],
                   orig_points[3 * idx + 2]};
  // if(idx== 0){
  //   printf("打印viewmatrix:\n");
  //   printf("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n", 
  //           viewmatrix[0], viewmatrix[4], viewmatrix[8],  viewmatrix[12],
  //           viewmatrix[1], viewmatrix[5], viewmatrix[9],  viewmatrix[13],
  //           viewmatrix[2], viewmatrix[6], viewmatrix[10], viewmatrix[14],
  //           viewmatrix[3], viewmatrix[7], viewmatrix[11], viewmatrix[15]);
  // }
  // Bring points to screen space
  p_view = transformPoint4x3(p_orig, viewmatrix);

  if (p_view.z <= 0.2f)
  { 
    // printf("p_view: %f %f %f\np_orig: %f %f %f\n", p_view.x, p_view.y, p_view.z, p_orig.x, p_orig.y, p_orig.z);
    if (prefiltered)
    {
      printf(
          "Point is filtered although prefiltered is set. This shouldn't "
          "happen!");
      __trap();
    }
    return false;
  }
  
  return true;
}

#define CHECK_CUDA(A, debug)                                                 \
  A;                                                                         \
  if (debug)                                                                 \
  {                                                                          \
    auto ret = cudaDeviceSynchronize();                                      \
    if (ret != cudaSuccess)                                                  \
    {                                                                        \
      std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ \
                << ": " << cudaGetErrorString(ret);                          \
      throw std::runtime_error(cudaGetErrorString(ret));                     \
    }                                                                        \
  }
