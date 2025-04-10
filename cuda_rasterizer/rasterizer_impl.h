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

#include <cuda_runtime_api.h>

#include <iostream>
#include <vector>

#include "rasterizer.h"

namespace CudaRasterizer {
template <typename T>
static void obtain(char*& chunk,
                   T*& ptr,
                   std::size_t count,
                   std::size_t alignment) {
  std::size_t offset =
      (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) &
      ~(alignment - 1);
  ptr = reinterpret_cast<T*>(offset);
  chunk = reinterpret_cast<char*>(ptr + count);
}

struct GeometryState {
  size_t scan_size;         // 计算前缀和（扫描）时的temp_storage_bytes，即辅助空间大小
  float* depths;            // 对于所有的高斯，在图像坐标系下的深度
  char* scanning_space;     // 计算前缀和（扫描）时的d_temp_storage，即辅助空间的起始指针
  bool* clamped;            // 对于所有的高斯，预处理从 SH 算 RGB 的时候被裁剪到正值，keep track of this for the backward pass
  int* internal_radii;      // 对于所有的高斯，在图像坐标系下估计为圆的半径
  float2* means2D;          // 图像坐标系下所有高斯的均值
  float* cov3D;             // 对于所有高斯，世界坐标系下的协方差
  float4* conic_opacity;    // 对于所有高斯，图像坐标系下2D协方差矩阵的逆和不透明度
  float* rgb;               // 对于所有高斯，预处理从SH算RGB的结果
  uint32_t* point_offsets;  // 每个高斯触碰tiles个数的前缀和，也就是偏移量
  uint32_t* tiles_touched;  // 每个高斯触碰的tiles个数

  static GeometryState fromChunk(char*& chunk, size_t P);
};

struct ImageState {
  uint32_t* bucket_count;
  uint32_t* bucket_offsets;
  size_t bucket_count_scan_size;
  char* bucket_count_scanning_space;
  float* pixel_colors;
  uint32_t* max_contrib;

  size_t scan_size;
  uint2* ranges;
  uint32_t* n_contrib;
  float* accum_alpha;
  char* contrib_scan;

  static ImageState fromChunk(char*& chunk, size_t N);
};

struct BinningState {
  size_t scan_size;
  size_t sorting_size;
  uint64_t* point_list_keys_unsorted;
  uint64_t* point_list_keys;
  uint32_t* point_list_unsorted;
  uint32_t* point_list;
  int* scan_src;
  int* scan_dst;
  char* scan_space;
  char* list_sorting_space;

  static BinningState fromChunk(char*& chunk, size_t P);
};

struct SampleState {
  uint32_t* bucket_to_tile;
  float* T;
  float* ar;
  static SampleState fromChunk(char*& chunk, size_t C);
};

template <typename T>
size_t required(size_t P) {
  char* size = nullptr;
  T::fromChunk(size, P);
  return ((size_t)size) + 128;
}
};  // namespace CudaRasterizer
