#include "base_parser.h"

#include <dirent.h>
#include <filesystem>

#include "params/params.h"
#include "utils/ply_utils/ply_utils_torch.h"
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>

namespace dataparser {

std::vector<std::filesystem::path>
read_filelists(const std::filesystem::path &directory,
               const std::string &prefix, const std::string &extension) {
  std::vector<std::filesystem::path> image_files;
  for (const auto &entry : std::filesystem::directory_iterator(directory)) {
    if (entry.path().filename().string().rfind(prefix, 0) == 0 &&
        entry.path().extension().string() == extension) {
      image_files.emplace_back(entry.path().string());
    }
  }
  std::sort(image_files.begin(), image_files.end());
  return image_files;
}

bool computePairNum(std::string pair1, std::string pair2) {
  std::string num1 = pair1.substr(pair1.find_last_of('/'));
  num1 = num1.substr(num1.find_first_of("0123456789"));
  num1 = num1.substr(0, num1.find_last_of('.'));

  std::string num2 = pair2.substr(pair2.find_last_of('/'));
  num2 = num2.substr(num2.find_first_of("0123456789"));
  num2 = num2.substr(0, num2.find_last_of('.'));
  return std::stod(num1) < std::stod(num2);
}

void sort_filelists(std::vector<std::filesystem::path> &filists) {
  if (filists.empty())
    return;
  std::sort(filists.begin(), filists.end(), computePairNum);
}

void load_file_list(const std::string &dir_path,
                    std::vector<std::filesystem::path> &out_filelsits,
                    const std::string &prefix,
                    const std::string &file_extension) {
  out_filelsits = read_filelists(dir_path, prefix, file_extension);
  sort_filelists(out_filelsits);
  // for (auto &name : out_filelsits) {
  //   std::cout << name << "\n";
  // }

  std::cout << "Load " << out_filelsits.size() << " data." << "\n";
}

torch::Tensor DataParser::get_pose(const int &idx, const int &pose_type) const {
  /**
   * @description: return pose matrix at idx with shape [4, 4]
   * @return {*}
   */
  //  [1, 4, 4] -> [4, 4]
  return get_pose(torch::tensor({idx}, torch::kLong), pose_type).squeeze();
}

torch::Tensor DataParser::get_pose(const torch::Tensor &idx,
                                   const int &pose_type) const {
  /**
   * @description: return pose matrix at idx with shape [N, 4, 4]
   * @return {*}
   */
  switch (pose_type) {
  case DataType::RawColor: {
    return color_poses_.index_select(0, idx.to(color_poses_.device()));
  }
  case DataType::RawDepth: {
    return depth_poses_.index_select(0, idx.to(depth_poses_.device()));
  }
  case DataType::TrainColor: {
    return train_color_poses_.index_select(0,
                                           idx.to(train_color_poses_.device()));
  }
  case DataType::TrainDepth: {
    return train_depth_poses_.index_select(0,
                                           idx.to(train_depth_poses_.device()));
  }
  case DataType::EvalColor: {
    return eval_color_poses_.index_select(0,
                                          idx.to(eval_color_poses_.device()));
  }
  case DataType::EvalDepth: {
    return eval_depth_poses_.index_select(0,
                                          idx.to(eval_color_poses_.device()));
  }
  default:
    throw std::runtime_error("Invalid pose type");
  }
}

torch::Tensor DataParser::get_image(const int &idx,
                                    const int &image_type) const {
  /**
   return [H, W, 3]
   */
  return get_image(torch::tensor({idx}, torch::kLong), image_type).squeeze();
}

torch::Tensor DataParser::get_image(const torch::Tensor &idx,
                                    const int &image_type) const {
  /**
   return [N, H, W, 3]
   */
  switch (image_type) {
  case DataType::RawColor: {
    TORCH_CHECK(idx.numel() == 1);
    return get_color_image(idx.item<int>(), DataType::RawColor);
  }
  case DataType::RawDepth: {
    TORCH_CHECK(idx.numel() == 1);
    return get_color_image(idx.item<int>(), DataType::RawDepth);
  }
  case DataType::TrainColor: {
    if (preload_) {
      return train_color_.index_select(0, idx);
    } else {
      TORCH_CHECK(idx.numel() == 1);
      return get_color_image(idx.item<int>(), DataType::TrainColor);
    }
  }
  case DataType::TrainDepth: {
    if (preload_) {
      return train_depth_pack_.depth.index_select(0, idx);
    } else {
      TORCH_CHECK(idx.numel() == 1);
      return get_depth_image(idx.item<int>());
    }
  }
  case DataType::EvalColor: {
    return get_color_image(idx.item<int>(), DataType::EvalColor);
  }
  default:
    throw std::runtime_error("Invalid image type");
  }
}

std::filesystem::path DataParser::get_file(const int &idx,
                                           const int &image_type) {
  switch (image_type) {
  case DataType::RawColor:
    return raw_color_filelists_[idx];
  case DataType::RawDepth:
    return raw_depth_filelists_[idx];
  case DataType::TrainColor:
    return train_color_filelists_[idx];
  case DataType::TrainDepth:
    return train_depth_filelists_[idx];
  case DataType::EvalColor:
    return eval_color_filelists_[idx];
  case DataType::EvalDepth:
    return eval_depth_filelists_[idx];
  default:
    throw std::runtime_error("Unsupported color image type for get_image_file");
  }
}

cv::Mat DataParser::get_image_cv_mat(const int &idx,
                                     const int &image_type) const {
  switch (image_type) {
  case DataType::RawColor: {
    if (idx < raw_color_filelists_.size()) {
      return cv::imread(raw_color_filelists_[idx], cv::IMREAD_ANYCOLOR);
    }
  }
  case DataType::RawDepth: {
    if (depth_type_ == DepthType::Image) {
      if (idx < raw_depth_filelists_.size()) {
        return cv::imread(raw_depth_filelists_[idx], cv::IMREAD_ANYDEPTH);
      }
    } else {
      return {};
    }
  }
  case DataType::TrainColor: {
    if (idx < train_color_filelists_.size()) {
      return cv::imread(train_color_filelists_[idx], cv::IMREAD_ANYCOLOR);
    }
  }
  case DataType::TrainDepth: {
    if (depth_type_ == DepthType::Image) {
      if (idx < train_depth_filelists_.size()) {
        return cv::imread(train_depth_filelists_[idx], cv::IMREAD_ANYDEPTH);
      }
    } else {
      return {};
    }
  }
  case DataType::EvalColor:
    if (idx < eval_color_filelists_.size()) {
      return cv::imread(eval_color_filelists_[idx], cv::IMREAD_ANYCOLOR);
    }
  case DataType::EvalDepth:
    if (idx < eval_depth_filelists_.size()) {
      return cv::imread(eval_depth_filelists_[idx], cv::IMREAD_ANYDEPTH);
    }
  default:
    throw std::runtime_error(
        "Unsupported color image type for get_image_cv_matc");
  }
}

torch::Tensor DataParser::get_color_image(const int &idx, const int &image_type,
                                          const float &scale) const {
  if (idx < raw_color_filelists_.size()) {
    cv::Mat color_mat = get_image_cv_mat(idx, image_type);
    if (color_mat.empty()) {
      return {};
    }

    int resize_height = sensor_.camera.height * scale;
    int resize_width = sensor_.camera.width * scale;
    if (scale != 1.0) {
      cv::resize(color_mat, color_mat, cv::Size(resize_width, resize_height));
    }

    if (color_mat.type() == CV_8UC1) {
      color_mat.convertTo(color_mat, CV_32FC1, color_scale_inv_);
      if (!color_mat.isContinuous()) {
        color_mat = color_mat.clone();
      }
      return torch::from_blob(color_mat.data, {resize_height, resize_width, 1},
                              torch::kFloat32)
          .clone();
    } else if (color_mat.type() == CV_8UC3) {
      cv::cvtColor(color_mat, color_mat, cv::COLOR_BGR2RGB);
      color_mat.convertTo(color_mat, CV_32FC3, color_scale_inv_);
      if (!color_mat.isContinuous()) {
        color_mat = color_mat.clone();
      }
      return torch::from_blob(color_mat.data, {resize_height, resize_width, 3},
                              torch::kFloat32)
          .clone();
    } else {
      throw std::runtime_error("Unsupported color image type");
    }
  }
  return {};
}

torch::Tensor DataParser::get_depth_image(const int &idx) const {
  if (depth_type_ == DepthType::Image) {
    // image
    cv::Mat depth_mat = get_image_cv_mat(idx, 1);
    if (depth_mat.type() == CV_16UC1) {
      depth_mat.convertTo(depth_mat, CV_32FC1, depth_scale_inv_);
    }

    if (!depth_mat.isContinuous()) {
      depth_mat = depth_mat.clone();
    }
    return torch::from_blob(depth_mat.data, {depth_mat.rows, depth_mat.cols, 1},
                            torch::kFloat32)
        .clone();
  } else if (depth_type_ == DepthType::PLY) {
    // ply
    std::map<std::string, torch::Tensor> map_ply_tensor;
    ply_utils::read_ply_file_to_map_tensor(raw_depth_filelists_[idx],
                                           map_ply_tensor);
    return map_ply_tensor["xyz"];
  } else if (depth_type_ == DepthType::BIN) {
    // bin
    std::fstream input(raw_depth_filelists_[idx],
                       std::ios::in | std::ios::binary);
    if (!input.good()) {
      throw std::runtime_error("Could not open file: " +
                               raw_depth_filelists_[idx].string());
    }
    input.seekg(0, std::ios::beg);

    pcl::PointCloud<pcl::PointXYZ>::Ptr points(
        new pcl::PointCloud<pcl::PointXYZ>);

    while (input.good() && !input.eof()) {
      pcl::PointXYZ point;
      input.read((char *)&point.x, 4 * sizeof(float));
      // filter nan
      if (std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z)) {
        continue;
      }
      points->push_back(point);
    }
    input.close();

    return torch::from_blob(points->points.data(),
                            {(long)points->points.size(), 4}, torch::kFloat32)
        .slice(1, 0, 3)
        .clone();
  } else if (depth_type_ == DepthType::PCD) {
    // pcd
    pcl::PointCloud<pcl::PointXYZ>::Ptr points(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile(raw_depth_filelists_[idx], *points);

    return torch::from_blob(points->points.data(),
                            {(long)points->points.size(), 4}, torch::kFloat32)
        .slice(1, 0, 3)
        .clone();
  }

  return {};
}

/*
  load_poses: load poses from file
  pose_path: path to pose file
  with_head: whether the pose file has head
  pose_type: 0 for 4x4 matrix, 4 col a line
             1 for 4x4 matrix, 16 col a line
             2 for 3x4 matrix, 12 col a line
             3 for tum format: t x y z qx qy qz qw
 */
std::vector<torch::Tensor> DataParser::load_poses(const std::string &pose_path,
                                                  const bool &with_head,
                                                  const int &pose_type) {
  if (!std::filesystem::exists(pose_path)) {
    throw std::runtime_error("Pose file does not exist: " + pose_path);
  }
  std::ifstream file(pose_path);
  std::string line;
  std::vector<torch::Tensor> poses;
  std::vector<double> time_stamps;

  if (pose_type == 0) {
    // pose_type: 0 for 4x4 matrix, 4 col a line

    int lines_per_matrix = with_head ? 5 : 4;
    int skip_line = with_head ? 1 : 0;

    torch::Tensor pose_tensor = torch::zeros({4, 4}, torch::kFloat);
    int i = 0;
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      int j = 0;
      float value;
      while (iss >> value) {
        pose_tensor[i][j] = value;
        j++;
      }
      i++;
      if (i == 4) {
        poses.push_back(pose_tensor);
        i = 0;
        pose_tensor = torch::zeros({4, 4}, torch::kFloat); // need new tensor
      }
    }
  } else if (pose_type == 1) {
    // for 4x4 matrix, 16 col a line
    int lines_per_matrix = with_head ? 17 : 16;
    int skip_line = with_head ? 1 : 0;

    torch::Tensor pose_tensor = torch::zeros({4, 4}, torch::kFloat);
    int i = 0;
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      int j = 0;
      float value;
      while (iss >> value) {
        pose_tensor[i][j] = value;
        j++;
        if ((j % 4) == 0) {
          i++;
          j = 0;
        }
      }
      poses.push_back(pose_tensor);
      i = 0;
      pose_tensor = torch::zeros({4, 4}, torch::kFloat); // need new tensor
    }
  } else if (pose_type == 2) {
    // kitti format: for 3x4 matrix, 12 col a line

    torch::Tensor pose_tensor = torch::eye(4, torch::kFloat);
    int i = 0;
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      int j = 0;
      float value;
      while (iss >> value) {
        pose_tensor[i][j] = value;
        j++;
        if ((j % 4) == 0) {
          i++;
          j = 0;
        }
      }
      poses.push_back(pose_tensor);
      i = 0;
      pose_tensor = torch::eye(4, torch::kFloat); // need new tensor
    }
  } else if (pose_type == 3) {
    // TUM format: t x y z qx qy qz qw
    torch::Tensor pose_tensor = torch::eye(4, torch::kFloat);
    torch::Tensor quat_tensor = torch::zeros(4, torch::kFloat);
    int i = 0;
    while (std::getline(file, line)) {
      // read lines data into a double array
      std::istringstream iss(line);
      double value;
      while (iss >> value) {
        if (i == 0) {
          time_stamps.push_back(value);
        } else if (i < 4) {
          pose_tensor[i - 1][3] = value;
        } else {
          quat_tensor[i - 4] = value;
        }
        i++;
        if (i == 8) {
          auto rot = utils::quat_to_rot(quat_tensor, true);
          pose_tensor.index_put_(
              {torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)},
              rot);
          poses.push_back(pose_tensor);
          i = 0;
          pose_tensor = torch::eye(4, torch::kFloat); // need new tensor
        }
      }
      std::cout << "\n";
    }
  }
  file.close();

  // Concatenate the list of tensors into a single tensor
  if (time_stamps.empty()) {
    return {torch::stack(poses)};
  } else {
    return {torch::stack(poses), torch::tensor(time_stamps, torch::kDouble)};
  }
}

void DataParser::align_pose_sensor(
    std::vector<std::filesystem::path> &out_filelsits,
    torch::Tensor &sensor_poses, float max_time_diff_sensor_and_pose) {
  if (time_stamps_.numel() > 0) {
    std::vector<torch::Tensor> align_sensor_poses;
    std::vector<std::filesystem::path> align_sensor_filelists;

    for (auto &file_path : out_filelsits) {
      double time_stamp = std::stod(file_path.stem().string());
      for (int time_stamp_idx = 0; time_stamp_idx < time_stamps_.numel();
           time_stamp_idx++) {
        if (abs(time_stamp - time_stamps_[time_stamp_idx].item<double>()) <
            max_time_diff_sensor_and_pose) {
          align_sensor_filelists.push_back(file_path);
          // TODO: pose idx is aligned?: yes time align with sensor pose
          align_sensor_poses.push_back(sensor_poses[time_stamp_idx]);
          break;
        }
      }
    }

    out_filelsits = align_sensor_filelists;
    sensor_poses = torch::stack(align_sensor_poses);
  }
}

void DataParser::load_colors(const std::string &file_extension,
                             const std::string &prefix, const bool eval,
                             const bool &llff) {
  if (!eval) {
    assert(std::filesystem::exists(color_path_));
    load_file_list(color_path_, raw_color_filelists_, prefix, file_extension);
    align_pose_sensor(raw_color_filelists_, color_poses_,
                      k_max_time_diff_camera_and_pose);
    int raw_color_num = raw_color_filelists_.size();

    int train_color_num;
    if (llff) {
      int eval_num = raw_color_num / 8;
      train_color_num = raw_color_num - eval_num;
    } else {
      train_color_num = raw_color_num;
    }
    std::cout << train_color_num << std::endl;

    train_color_poses_ = torch::zeros({train_color_num, 3, 4});
    train_color_filelists_.resize(train_color_num);
#pragma omp parallel for
    for (int i = 1; i <= raw_color_num; i++) {
      auto pose = get_pose(i - 1, DataType::RawColor).slice(0, 0, 3);
      if (llff) {
        if (i % 8 != 0) {
          train_color_poses_.index_put_({i - i / 8 - 1}, pose);
          train_color_filelists_[i - i / 8 - 1] = raw_color_filelists_[i - 1];
        }
      } else {
        train_color_poses_.index_put_({i - 1}, pose);
        train_color_filelists_[i - 1] = raw_color_filelists_[i - 1];
      }
    }

    if (preload_) {
      train_color_ = torch::zeros(
          {train_color_num, (int)(sensor_.camera.height * res_scale_),
           (int)(sensor_.camera.width * res_scale_), 3});
#pragma omp parallel for
      for (int i = 1; i <= raw_color_filelists_.size(); i++) {
        auto color = get_color_image(i - 1, DataType::RawColor, res_scale_);
        auto pose = get_pose(i - 1, DataType::RawColor).slice(0, 0, 3);
        if (llff) {
          if (i % 8 != 0) {
            train_color_.index_put_({i - i / 8 - 1}, color);
          }
        } else {
          train_color_.index_put_({i - 1}, color);
        }
      }
      std::cout << train_color_.sizes() << std::endl;
    }
  } else {
    assert(std::filesystem::exists(eval_color_path_));
    load_file_list(eval_color_path_, eval_color_filelists_, prefix,
                   file_extension);
  }
}

void DataParser::load_depths(const std::string &file_extension,
                             const std::string &prefix, const bool eval,
                             const bool &llff) {
  if (!eval) {
    assert(std::filesystem::exists(depth_path_));
    load_file_list(depth_path_, raw_depth_filelists_, prefix, file_extension);
    align_pose_sensor(raw_depth_filelists_, depth_poses_,
                      k_max_time_diff_lidar_and_pose);
    int raw_depth_num = raw_depth_filelists_.size();

    int train_depth_num;
    if (llff) {
      int eval_num = raw_depth_num / 8;
      train_depth_num = raw_depth_num - eval_num;
    } else {
      train_depth_num = raw_depth_num;
    }

    if (1) {
      auto tmp_points_ndir_dirn = get_distance_ndir_zdirn(0);
      auto tmp_depth = tmp_points_ndir_dirn[0].view({-1, 1});
      int per_frame_point_num = tmp_depth.size(0);
      bool is_need_ds = k_ds_pt_num < per_frame_point_num;
      per_frame_point_num = is_need_ds ? k_ds_pt_num : per_frame_point_num;

      train_depth_pack_.depth =
          torch::zeros({train_depth_num, per_frame_point_num, 1});
      train_depth_pack_.direction =
          torch::zeros({train_depth_num, per_frame_point_num, 3});
      train_depth_pack_.xyz =
          torch::zeros({train_depth_num, per_frame_point_num, 3});
      train_depth_pack_.origin = torch::zeros({train_depth_num, 3});

      train_depth_poses_ = torch::zeros({train_depth_num, 3, 4});

      train_depth_filelists_.resize(train_depth_num);

#pragma omp parallel for
      for (int i = 1; i <= raw_depth_num; i++) {
        auto points_ndir_dirn = get_distance_ndir_zdirn(i - 1);

        auto depth = points_ndir_dirn[0].view({-1, 1});
        auto direction = points_ndir_dirn[1].view({-1, 3});
        auto pose = get_pose(i - 1, DataType::RawDepth).slice(0, 0, 3);
        auto pos = pose.slice(1, 3, 4).squeeze();
        auto rot = pose.slice(1, 0, 3);
        direction = direction.mm(rot.t());

        auto valid_mask = depth.squeeze() > k_min_range;
        auto valid_idx = torch::nonzero(valid_mask).view({-1});
        depth = depth.index_select(0, valid_idx);
        direction = direction.index_select(0, valid_idx);

        if (is_need_ds) {
          auto sample_idx =
              torch::randint(0, depth.size(0), {k_ds_pt_num}, torch::kLong);
          depth = depth.index_select(0, sample_idx);
          direction = direction.index_select(0, sample_idx);
        }

        auto xyz = direction * depth + pos;
        if (llff) {
          if (i % 8 != 0) {
            train_depth_pack_.depth.index_put_({i - i / 8 - 1}, depth);
            train_depth_pack_.direction.index_put_({i - i / 8 - 1}, direction);
            train_depth_pack_.xyz.index_put_({i - i / 8 - 1}, xyz);

            train_depth_pack_.origin.index_put_({i - i / 8 - 1},
                                                pos.view({1, 3}));

            train_depth_poses_.index_put_({i - i / 8 - 1}, pose);
            train_depth_filelists_[i - i / 8 - 1] = raw_depth_filelists_[i - 1];
          }
        } else {
          train_depth_pack_.depth.index_put_({i - 1}, depth);
          train_depth_pack_.direction.index_put_({i - 1}, direction);
          train_depth_pack_.xyz.index_put_({i - 1}, xyz);
          train_depth_pack_.origin.index_put_({i - 1}, pos.view({1, 3}));

          train_depth_poses_.index_put_({i - i / 8 - 1}, pose);
          train_depth_filelists_[i - 1] = raw_depth_filelists_[i - 1];
        }
      }
      std::cout << train_depth_pack_.depth.sizes() << '\n';
    }
  } else {
    assert(std::filesystem::exists(eval_depth_path_));
    load_file_list(eval_depth_path_, eval_depth_filelists_, prefix,
                   file_extension);
  }
}

std::vector<at::Tensor> DataParser::get_depth_zdir(const int &idx) {
  /**
   * @description:
   * @return {depth_image, zdir}, where zdir.z = 1;
             {[height width 1], [height width 3]}
   */
  // [height width 1]
  auto depth_image = get_depth_image(idx);

  // [height width 3]
  auto zdir = sensor::get_image_coords_zdir(
      sensor_.camera.height, sensor_.camera.width, sensor_.camera.fx,
      sensor_.camera.fy, sensor_.camera.cx, sensor_.camera.cy,
      depth_image.device());
  return {depth_image, zdir};
}
std::vector<at::Tensor> DataParser::get_distance_ndir_zdirn(const int &idx) {
  /**
   * @description:
   * @return {distance, ndir, dir_norm}, where ndir.norm = 1;
             {[height width 1], [height width 3], [height width 1]}
   */
  auto depth_zdir_vec = get_depth_zdir(idx);
  auto depth_image = depth_zdir_vec[0];
  auto zdir = depth_zdir_vec[1];
  // [height width 1]
  auto zdir_norm = zdir.norm(2, -1, true);
  auto ndir = zdir / zdir_norm;
  auto distance = depth_image * zdir_norm;
  return {distance, ndir, zdir_norm};
}

at::Tensor DataParser::get_points(const int &idx) {
  auto depth_zdir_vec = get_depth_zdir(idx);
  auto depth_image = depth_zdir_vec[0];
  auto zdir = depth_zdir_vec[1];
  // [height width 3]
  auto points = depth_image * zdir;
  return points;
}

std::vector<at::Tensor> DataParser::get_points_dist_ndir_zdirn(const int &idx) {
  auto distance_ndir_dirn = get_distance_ndir_zdirn(idx);
  auto distance = distance_ndir_dirn[0];
  auto ndir = distance_ndir_dirn[1];
  // [height width 3]
  auto points = distance * ndir;
  return {points, distance, ndir, distance_ndir_dirn[2]};
}
} // namespace dataparser