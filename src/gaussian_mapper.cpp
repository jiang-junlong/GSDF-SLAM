#include "include/gaussian_mapper.h"

GaussianMapper::GaussianMapper(
    std::filesystem::path dataset_path,
    std::filesystem::path gaussian_config_file_path,
    std::filesystem::path result_dir,
    int seed,
    torch::DeviceType device_type)
    : initial_mapped_(false),
      interrupt_training_(false),
      stopped_(false),
      iteration_(0),
      ema_loss_for_log_(0.0f),
      SLAM_ended_(false),
      loop_closure_iteration_(false),
      min_num_initial_map_kfs_(15UL),
      large_rot_th_(1e-1f),
      large_trans_th_(1e-2f),
      training_report_interval_(0)
{
    // Random seed
    std::srand(seed);
    torch::manual_seed(seed);

    // Device
    if (device_type == torch::kCUDA && torch::cuda::is_available())
    {
        std::cout << "[Gaussian Mapper]CUDA available! Training on GPU." << std::endl;
        device_type_ = torch::kCUDA;
        model_params_.data_device_ = "cuda";
    }
    else
    {
        std::cout << "[Gaussian Mapper]Training on CPU." << std::endl;
        device_type_ = torch::kCPU;
        model_params_.data_device_ = "cpu";
    }

    result_dir_ = result_dir;
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    config_file_path_ = gaussian_config_file_path;
    readConfigFromFile(gaussian_config_file_path);

    std::vector<float> bg_color;
    if (model_params_.white_background_)
        bg_color = {1.0f, 1.0f, 1.0f};
    else
        bg_color = {0.0f, 0.0f, 0.0f};
    background_ = torch::tensor(bg_color, torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    override_color_ = torch::empty(0, torch::TensorOptions().device(device_type_));

    // 初始化场景和模型
    gaussians_ = std::make_shared<GaussianModel>(model_params_);
    scene_ = std::make_shared<GaussianScene>(model_params_);
    dataloader_ptr_ = std::make_unique<dataloader::DataLoader>(dataset_path, 5, device_type_);

    /*******************************************************************************************/
    new (&ort_env_) Ort::Env(ORT_LOGGING_LEVEL_WARNING, "skyseg");
    session_options_ = Ort::SessionOptions();
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // ort_memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    ort_memory_info_ptr_ = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

    skyseg_session_ = std::make_unique<Ort::Session>(ort_env_, "/home/jiang/myproject/GSDF-SLAM/skyseg.onnx", session_options_);
    /*******************************************************************************************/
}

void GaussianMapper::readConfigFromFile(std::filesystem::path cfg_path)
{
    cv::FileStorage settings_file(cfg_path.string().c_str(), cv::FileStorage::READ);
    if (!settings_file.isOpened())
    {
        std::cerr << "[Gaussian Mapper]Failed to open settings file at: " << cfg_path << std::endl;
        exit(-1);
    }

    std::cout << "[Gaussian Mapper]Reading parameters from " << cfg_path << std::endl;
    std::unique_lock<std::mutex> lock(mutex_settings_);

    // Model parameters
    model_params_.sh_degree_ = settings_file["Model.sh_degree"].operator int();
    model_params_.resolution_ = settings_file["Model.resolution"].operator float();
    model_params_.white_background_ = (settings_file["Model.white_background"].operator int()) != 0;
    model_params_.eval_ = (settings_file["Model.eval"].operator int()) != 0;

    // Pipeline Parameters
    z_near_ = settings_file["Camera.z_near"].operator float();
    z_far_ = settings_file["Camera.z_far"].operator float();

    inactive_geo_densify_ = (settings_file["Mapper.inactive_geo_densify"].operator int()) != 0;
    max_depth_cached_ = settings_file["Mapper.depth_cache"].operator int();
    min_num_initial_map_kfs_ = static_cast<unsigned long>(settings_file["Mapper.min_num_initial_map_kfs"].operator int());
    new_keyframe_times_of_use_ = settings_file["Mapper.new_keyframe_times_of_use"].operator int();
    large_rot_th_ = settings_file["Mapper.large_rotation_threshold"].operator float();
    large_trans_th_ = settings_file["Mapper.large_translation_threshold"].operator float();
    stable_num_iter_existence_ = settings_file["Mapper.stable_num_iter_existence"].operator int();

    pipe_params_.convert_SHs_ = (settings_file["Pipeline.convert_SHs"].operator int()) != 0;
    pipe_params_.compute_cov3D_ = (settings_file["Pipeline.compute_cov3D"].operator int()) != 0;

    do_gaus_pyramid_training_ = (settings_file["GausPyramid.do"].operator int()) != 0;
    num_gaus_pyramid_sub_levels_ = settings_file["GausPyramid.num_sub_levels"].operator int();
    int sub_level_times_of_use = settings_file["GausPyramid.sub_level_times_of_use"].operator int();
    kf_gaus_pyramid_times_of_use_.resize(num_gaus_pyramid_sub_levels_);
    kf_gaus_pyramid_factors_.resize(num_gaus_pyramid_sub_levels_);
    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l)
    {
        kf_gaus_pyramid_times_of_use_[l] = sub_level_times_of_use;
        kf_gaus_pyramid_factors_[l] = std::pow(0.5f, num_gaus_pyramid_sub_levels_ - l);
    }

    keyframe_record_interval_ = settings_file["Record.keyframe_record_interval"].operator int();
    all_keyframes_record_interval_ = settings_file["Record.all_keyframes_record_interval"].operator int();
    record_rendered_image_ = (settings_file["Record.record_rendered_image"].operator int()) != 0;
    record_ground_truth_image_ = (settings_file["Record.record_ground_truth_image"].operator int()) != 0;
    record_loss_image_ = (settings_file["Record.record_loss_image"].operator int()) != 0;
    training_report_interval_ = settings_file["Record.training_report_interval"].operator int();
    record_loop_ply_ = (settings_file["Record.record_loop_ply"].operator int()) != 0;

    // Optimization Parameters
    opt_params_.iterations_ = settings_file["Optimization.max_num_iterations"].operator int();
    opt_params_.position_lr_init_ = settings_file["Optimization.position_lr_init"].operator float();
    opt_params_.position_lr_final_ = settings_file["Optimization.position_lr_final"].operator float();
    opt_params_.position_lr_delay_mult_ = settings_file["Optimization.position_lr_delay_mult"].operator float();
    opt_params_.position_lr_max_steps_ = settings_file["Optimization.position_lr_max_steps"].operator int();
    opt_params_.feature_lr_ = settings_file["Optimization.feature_lr"].operator float();
    opt_params_.opacity_lr_ = settings_file["Optimization.opacity_lr"].operator float();
    opt_params_.scaling_lr_ = settings_file["Optimization.scaling_lr"].operator float();
    opt_params_.rotation_lr_ = settings_file["Optimization.rotation_lr"].operator float();

    opt_params_.percent_dense_ = settings_file["Optimization.percent_dense"].operator float();
    opt_params_.lambda_dssim_ = settings_file["Optimization.lambda_dssim"].operator float();
    opt_params_.densification_interval_ = settings_file["Optimization.densification_interval"].operator int();
    opt_params_.opacity_reset_interval_ = settings_file["Optimization.opacity_reset_interval"].operator int();
    opt_params_.densify_from_iter_ = settings_file["Optimization.densify_from_iter_"].operator int();
    opt_params_.densify_until_iter_ = settings_file["Optimization.densify_until_iter"].operator int();
    opt_params_.densify_grad_threshold_ = settings_file["Optimization.densify_grad_threshold"].operator float();

    prune_big_point_after_iter_ = settings_file["Optimization.prune_big_point_after_iter"].operator int();
    densify_min_opacity_ = settings_file["Optimization.densify_min_opacity"].operator float();

    // Viewer Parameters
    rendered_image_viewer_scale_ = settings_file["GaussianViewer.image_scale"].operator float();
    rendered_image_viewer_scale_main_ = settings_file["GaussianViewer.image_scale_main"].operator float();
}
void printTensorMemory(const at::Tensor &t, const std::string &name)
{
    auto numel = t.numel();               // 元素数量
    auto element_size = t.element_size(); // 每个元素字节数
    auto total_bytes = numel * element_size;

    std::cout << "Tensor [" << name << "] uses "
              << total_bytes / 1024.0 << " KB (" << total_bytes << " bytes), "
              << "dtype = " << t.dtype()
              << ", shape = " << t.sizes()
              << ", device = " << t.device()
              << std::endl;
}
void GaussianMapper::run()
{
    size_t n = dataloader_ptr_->dataparser_ptr_->raw_depth_filelists_.size();
    for (int i = 0; i < n; ++i)
    {
        std::cout << "Processing idx:" << i << ", " << (i + 1) << "/" << n << "\r";
        std::cout.flush();
        // printTensorMemory(gaussians_->features_dc_, "Gaussian");
        cv::Mat image;
        torch::Tensor cam_pose, point_cloud, color;
        dataloader_ptr_->get_item(i, cam_pose, point_cloud, color, image);

        // camera
        class Camera camera;
        camera.camera_id_ = i;
        camera.setModelId(Camera::CameraModelType::PINHOLE);
        camera.width_ = image.cols;
        camera.height_ = image.rows;
        camera.num_gaus_pyramid_sub_levels_ = num_gaus_pyramid_sub_levels_;
        camera.gaus_pyramid_width_.resize(num_gaus_pyramid_sub_levels_);
        camera.gaus_pyramid_height_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l)
        {
            camera.gaus_pyramid_width_[l] = camera.width_ * kf_gaus_pyramid_factors_[l];
            camera.gaus_pyramid_height_[l] = camera.height_ * kf_gaus_pyramid_factors_[l];
        }
        camera.params_[0] = dataloader_ptr_->dataparser_ptr_->P(0, 0);
        camera.params_[1] = dataloader_ptr_->dataparser_ptr_->P(1, 1);
        camera.params_[2] = dataloader_ptr_->dataparser_ptr_->P(0, 2);
        camera.params_[3] = dataloader_ptr_->dataparser_ptr_->P(1, 2);

        cv::Mat K = (cv::Mat_<float>(3, 3) << camera.params_[0], 0.f, camera.params_[2],
                     0.f, camera.params_[1], camera.params_[3],
                     0.f, 0.f, 1.f);
        camera.initUndistortRectifyMapAndMask(K, cv::Size(camera.width_, camera.height_), K, true);

        cv::Mat viewer_main_undistort_mask;
        int viewer_image_height_main_ = camera.height_ * this->rendered_image_viewer_scale_main_;
        int viewer_image_width_main_ = camera.width_ * this->rendered_image_viewer_scale_main_;
        cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                   cv::Size(viewer_image_width_main_, viewer_image_height_main_));
        this->viewer_main_undistort_mask_[camera.camera_id_] =
            tensor_utils::cvMat2TorchTensor_Float32(viewer_main_undistort_mask, this->device_type_);

        scene_->addCamera(camera);
        // Create a new keyframe
        std::shared_ptr<GaussianKeyframe> new_kf = std::make_shared<GaussianKeyframe>(i, getIteration());
        new_kf->zfar_ = z_far_;
        new_kf->znear_ = z_near_;
        new_kf->setCameraParams(camera);
        torch::Tensor cam_inv = cam_pose.inverse();
        new_kf->setPose(cam_inv);
        new_kf->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(image, device_type_);
        new_kf->img_filename_ = std::to_string(i);
        new_kf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
        new_kf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
        new_kf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
        new_kf->computeTransformTensors(); // 将刚才的位姿变成tensor
        scene_->addKeyframe(new_kf, &kfid_shuffled_);
        // new_kf->img_undist_ = image.clone();

        cv::Mat img_rgb;
        cv::resize(image, img_rgb, cv::Size(320, 320)); // 调整尺寸
        std::vector<cv::Mat> channels(3);
        cv::split(img_rgb, channels);

        channels[0] = (channels[0] - 0.485f) / 0.229f; // R
        channels[1] = (channels[1] - 0.456f) / 0.224f; // G
        channels[2] = (channels[2] - 0.406f) / 0.225f; // B

        cv::merge(channels, img_rgb);

        // 转成 tensor，HWC -> CHW
        at::Tensor img_tensor = torch::from_blob(img_rgb.data, {1, 320, 320, 3}, torch::kFloat).permute({0, 3, 1, 2}).contiguous();

        // 构造输入 tensor
        std::array<int64_t, 4> input_shape{1, 3, 320, 320};
        Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
            *ort_memory_info_ptr_,
            img_tensor.data_ptr<float>(), img_tensor.numel(),
            input_shape.data(), input_shape.size());

        // 获取输入输出名
        const char *input_name = skyseg_session_->GetInputName(0, Ort::AllocatorWithDefaultOptions());
        const char *output_name = skyseg_session_->GetOutputName(0, Ort::AllocatorWithDefaultOptions());

        // 推理
        auto output_tensors = skyseg_session_->Run(Ort::RunOptions{nullptr},
                                                   &input_name, &input_tensor_ort, 1,
                                                   &output_name, 1);

        // 输出处理
        float *output_data = output_tensors[0].GetTensorMutableData<float>();
        cv::Mat raw_output(320, 320, CV_32FC1, output_data);
        cv::Mat norm_output;

        double min_val, max_val;
        cv::minMaxLoc(raw_output, &min_val, &max_val);
        norm_output = (raw_output - min_val) / (max_val - min_val);

        // 放缩到 0~255 并转 uint8
        norm_output *= 255.0;
        norm_output.convertTo(norm_output, CV_8UC1);
        cv::Mat mask;
        cv::resize(norm_output, mask, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_NEAREST);

        mask = mask > 32;

        // 3. 将 image 中 mask 区域置0
        cv::Mat masked_image;
        image.copyTo(masked_image);
        masked_image.setTo(cv::Scalar(0, 0, 0), mask);
        new_kf->img_undist_ = masked_image.clone();
        this->undistort_mask_[camera.camera_id_] =
            tensor_utils::cvMat2TorchTensor_Float32(
                masked_image, this->device_type_);
        this->viewer_camera_id_ = camera.camera_id_;
        if (!this->viewer_camera_id_set_)
        {
            this->viewer_camera_id_set_ = true;
        }

        // Prepare multi resolution images for training
        increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());
        if (device_type_ == torch::kCUDA)
        {
            cv::cuda::GpuMat img_gpu;
            img_gpu.upload(new_kf->img_undist_);                                       // 上传图片到GPU
            new_kf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_); // 设置图像金字塔容器大小
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l)
            {
                cv::cuda::GpuMat img_resized;
                cv::cuda::resize(img_gpu, img_resized, cv::Size(new_kf->gaus_pyramid_width_[l], new_kf->gaus_pyramid_height_[l]));
                new_kf->gaus_pyramid_original_image_[l] =
                    tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
            }
        }
        else
        {
            new_kf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l)
            {
                cv::Mat img_resized;
                cv::resize(new_kf->img_undist_, img_resized, cv::Size(new_kf->gaus_pyramid_width_[l], new_kf->gaus_pyramid_height_[l]));
                new_kf->gaus_pyramid_original_image_[l] = tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
            }
        }

        // 准备训练
        {
            std::unique_lock<std::mutex> lock_render(mutex_render_);
            if (i == 0)
            {
                scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
                gaussians_->createFromPcd(point_cloud, color, scene_->cameras_extent_); // 从缓存点云中创建gaussians
                std::unique_lock<std::mutex> lock(mutex_settings_);
                gaussians_->trainingSetup(opt_params_);
                this->initial_mapped_ = true;
            }
            else
            {

                gaussians_->increasePcd(point_cloud, color, i, scene_->cameras_extent_); // 从缓存点云中创建gaussians
                this->initial_mapped_ = true;
            }
        }

        // 启动训练一次
        for (int i = 0; i < 1; ++i)
        {
            trainForOneIteration();
        }
    }
}

void GaussianMapper::trainForOneIteration()
{
    increaseIteration(1); // 迭代轮数计数+1
    auto iter_start_timing = std::chrono::steady_clock::now();

    // Pick a random Camera
    std::shared_ptr<GaussianKeyframe> viewpoint_cam = useOneRandomSlidingWindowKeyframe();
    // std::shared_ptr<GaussianKeyframe> viewpoint_cam = scene_->keyframes().rbegin()->second;
    // std::cout << "scene keyframes size: " << scene_->keyframes().size() << std::endl;
    if (!viewpoint_cam)
    {
        increaseIteration(-1);
        return;
    }

    writeKeyframeUsedTimes(result_dir_ / "used_times");
    int training_level = num_gaus_pyramid_sub_levels_;
    int image_height, image_width;
    torch::Tensor gt_image, mask;
    if (isdoingGausPyramidTraining())
        training_level = viewpoint_cam->getCurrentGausPyramidLevel();
    if (training_level == num_gaus_pyramid_sub_levels_)
    {
        image_height = viewpoint_cam->image_height_;
        image_width = viewpoint_cam->image_width_;
        gt_image = viewpoint_cam->original_image_.cuda();
        mask = undistort_mask_[viewpoint_cam->camera_id_];
    }
    else
    {
        image_height = viewpoint_cam->gaus_pyramid_height_[training_level];
        image_width = viewpoint_cam->gaus_pyramid_width_[training_level];
        gt_image = viewpoint_cam->gaus_pyramid_original_image_[training_level].cuda();
        mask = scene_->cameras_.at(viewpoint_cam->camera_id_).gaus_pyramid_undistort_mask_[training_level];
    }

    // Mutex lock for usage of the gaussian model
    std::unique_lock<std::mutex> lock_render(mutex_render_);

    // Every 1000 its we increase the levels of SH up to a maximum degree
    gaussians_->setShDegree(3);

    // Update learning rate 更新学习率
    gaussians_->updateLearningRate(getIteration());
    gaussians_->setFeatureLearningRate(featureLearningRate());
    gaussians_->setOpacityLearningRate(opacityLearningRate());
    gaussians_->setScalingLearningRate(scalingLearningRate());
    gaussians_->setRotationLearningRate(rotationLearningRate());

    // Render 渲染
    auto render_pkg = GaussianRenderer::render(
        viewpoint_cam,
        image_height,
        image_width,
        gaussians_,
        pipe_params_,
        background_,
        override_color_);
    auto rendered_image = std::get<0>(render_pkg);

    // 用于调试
    // cv::Mat rendered_image_mat = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
    // cv::imshow("rendered_image", rendered_image_mat);
    // cv::Mat gt_image_mat = tensor_utils::torchTensor2CvMat_Float32(gt_image);
    // cv::imshow("gt_image", gt_image_mat);
    // cv::waitKey(0);

    // for (int i = 0; i < rendered_image_mat.rows; i++)
    // {
    //     for (int j = 0; j < rendered_image_mat.cols; j++)
    //     {
    //         std::cout << rendered_image_mat.at<float>(i, j) << " ";
    //     }
    // }

    auto viewspace_point_tensor = std::get<1>(render_pkg);
    auto visibility_filter = std::get<2>(render_pkg);
    auto radii = std::get<3>(render_pkg);

    // Get rid of black edges caused by undistortion
    torch::Tensor white = torch::ones_like(rendered_image); // 全白张量（值=1.0）
    torch::Tensor masked_image = torch::where(mask > 0, rendered_image, white);
    torch::Tensor masked_gt_image = torch::where(mask > 0, gt_image, white);
    // cv::Mat rendered_image_mat = tensor_utils::torchTensor2CvMat_Float32(masked_image);
    // cv::imshow("rendered_image", rendered_image_mat);
    // cv::Mat gt_image_mat = tensor_utils::torchTensor2CvMat_Float32(masked_gt_image);
    // cv::imshow("gt_image", gt_image_mat);
    // cv::Mat mask_mat = tensor_utils::torchTensor2CvMat_Float32(mask);
    // cv::imshow("mask", mask_mat);
    // cv::waitKey(0);

    auto Ll1 = loss_utils::l1_loss(masked_image, masked_gt_image);
    // std::cout << "loss: " << Ll1.item().toFloat() << std::endl;
    float lambda_dssim = lambdaDssim();
    auto loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - loss_utils::ssim(masked_image, gt_image, device_type_));

    loss.backward(); // 反向传播

    torch::cuda::synchronize(); // 等待GPU完成所有任务

    {
        torch::NoGradGuard no_grad;
        ema_loss_for_log_ = 0.4f * loss.item().toFloat() + 0.6 * ema_loss_for_log_;

        // 致密化
        if (getIteration() < opt_params_.densify_until_iter_)
        {
            // Keep track of max radii in image-space for pruning
            gaussians_->max_radii2D_.index_put_(
                {visibility_filter},
                torch::max(gaussians_->max_radii2D_.index({visibility_filter}), radii.index({visibility_filter})));
            // if (!isdoingGausPyramidTraining() || training_level < num_gaus_pyramid_sub_levels_)
            gaussians_->addDensificationStats(viewspace_point_tensor, visibility_filter);

            if ((getIteration() > opt_params_.densify_from_iter_) &&
                (getIteration() % densifyInterval() == 0))
            {
                int size_threshold = (getIteration() > prune_big_point_after_iter_) ? 20 : 0;
                gaussians_->densifyAndPrune(
                    densifyGradThreshold(),
                    densify_min_opacity_, // 0.005,//
                    scene_->cameras_extent_,
                    size_threshold);
            }

            if (opacityResetInterval() && (getIteration() % opacityResetInterval() == 0 || (model_params_.white_background_ && getIteration() == opt_params_.densify_from_iter_)))
                gaussians_->resetOpacity();
        }

        // 优化步
        gaussians_->optimizer_->step();
        gaussians_->optimizer_->zero_grad(true);
    }
}

void GaussianMapper::trainingReport(
    int iteration,
    int num_iterations,
    torch::Tensor &Ll1,
    torch::Tensor &loss,
    float ema_loss_for_log,
    std::function<torch::Tensor(torch::Tensor &, torch::Tensor &)> l1_loss,
    int64_t elapsed_time,
    GaussianModel &gaussians,
    GaussianScene &scene,
    GaussianPipelineParams &pipe,
    torch::Tensor &background)
{
    std::cout << std::fixed << std::setprecision(8)
              << "Training iteration " << iteration << "/" << num_iterations
              << ", time elapsed:" << elapsed_time / 1000.0 << "s"
              << ", ema_loss:" << ema_loss_for_log
              << ", num_points:" << gaussians.xyz_.size(0)
              << std::endl;
}

bool GaussianMapper::isStopped()
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    return this->stopped_;
}

void GaussianMapper::signalStop(const bool going_to_stop)
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    this->stopped_ = going_to_stop;
}

void GaussianMapper::generateKfidRandomShuffle()
{
    // if (viewpoint_sliding_window_.empty())
    //     return;

    // std::size_t sliding_window_size = viewpoint_sliding_window_.size();
    // kfid_shuffle_.resize(sliding_window_size);
    // std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
    // std::mt19937 g(rd_());
    // std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

    if (scene_->keyframes().empty())
        return;

    std::size_t nkfs = scene_->keyframes().size();
    kfid_shuffle_.resize(nkfs);
    std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
    std::mt19937 g(rd_());
    std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

    kfid_shuffled_ = true;
}

std::shared_ptr<GaussianKeyframe>
GaussianMapper::useOneRandomSlidingWindowKeyframe()
{
    // auto t1 = std::chrono::steady_clock::now();
    if (scene_->keyframes().empty())
        return nullptr;

    if (!kfid_shuffled_)
        generateKfidRandomShuffle();

    std::shared_ptr<GaussianKeyframe> viewpoint_cam = nullptr;
    int random_cam_idx;

    if (kfid_shuffled_)
    {
        int start_shuffle_idx = kfid_shuffle_idx_;
        do
        {
            // Next shuffled idx
            ++kfid_shuffle_idx_;
            if (kfid_shuffle_idx_ >= kfid_shuffle_.size())
                kfid_shuffle_idx_ = 0;
            // Add 1 time of use to all kfs if they are all unavalible
            if (kfid_shuffle_idx_ == start_shuffle_idx)
                for (auto &kfit : scene_->keyframes())
                    increaseKeyframeTimesOfUse(kfit.second, 1);
            // Get viewpoint kf
            random_cam_idx = kfid_shuffle_[kfid_shuffle_idx_];
            auto random_cam_it = scene_->keyframes().begin();
            for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
                ++random_cam_it;
            viewpoint_cam = (*random_cam_it).second;
        } while (viewpoint_cam->remaining_times_of_use_ <= 0);
    }

    // Count used times
    auto viewpoint_fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];

    // Handle times of use
    --(viewpoint_cam->remaining_times_of_use_);

    // auto t2 = std::chrono::steady_clock::now();
    // auto t21 = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
    // std::cout<<t21 <<" ns"<<std::endl;
    return viewpoint_cam;
}

std::shared_ptr<GaussianKeyframe>
GaussianMapper::useOneRandomKeyframe()
{
    if (scene_->keyframes().empty())
        return nullptr;

    // Get randomly
    int nkfs = static_cast<int>(scene_->keyframes().size());
    int random_cam_idx = std::rand() / ((RAND_MAX + 1u) / nkfs);
    auto random_cam_it = scene_->keyframes().begin();
    for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
        ++random_cam_it;
    std::shared_ptr<GaussianKeyframe> viewpoint_cam = (*random_cam_it).second;

    // Count used times
    auto viewpoint_fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];

    return viewpoint_cam;
}

void GaussianMapper::increaseKeyframeTimesOfUse(
    std::shared_ptr<GaussianKeyframe> pkf,
    int times)
{
    pkf->remaining_times_of_use_ += times;
}

cv::Mat GaussianMapper::renderFromPose(
    const Sophus::SE3f &Tcw,
    const int width,
    const int height,
    const bool main_vision)
{
    if (!initial_mapped_ || getIteration() <= 0)
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>();
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    // Pose
    pkf->setPose(
        Tcw.unit_quaternion().cast<double>(),
        Tcw.translation().cast<double>());
    try
    {
        // Camera
        Camera &camera = scene_->cameras_.at(viewer_camera_id_);
        pkf->setCameraParams(camera);
        // Transformations
        pkf->computeTransformTensors();
    }
    catch (std::out_of_range)
    {
        throw std::runtime_error("[GaussianMapper::renderFromPose]KeyFrame Camera not found!");
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> render_pkg;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        // Render
        render_pkg = GaussianRenderer::render(
            pkf,
            height,
            width,
            gaussians_,
            pipe_params_,
            background_,
            override_color_);
    }

    // Result
    torch::Tensor masked_image;
    if (main_vision)
        masked_image = std::get<0>(render_pkg) * viewer_main_undistort_mask_[pkf->camera_id_];
    else
        masked_image = std::get<0>(render_pkg) * viewer_sub_undistort_mask_[pkf->camera_id_];
    return tensor_utils::torchTensor2CvMat_Float32(masked_image);
}

void GaussianMapper::savePly(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    keyframesToJson(result_dir);
    saveModelParams(result_dir);

    std::filesystem::path ply_dir = result_dir / "point_cloud";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    ply_dir = ply_dir / ("iteration_" + std::to_string(getIteration()));
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    gaussians_->savePly(ply_dir / "point_cloud.ply");
    // gaussians_->saveSparsePointsPly(result_dir / "input.ply");     // 这个应该是为ORBSLAM服务的
}

void GaussianMapper::keyframesToJson(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    std::filesystem::path result_path = result_dir / "cameras.json";
    std::ofstream out_stream;
    out_stream.open(result_path);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json file at " + result_path.string());

    Json::Value json_root;
    Json::StreamWriterBuilder builder;
    const std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());

    int i = 0;
    for (const auto &kfit : scene_->keyframes())
    {
        const auto pkf = kfit.second;
        Eigen::Matrix4f Rt;
        Rt.setZero();
        Eigen::Matrix3f R = pkf->R_quaternion_.toRotationMatrix().cast<float>();
        Rt.topLeftCorner<3, 3>() = R;
        Eigen::Vector3f t = pkf->t_.cast<float>();
        Rt.topRightCorner<3, 1>() = t;
        Rt(3, 3) = 1.0f;

        Eigen::Matrix4f Twc = Rt.inverse();
        Eigen::Vector3f pos = Twc.block<3, 1>(0, 3);
        Eigen::Matrix3f rot = Twc.block<3, 3>(0, 0);

        Json::Value json_kf;
        json_kf["id"] = static_cast<Json::Value::UInt64>(pkf->fid_);
        json_kf["img_name"] = pkf->img_filename_; //(std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_));
        json_kf["width"] = pkf->image_width_;
        json_kf["height"] = pkf->image_height_;

        json_kf["position"][0] = pos.x();
        json_kf["position"][1] = pos.y();
        json_kf["position"][2] = pos.z();

        json_kf["rotation"][0][0] = rot(0, 0);
        json_kf["rotation"][0][1] = rot(0, 1);
        json_kf["rotation"][0][2] = rot(0, 2);
        json_kf["rotation"][1][0] = rot(1, 0);
        json_kf["rotation"][1][1] = rot(1, 1);
        json_kf["rotation"][1][2] = rot(1, 2);
        json_kf["rotation"][2][0] = rot(2, 0);
        json_kf["rotation"][2][1] = rot(2, 1);
        json_kf["rotation"][2][2] = rot(2, 2);

        json_kf["fy"] = graphics_utils::fov2focal(pkf->FoVy_, pkf->image_height_);
        json_kf["fx"] = graphics_utils::fov2focal(pkf->FoVx_, pkf->image_width_);

        json_root[i] = Json::Value(json_kf);
        ++i;
    }

    writer->write(json_root, &out_stream);
}

void GaussianMapper::saveModelParams(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    std::filesystem::path result_path = result_dir / "cfg_args";
    std::ofstream out_stream;
    out_stream.open(result_path);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open file at " + result_path.string());

    out_stream << "Namespace("
               << "eval=" << (model_params_.eval_ ? "True" : "False") << ", "
               << "images=" << "\'" << model_params_.images_ << "\', "
               << "model_path=" << "\'" << model_params_.model_path_.string() << "\', "
               << "resolution=" << model_params_.resolution_ << ", "
               << "sh_degree=" << model_params_.sh_degree_ << ", "
               << "source_path=" << "\'" << model_params_.source_path_.string() << "\', "
               << "white_background=" << (model_params_.white_background_ ? "True" : "False") << ", "
               << ")";

    out_stream.close();
}

void GaussianMapper::writeKeyframeUsedTimes(std::filesystem::path result_dir, std::string name_suffix)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    std::filesystem::path result_path = result_dir / ("keyframe_used_times" + name_suffix + ".txt");
    std::ofstream out_stream;
    out_stream.open(result_path, std::ios::app);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json at " + result_path.string());

    out_stream << "##[Gaussian Mapper]Iteration " << getIteration() << " keyframe id, used times, remaining times:\n";
    for (const auto &used_times_it : kfs_used_times_)
        out_stream << used_times_it.first << " "
                   << used_times_it.second << " "
                   << scene_->keyframes().at(used_times_it.first)->remaining_times_of_use_
                   << "\n";
    out_stream << "##=========================================" << std::endl;

    out_stream.close();
}

int GaussianMapper::getIteration()
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    return iteration_;
}
void GaussianMapper::increaseIteration(const int inc)
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    iteration_ += inc;
}

float GaussianMapper::positionLearningRateInit()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.position_lr_init_;
}
float GaussianMapper::featureLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.feature_lr_;
}
float GaussianMapper::opacityLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.opacity_lr_;
}
float GaussianMapper::scalingLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.scaling_lr_;
}
float GaussianMapper::rotationLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.rotation_lr_;
}
float GaussianMapper::percentDense()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.percent_dense_;
}
float GaussianMapper::lambdaDssim()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.lambda_dssim_;
}
int GaussianMapper::opacityResetInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.opacity_reset_interval_;
}
float GaussianMapper::densifyGradThreshold()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.densify_grad_threshold_;
}
int GaussianMapper::densifyInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.densification_interval_;
}
int GaussianMapper::newKeyframeTimesOfUse()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return new_keyframe_times_of_use_;
}
int GaussianMapper::stableNumIterExistence()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return stable_num_iter_existence_;
}
bool GaussianMapper::isKeepingTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return keep_training_;
}
bool GaussianMapper::isdoingGausPyramidTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return do_gaus_pyramid_training_;
}
bool GaussianMapper::isdoingInactiveGeoDensify()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return inactive_geo_densify_;
}

void GaussianMapper::setPositionLearningRateInit(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.position_lr_init_ = lr;
}
void GaussianMapper::setFeatureLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.feature_lr_ = lr;
}
void GaussianMapper::setOpacityLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.opacity_lr_ = lr;
}
void GaussianMapper::setScalingLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.scaling_lr_ = lr;
}
void GaussianMapper::setRotationLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.rotation_lr_ = lr;
}
void GaussianMapper::setPercentDense(const float percent_dense)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.percent_dense_ = percent_dense;
    gaussians_->setPercentDense(percent_dense);
}
void GaussianMapper::setLambdaDssim(const float lambda_dssim)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.lambda_dssim_ = lambda_dssim;
}
void GaussianMapper::setOpacityResetInterval(const int interval)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.opacity_reset_interval_ = interval;
}
void GaussianMapper::setDensifyGradThreshold(const float th)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.densify_grad_threshold_ = th;
}
void GaussianMapper::setDensifyInterval(const int interval)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.densification_interval_ = interval;
}
void GaussianMapper::setNewKeyframeTimesOfUse(const int times)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    new_keyframe_times_of_use_ = times;
}
void GaussianMapper::setStableNumIterExistence(const int niter)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    stable_num_iter_existence_ = niter;
}
void GaussianMapper::setKeepTraining(const bool keep)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    keep_training_ = keep;
}
void GaussianMapper::setDoGausPyramidTraining(const bool gaus_pyramid)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    do_gaus_pyramid_training_ = gaus_pyramid;
}
void GaussianMapper::setDoInactiveGeoDensify(const bool inactive_geo_densify)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    inactive_geo_densify_ = inactive_geo_densify;
}

VariableParameters GaussianMapper::getVaribleParameters()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    VariableParameters params;
    params.position_lr_init = opt_params_.position_lr_init_;
    params.feature_lr = opt_params_.feature_lr_;
    params.opacity_lr = opt_params_.opacity_lr_;
    params.scaling_lr = opt_params_.scaling_lr_;
    params.rotation_lr = opt_params_.rotation_lr_;
    params.percent_dense = opt_params_.percent_dense_;
    params.lambda_dssim = opt_params_.lambda_dssim_;
    params.opacity_reset_interval = opt_params_.opacity_reset_interval_;
    params.densify_grad_th = opt_params_.densify_grad_threshold_;
    params.densify_interval = opt_params_.densification_interval_;
    params.new_kf_times_of_use = new_keyframe_times_of_use_;
    params.stable_num_iter_existence = stable_num_iter_existence_;
    params.keep_training = keep_training_;
    params.do_gaus_pyramid_training = do_gaus_pyramid_training_;
    params.do_inactive_geo_densify = inactive_geo_densify_;
    return params;
}

void GaussianMapper::setVaribleParameters(const VariableParameters &params)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.position_lr_init_ = params.position_lr_init;
    opt_params_.feature_lr_ = params.feature_lr;
    opt_params_.opacity_lr_ = params.opacity_lr;
    opt_params_.scaling_lr_ = params.scaling_lr;
    opt_params_.rotation_lr_ = params.rotation_lr;
    opt_params_.percent_dense_ = params.percent_dense;
    gaussians_->setPercentDense(params.percent_dense);
    opt_params_.lambda_dssim_ = params.lambda_dssim;
    opt_params_.opacity_reset_interval_ = params.opacity_reset_interval;
    opt_params_.densify_grad_threshold_ = params.densify_grad_th;
    opt_params_.densification_interval_ = params.densify_interval;
    new_keyframe_times_of_use_ = params.new_kf_times_of_use;
    stable_num_iter_existence_ = params.stable_num_iter_existence;
    keep_training_ = params.keep_training;
    do_gaus_pyramid_training_ = params.do_gaus_pyramid_training;
    inactive_geo_densify_ = params.do_inactive_geo_densify;
}

void GaussianMapper::loadPly(std::filesystem::path ply_path, std::filesystem::path camera_path)
{
    this->gaussians_->loadPly(ply_path);

    // Camera
    if (!camera_path.empty() && std::filesystem::exists(camera_path))
    {
        cv::FileStorage camera_file(camera_path.string().c_str(), cv::FileStorage::READ);
        if (!camera_file.isOpened())
            throw std::runtime_error("[Gaussian Mapper]Failed to open settings file at: " + camera_path.string());

        Camera camera;
        camera.camera_id_ = 0;
        camera.width_ = camera_file["Camera.w"].operator int();
        camera.height_ = camera_file["Camera.h"].operator int();

        std::string camera_type = camera_file["Camera.type"].string();
        if (camera_type == "Pinhole")
        {
            camera.setModelId(Camera::CameraModelType::PINHOLE);

            float fx = camera_file["Camera.fx"].operator float();
            float fy = camera_file["Camera.fy"].operator float();
            float cx = camera_file["Camera.cx"].operator float();
            float cy = camera_file["Camera.cy"].operator float();

            float k1 = camera_file["Camera.k1"].operator float();
            float k2 = camera_file["Camera.k2"].operator float();
            float p1 = camera_file["Camera.p1"].operator float();
            float p2 = camera_file["Camera.p2"].operator float();
            float k3 = camera_file["Camera.k3"].operator float();

            cv::Mat K = (cv::Mat_<float>(3, 3)
                             << fx,
                         0.f, cx,
                         0.f, fy, cy,
                         0.f, 0.f, 1.f);

            camera.params_[0] = fx;
            camera.params_[1] = fy;
            camera.params_[2] = cx;
            camera.params_[3] = cy;

            std::vector<float> dist_coeff = {k1, k2, p1, p2, k3};
            camera.dist_coeff_ = cv::Mat(5, 1, CV_32F, dist_coeff.data());
            camera.initUndistortRectifyMapAndMask(K, cv::Size(camera.width_, camera.height_), K, false);

            undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    camera.undistort_mask, device_type_);

            cv::Mat viewer_main_undistort_mask;
            int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;
            int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;
            cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                       cv::Size(viewer_image_width_main_, viewer_image_height_main_));
            viewer_main_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_main_undistort_mask, device_type_);
        }
        else
        {
            throw std::runtime_error("[Gaussian Mapper]Unsupported camera model: " + camera_path.string());
        }

        if (!viewer_camera_id_set_)
        {
            viewer_camera_id_ = camera.camera_id_;
            viewer_camera_id_set_ = true;
        }
        this->scene_->addCamera(camera);
    }

    // Ready
    this->initial_mapped_ = true;
    increaseIteration();
}