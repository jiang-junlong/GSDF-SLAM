cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(photo-slam LANGUAGES CXX CUDA)
if(NOT CMAKE_BUILD_TYPE)
    # set(CMAKE_BUILD_TYPE Debug)
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
# 启用 AddressSanitizer
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")
# set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")

set(CAFFE2_USE_CUDNN on)
set(Torch_DIR /home/jiang/envs/libtorch-cu118/share/cmake/Torch)
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(yaml-cpp REQUIRED)

find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)

# find_package(realsense2)
# set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE INTERNAL "No dev warnings") # 关闭pcl烦人的警告
find_package(PCL REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/submodules/Sophus
    ${EIGEN3_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    )
##################################################################################
##  Build the gaussian splatting component libraries to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(simple_knn STATIC
    submodules/simple-knn/simple_knn.cu
    submodules/simple-knn/simple_knn.h
    submodules/simple-knn/spatial.cu
    submodules/simple-knn/spatial.h)
# target_compile_features(simple_knn PUBLIC cxx_std_17)
target_link_libraries(simple_knn "${TORCH_LIBRARIES}")

add_library(cuda_rasterizer STATIC
    include/operate_points.h
    src/operate_points.cu
    include/rasterize_points.h
    src/rasterize_points.cu
    include/stereo_vision.h
    src/stereo_vision.cu
    cuda_rasterizer/auxiliary.h
    cuda_rasterizer/backward.cu
    cuda_rasterizer/backward.h
    cuda_rasterizer/config.h
    cuda_rasterizer/forward.cu
    cuda_rasterizer/forward.h
    cuda_rasterizer/operate_points.h
    cuda_rasterizer/rasterizer.h
    cuda_rasterizer/rasterizer_impl.cu
    cuda_rasterizer/rasterizer_impl.h
    cuda_rasterizer/stereo_vision.h)
set_target_properties(cuda_rasterizer PROPERTIES CUDA_ARCHITECTURES "75;86")
# target_compile_features(cuda_rasterizer PUBLIC cxx_std_17)
target_include_directories(cuda_rasterizer PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cuda_rasterizer
    glm::glm
    "${TORCH_LIBRARIES}"
    Eigen3::Eigen)

##################################################################################
##  Build the ImGui library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(imgui STATIC
    viewer/imgui/imconfig.h
    viewer/imgui/imgui_demo.cpp
    viewer/imgui/imgui_draw.cpp
    viewer/imgui/imgui_impl_glfw.cpp
    viewer/imgui/imgui_impl_glfw.h
    viewer/imgui/imgui_impl_opengl3_loader.h
    viewer/imgui/imgui_impl_opengl3.cpp
    viewer/imgui/imgui_impl_opengl3.h
    viewer/imgui/imgui_internal.h
    viewer/imgui/imgui_tables.cpp
    viewer/imgui/imgui_widgets.cpp
    viewer/imgui/imgui.cpp
    viewer/imgui/imgui.h
    viewer/imgui/imstb_rectpack.h
    viewer/imgui/imstb_textedit.h
    viewer/imgui/imstb_truetype.h)
target_link_libraries(imgui glfw OpenGL::GL)

##################################################################################
##  Build the gaussian mapper library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################
add_library(ply_utils submodules/utils/ply_utils/ply_utils_pcl.cpp
                      submodules/utils/ply_utils/ply_utils_torch.cpp)
target_link_libraries(ply_utils  
                      ${PCL_LIBRARIES}
                      ${TORCH_LIBRARIES}
                      )


add_library(gaussian_mapper STATIC
    include/gaussian_keyframe.h
    include/gaussian_model.h
    include/gaussian_parameters.h
    include/gaussian_rasterizer.h
    include/gaussian_renderer.h
    include/gaussian_scene.h
    include/gaussian_trainer.h
    include/gaussian_mapper.h
    include/general_utils.h
    include/graphics_utils.h
    include/loss_utils.h
    include/sh_utils.h
    include/tensor_utils.h
    include/camera.h
    include/point_cloud.h
    include/types.h
    src/gaussian_keyframe.cpp
    src/gaussian_model.cpp
    src/gaussian_parameters.cpp
    src/gaussian_rasterizer.cpp
    src/gaussian_renderer.cpp
    src/gaussian_scene.cpp
    src/gaussian_trainer.cpp
    src/gaussian_mapper.cpp)
target_link_libraries(gaussian_mapper
    ${OpenCV_LIBRARIES}
    jsoncpp_lib
    "${TORCH_LIBRARIES}"
    Eigen3::Eigen
    simple_knn
    cuda_rasterizer
    ply_utils
    )

##################################################################################
##  Build the viewer library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(gaussian_viewer STATIC
    viewer/drawer_utils.h
    viewer/imgui_viewer.cpp
    viewer/imgui_viewer.h
    )
target_link_libraries(gaussian_viewer
    gaussian_mapper
    imgui
    ${OpenCV_LIBRARIES}
    jsoncpp_lib
    "${TORCH_LIBRARIES}"
    glm::glm
    glfw
    OpenGL::GL)


##################################################################################
# Dataloader
##################################################################################

add_library(data_parser submodules/data_loader/data_parsers/base_parser.cpp
                        submodules/params/params.cpp
                        submodules/utils/utils.cpp
                        )
target_link_libraries(data_parser ${TORCH_LIBRARIES} ply_utils ${OpenCV_LIBS} ${catkin_LIBRARIES} yaml-cpp)

add_library(ray_utils submodules/utils/ray_utils/ray_utils.cpp)
target_link_libraries(ray_utils ${TORCH_LIBRARIES})

add_library(data_loader submodules/data_loader/data_loader.cpp
                        submodules/utils/coordinates.cpp
                        )
target_link_libraries(data_loader data_parser ray_utils cuda_colorize)


##################################################################################
##  toolbox
##################################################################################

add_library(cuda_colorize STATIC
    cuda_colorize/colorize_pointcloud.h
    cuda_colorize/colorize_pointcloud.cu
)

set_target_properties(cuda_colorize PROPERTIES CUDA_ARCHITECTURES "75;86")

target_include_directories(cuda_colorize PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(cuda_colorize
    "${TORCH_LIBRARIES}"
    "${CUDA_LIBRARIES}"
)


##################################################################################
##  Build the test examples to ${PROJECT_SOURCE_DIR}/bin
##################################################################################
find_package(Threads REQUIRED)


# add_executable(view_result demo/view_result.cpp)
# target_link_libraries(view_result
#     gaussian_viewer
#     gaussian_mapper)

add_executable(train_kitti demo/train_kitti.cpp)
target_link_libraries(train_kitti
        gaussian_viewer
        gaussian_mapper
        data_loader
        # cuda_colorize
        Threads::Threads
        )

add_executable(test_dataloader demo/test_dataloader.cpp)
target_link_libraries(test_dataloader
        gaussian_viewer
        gaussian_mapper
        data_loader
        )

add_executable(color_point_cloud color_point_cloud.cpp)
target_link_libraries(color_point_cloud
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
    "${TORCH_LIBRARIES}"
    Eigen3::Eigen
    glm::glm
    glfw
    OpenGL::GL
)
