cmake_minimum_required(VERSION 3.24)
project(inferlite LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# 默认按本机 GPU 生成代码;生成编译命令供编辑器使用。
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA 目标架构")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 依赖全部使用系统安装版本,不在配置阶段下载第三方项目;
# 非默认前缀用 -DCMAKE_PREFIX_PATH=<前缀> 指定。
find_package(CUDAToolkit REQUIRED)
find_package(glog REQUIRED)
find_package(re2 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# CONFIGURE_DEPENDS 会在新增源码时触发重新配置。
file(GLOB_RECURSE INFERLITE_SOURCES CONFIGURE_DEPENDS
  InferLite/src/*.cpp InferLite/src/*.cu)
add_library(inferlite SHARED ${INFERLITE_SOURCES})
target_include_directories(inferlite PUBLIC InferLite/include)
target_link_libraries(inferlite PUBLIC
  glog::glog CUDA::cudart re2::re2 nlohmann_json::nlohmann_json)

add_subdirectory(demo)
