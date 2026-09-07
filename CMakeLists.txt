cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(inferlite LANGUAGES CXX CUDA)

# C++ 与 CUDA 使用同一语言标准;生成供编辑器使用的编译命令。
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 依赖全部使用系统安装版本,不在配置阶段下载第三方项目。
find_package(CUDAToolkit REQUIRED)
find_package(glog REQUIRED)
find_path(RE2_INCLUDE_DIR re2/re2.h REQUIRED)
find_library(RE2_LIBRARY re2 REQUIRED)
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp REQUIRED)

# 源码统一生成主库;CONFIGURE_DEPENDS 会在新增源码时触发重新配置。
file(GLOB_RECURSE INFERLITE_SOURCES CONFIGURE_DEPENDS
  InferLite/src/*.cpp InferLite/src/*.cu)
add_library(inferlite SHARED ${INFERLITE_SOURCES})
target_include_directories(inferlite PUBLIC
  InferLite/include ${RE2_INCLUDE_DIR} ${NLOHMANN_JSON_INCLUDE_DIR})
target_link_libraries(inferlite PUBLIC
  glog::glog CUDA::cudart ${RE2_LIBRARY})

add_subdirectory(demo)
