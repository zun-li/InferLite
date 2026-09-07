#ifndef INFERLITE_INCLUDE_MODEL_RAW_MODEL_DATA_H_
#define INFERLITE_INCLUDE_MODEL_RAW_MODEL_DATA_H_
#include <cstddef>
#include <cstdint>
namespace model {
// 通过 mmap 直接映射权重文件,所有权重按 fp32 排列在文件头之后。
struct RawModelData {
  ~RawModelData();

  // 以 float 个数为单位的偏移,返回对应权重的起始地址。
  const void* weight(size_t offset) const;

  int32_t fd = -1;
  size_t file_size = 0;
  void* data = nullptr;
  void* weight_data = nullptr;
};
}  // namespace model
#endif  // INFERLITE_INCLUDE_MODEL_RAW_MODEL_DATA_H_
