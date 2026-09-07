#ifndef INFERLITE_INCLUDE_BASE_BASE_H_
#define INFERLITE_INCLUDE_BASE_BASE_H_
#include <glog/logging.h>
#include <cstdint>
#include <string>

namespace model {
// 模型运行时所需的全部中间缓冲区,在 init_mem 中一次性分配。
enum class ModelBufferType {
  kInputTokens = 0,
  kInputEmbeddings = 1,
  kOutputRMSNorm = 2,
  kKeyCache = 3,
  kValueCache = 4,
  kQuery = 5,
  kInputPos = 6,
  kScoreStorage = 7,
  kOutputMHA = 8,
  kAttnOutput = 9,
  kW1Output = 10,
  kW2Output = 11,
  kW3Output = 12,
  kFFNRMSNorm = 13,
  kForwardOutput = 14,
  kSinCache = 15,
  kCosCache = 16,
};
}  // namespace model

namespace base {
// 只有两类内存:主机内存(权重的 mmap 视图、token id)与显存(其余全部)。
enum class DeviceType : uint8_t {
  kUnknown = 0,
  kCPU = 1,
  kCUDA = 2,
};

enum class DataType : uint8_t {
  kUnknown = 0,
  kFp32 = 1,
  kInt32 = 2,
};

inline size_t DataTypeSize(DataType data_type) {
  if (data_type == DataType::kFp32) {
    return sizeof(float);
  } else if (data_type == DataType::kInt32) {
    return sizeof(int32_t);
  } else {
    return 0;
  }
}

class NoCopyable {
 protected:
  NoCopyable() = default;

  ~NoCopyable() = default;

  NoCopyable(const NoCopyable&) = delete;

  NoCopyable& operator=(const NoCopyable&) = delete;
};

enum StatusCode : uint8_t {
  kSuccess = 0,
  kPathNotValid = 1,
  kModelParseError = 2,
  kInternalError = 3,
  kKeyValueHasExist = 4,
  kInvalidArgument = 5,
};

class Status {
 public:
  Status(int code = StatusCode::kSuccess, std::string err_message = "");

  Status(const Status& other) = default;

  Status& operator=(const Status& other) = default;

  Status& operator=(int code);

  bool operator==(int code) const;

  bool operator!=(int code) const;

  operator int() const;

  operator bool() const;

  int32_t get_err_code() const;

  const std::string& get_err_msg() const;

  void set_err_msg(const std::string& err_msg);

 private:
  int code_ = StatusCode::kSuccess;
  std::string message_;
};

namespace error {
#define STATUS_CHECK(call)                                                                 \
  do {                                                                                     \
    const base::Status& status = call;                                                     \
    if (!status) {                                                                         \
      const size_t buf_size = 512;                                                         \
      char buf[buf_size];                                                                  \
      snprintf(buf, buf_size - 1,                                                          \
               "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", __FILE__, \
               __LINE__, int(status), status.get_err_msg().c_str());                       \
      LOG(FATAL) << buf;                                                                   \
    }                                                                                      \
  } while (0)

Status Success(const std::string& err_msg = "");

Status PathNotValid(const std::string& err_msg = "");

Status ModelParseError(const std::string& err_msg = "");

Status InternalError(const std::string& err_msg = "");

Status KeyHasExits(const std::string& err_msg = "");

Status InvalidArgument(const std::string& err_msg = "");

}  // namespace error

std::ostream& operator<<(std::ostream& os, const Status& x);

}  // namespace base
#endif  // INFERLITE_INCLUDE_BASE_BASE_H_
