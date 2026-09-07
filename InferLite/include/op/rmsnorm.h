#ifndef INFERLITE_INCLUDE_OP_RMSNORM_H_
#define INFERLITE_INCLUDE_OP_RMSNORM_H_
#include "layer.h"
namespace op {
class RmsNormLayer : public LayerParam {
 public:
  explicit RmsNormLayer(int32_t dim);

  base::Status check() const override;

  base::Status forward() override;

 private:
  int32_t dim_ = 0;
};
}  // namespace op
#endif  // INFERLITE_INCLUDE_OP_RMSNORM_H_
