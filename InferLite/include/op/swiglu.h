#ifndef INFERLITE_INCLUDE_OP_SWIGLU_H_
#define INFERLITE_INCLUDE_OP_SWIGLU_H_
#include "layer.h"
namespace op {
class SwiGLULayer : public Layer {
 public:
  explicit SwiGLULayer(int32_t hidden_dim);

  base::Status check() const override;

  base::Status forward() override;

 private:
  int32_t hidden_dim_ = 0;
};
}  // namespace op
#endif  // INFERLITE_INCLUDE_OP_SWIGLU_H_
