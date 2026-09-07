#ifndef INFERLITE_INCLUDE_OP_MATMUL_H_
#define INFERLITE_INCLUDE_OP_MATMUL_H_
#include "layer.h"
namespace op {
class MatmulLayer : public LayerParam {
 public:
  MatmulLayer(int32_t dim0, int32_t dim1);

  base::Status check() const override;

  base::Status forward() override;

 private:
  int32_t dim0_ = 0;
  int32_t dim1_ = 0;
};
}  // namespace op
#endif  // INFERLITE_INCLUDE_OP_MATMUL_H_
