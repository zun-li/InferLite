#ifndef INFERLITE_INCLUDE_OP_ADD_H
#define INFERLITE_INCLUDE_OP_ADD_H
#include "layer.h"
namespace op {
class VecAddLayer : public Layer {
 public:
  VecAddLayer();

  base::Status check() const override;

  base::Status forward() override;
};
}  // namespace op
#endif  // INFERLITE_INCLUDE_OP_ADD_H
