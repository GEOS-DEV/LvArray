#include "common.hpp"

namespace LvArray
{
namespace dense
{

Operation transposeOp( Operation const op )
{
  switch( op )
  {
    case Operation::NO_OP: return Operation::TRANSPOSE;
    case Operation::TRANSPOSE: return Operation::NO_OP;
    case Operation::ADJOINT: LVARRAY_ERROR( "Not supported" );
  }

  return Operation::NO_OP;
}

} // namespace dense
} // namespace LvArray