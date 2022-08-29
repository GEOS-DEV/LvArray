#include "common.hpp"

namespace LvArray
{
namespace dense
{

////////////////////////////////////////////////////////////////////////////////////////////////////
char const * getOption( SymmetricMatrixStorageType const option )
{
  static constexpr char const * const upper = "U";
  static constexpr char const * const lower = "L";

  return option == SymmetricMatrixStorageType::UPPER_TRIANGULAR ? upper : lower;
}

} // namespace dense
} // namespace LvArray