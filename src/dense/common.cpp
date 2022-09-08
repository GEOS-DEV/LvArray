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

////////////////////////////////////////////////////////////////////////////////////////////////////
MemorySpace getSpaceForBackend( BuiltInBackends const backend )
{
#if defined( LVARRAY_USE_MAGMA )
  // TODO: This needs to be changed to MemorySpace::hip or whatever.
  if( backend == BuiltInBackends::MAGMA_GPU ) return MemorySpace::cuda;
#else
  LVARRAY_UNUSED_VARIABLE( backend );
#endif

  return MemorySpace::host;
}

} // namespace dense
} // namespace LvArray