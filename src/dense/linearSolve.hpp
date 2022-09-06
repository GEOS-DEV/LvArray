#pragma once

#include "common.hpp"

namespace LvArray
{
namespace dense
{

/**
 * @brief Solves the matrix equation A X = B for X using (s, d, c, z)gesv.
 * 
 * @tparam T The type of values in the matrices. Must be one of float, double, std::complex< float >, or std::complex< double >.
 * @param backend The built in backend that implements (s, d, c, z)gesv.
 * @param A The input matrix, which is overwritten with L and U from the LU decomposition.
 * @param B The input right hand side, is overwritten with the solution X.
 * @param pivots The permutation matrix used when factoring A.
 * 
 * @note When using @c MAGMA_GPU as the backend both @param A and @param B should be on the GPU while @param pivots
 *   remains on the host.
 */
template< typename T >
void gesv(
  BuiltInBackends const backend,
  Matrix< T > const & A,
  Matrix< T > const & B,
  Vector< DenseInt > const & pivots );

/**
 *
 */
template< typename BACK_END, typename T, int USD_A, int NDIM_B, int USD_B, typename INDEX_TYPE >
void gesv(
  BACK_END && backend,
  ArraySlice< T, 2, USD_A, INDEX_TYPE > const & A,
  ArraySlice< T, NDIM_B, USD_B, INDEX_TYPE > const & B,
  ArraySlice< DenseInt, 1, 0, INDEX_TYPE > const & pivots )
{
  Matrix< T > AMatrix( A );
  Matrix< T > BMatrix( B );
  Vector< DenseInt > pivotsVector( pivots );

  gesv(
    std::forward< BACK_END >( backend ),
    AMatrix,
    BMatrix,
    pivots );
}

/**
 *
 */
template< typename BACK_END, typename T, int USD_A, int NDIM_B, int USD_B, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
void gesv(
  BACK_END && backend,
  ArrayView< T, 2, USD_A, INDEX_TYPE, BUFFER_TYPE > const & A,
  ArrayView< T, NDIM_B, USD_B, INDEX_TYPE, BUFFER_TYPE > const & B,
  ArrayView< DenseInt, 1, 0, INDEX_TYPE, BUFFER_TYPE > const & pivots )
{
  // TODO(corbett5): Unclear about the touch here since A is destroyed but the LU decomposition may still be useful.
  MemorySpace const space = getSpaceForBackend( backend );
  A.move( space, true );
  B.move( space, true );

#if defined( LVARRAY_USE_MAGMA )
  // MAGMA wants the pivots on the CPU.
  if( backend == BuiltInBackends::MAGMA_GPU )
  {
    pivots.move( MemorySpace::host, true );
  }
  else
#endif
  {
    pivots.move( space, true );
  }

  return gesv(
    std::forward< BACK_END >( backend ),
    A.toSlice(),
    B.toSlice(),
    pivots.toSlice() );
}

} // namespace dense
} // namespace LvArray