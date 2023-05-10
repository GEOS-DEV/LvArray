#pragma once

#include "common.hpp"

namespace LvArray
{
namespace dense
{

template< typename INTERFACE, typename MATRIX_A, typename MATRIX_B, typename MATRIX_C, typename SCALAR >
void gemm(
  Operation opA,
  Operation opB,
  SCALAR const alpha,
  MATRIX_A const & Ain,
  MATRIX_B const & Bin,
  SCALAR const beta,
  MATRIX_C const & Cin )
{
  Matrix< SCALAR const > A = toMatrix( Ain, INTERFACE::MEMORY_SPACE, false );
  Matrix< SCALAR const > B = toMatrix( Bin, INTERFACE::MEMORY_SPACE, false );
  Matrix< SCALAR > const C = toMatrix( Cin, INTERFACE::MEMORY_SPACE, true );

  // Check the sizes
  LVARRAY_ERROR_IF_NE( C.sizes[ 0 ], A.sizes[ 0 + (opA != Operation::NO_OP) ] );
  LVARRAY_ERROR_IF_NE( C.sizes[ 1 ], B.sizes[ 1 - (opB != Operation::NO_OP) ] );
  LVARRAY_ERROR_IF_NE( A.sizes[ 1 - (opA != Operation::NO_OP) ],
                       B.sizes[ 0 + (opB != Operation::NO_OP) ] );

  // Check that everything is contiguous
  LVARRAY_ERROR_IF( !A.isContiguous(), "Matrix A must have one stride on dimension." );
  LVARRAY_ERROR_IF( !B.isContiguous(), "Matrix B must have one stride one dimension." );
  LVARRAY_ERROR_IF( !C.isColumnMajor(), "Matrix C must be column major." );

  // TODO(corbett5): Don't think this will work for Hermitian matrices.
  if( !A.isColumnMajor() )
  {
    A = A.transpose();
    opA = transposeOp( opA );
  }
  if( !B.isColumnMajor() )
  {
    B = B.transpose();
    opB = transposeOp( opB );
  }

  INTERFACE::gemm( opA, opB, alpha, A, B, beta, C );
}


} // namespace dense
} // namespace LvArray