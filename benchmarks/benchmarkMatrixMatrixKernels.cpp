/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "benchmarkMatrixMatrixKernels.hpp"

namespace LvArray
{
namespace benchmarking
{

#define MATRIX_MATRIX_KERNEL( N, L, M, a_ik, b_kj, c_ij ) \
  for( INDEX_TYPE i = 0; i < N; ++i ) \
  { \
    for( INDEX_TYPE j = 0; j < M; ++j ) \
    { \
      VALUE_TYPE dot = 0; \
      for( INDEX_TYPE k = 0; k < L; ++k ) \
      { \
        dot = dot + a_ik * b_kj; \
      } \
      c_ij += dot; \
    } \
  } \
  return

#define MATRIX_MATRIX_KERNEL_RAJA( N, L, M, a_ik, b_kj, c_ij ) \
  forall< POLICY >( N, [=] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i ) \
  { \
    for( INDEX_TYPE j = 0; j < M; ++j ) \
    { \
      VALUE_TYPE dot = 0; \
      for( INDEX_TYPE k = 0; k < L; ++k ) \
      { \
        dot = dot + a_ik * b_kj; \
      } \
      c_ij += dot; \
    } \
  } ); \
  return

template< typename PERMUTATION >
void MatrixMatrixNative< PERMUTATION >::
fortranArrayKernel( ArrayT< VALUE_TYPE, PERMUTATION > const & a,
                    ArrayT< VALUE_TYPE, PERMUTATION > const & b,
                    ArrayT< VALUE_TYPE, PERMUTATION > const & c )
{ MATRIX_MATRIX_KERNEL( a.size( 0 ), a.size( 1 ), b.size( 1 ), a( i, k ), b( k, j ), c( i, j ) ); }

template< typename PERMUTATION >
void MatrixMatrixNative< PERMUTATION >::
fortranViewKernel( ArrayViewT< VALUE_TYPE const, PERMUTATION > const & a,
                   ArrayViewT< VALUE_TYPE const, PERMUTATION > const & b,
                   ArrayViewT< VALUE_TYPE, PERMUTATION > const & c )
{ MATRIX_MATRIX_KERNEL( a.size( 0 ), a.size( 1 ), b.size( 1 ), a( i, k ), b( k, j ), c( i, j ) ); }

template< typename PERMUTATION >
void MatrixMatrixNative< PERMUTATION >::
fortranSliceKernel( ArraySliceT< VALUE_TYPE const, PERMUTATION > const a,
                    ArraySliceT< VALUE_TYPE const, PERMUTATION > const b,
                    ArraySliceT< VALUE_TYPE, PERMUTATION > const c )
{ MATRIX_MATRIX_KERNEL( a.size( 0 ), a.size( 1 ), b.size( 1 ), a( i, k ), b( k, j ), c( i, j ) ); }

template< typename PERMUTATION >
void MatrixMatrixNative< PERMUTATION >::
subscriptArrayKernel( ArrayT< VALUE_TYPE, PERMUTATION > const & a,
                      ArrayT< VALUE_TYPE, PERMUTATION > const & b,
                      ArrayT< VALUE_TYPE, PERMUTATION > const & c )
{ MATRIX_MATRIX_KERNEL( a.size( 0 ), a.size( 1 ), b.size( 1 ), a[ i ][ k ], b[ k ][ j ], c[ i ][ j ] ); }

template< typename PERMUTATION >
void MatrixMatrixNative< PERMUTATION >::
subscriptViewKernel( ArrayViewT< VALUE_TYPE const, PERMUTATION > const & a,
                     ArrayViewT< VALUE_TYPE const, PERMUTATION > const & b,
                     ArrayViewT< VALUE_TYPE, PERMUTATION > const & c )
{ MATRIX_MATRIX_KERNEL( a.size( 0 ), a.size( 1 ), b.size( 1 ), a[ i ][ k ], b[ k ][ j ], c[ i ][ j ] ); }

template< typename PERMUTATION >
void MatrixMatrixNative< PERMUTATION >::
subscriptSliceKernel( ArraySliceT< VALUE_TYPE const, PERMUTATION > const a,
                      ArraySliceT< VALUE_TYPE const, PERMUTATION > const b,
                      ArraySliceT< VALUE_TYPE, PERMUTATION > const c )
{ MATRIX_MATRIX_KERNEL( a.size( 0 ), a.size( 1 ), b.size( 1 ), a[ i ][ k ], b[ k ][ j ], c[ i ][ j ] ); }

template< typename PERMUTATION >
void MatrixMatrixNative< PERMUTATION >::
RAJAViewKernel( RajaView< VALUE_TYPE const, PERMUTATION > const & a,
                RajaView< VALUE_TYPE const, PERMUTATION > const & b,
                RajaView< VALUE_TYPE, PERMUTATION > const & c )
{ MATRIX_MATRIX_KERNEL( getRAJAViewLayout( a ).sizes[ 0 ], getRAJAViewLayout( a ).sizes[ 1 ], getRAJAViewLayout( b ).sizes[ 1 ], a( i, k ), b( k, j ), c( i, j ) ); }

template<>
void MatrixMatrixNative< RAJA::PERM_IJ >::
pointerKernel( INDEX_TYPE const N,
               INDEX_TYPE const L,
               INDEX_TYPE const M,
               VALUE_TYPE const * const LVARRAY_RESTRICT a,
               VALUE_TYPE const * const LVARRAY_RESTRICT b,
               VALUE_TYPE * const LVARRAY_RESTRICT c )
{
  MATRIX_MATRIX_KERNEL( N, L, M,
                        a[ ACCESS_IJ( N, L, i, k ) ],
                        b[ ACCESS_IJ( L, M, k, j ) ],
                        c[ ACCESS_IJ( N, M, i, j ) ] );
}

template<>
void MatrixMatrixNative< RAJA::PERM_JI >::
pointerKernel( INDEX_TYPE const N,
               INDEX_TYPE const L,
               INDEX_TYPE const M,
               VALUE_TYPE const * const LVARRAY_RESTRICT a,
               VALUE_TYPE const * const LVARRAY_RESTRICT b,
               VALUE_TYPE * const LVARRAY_RESTRICT c )
{
  MATRIX_MATRIX_KERNEL( N, L, M,
                        a[ ACCESS_JI( N, L, i, k ) ],
                        b[ ACCESS_JI( L, M, k, j ) ],
                        c[ ACCESS_JI( N, M, i, j ) ] );
}

template< typename PERMUTATION, typename POLICY >
void MatrixMatrixRAJA< PERMUTATION, POLICY >::
fortranViewKernel( ArrayViewT< VALUE_TYPE const, PERMUTATION > const & a,
                   ArrayViewT< VALUE_TYPE const, PERMUTATION > const & b,
                   ArrayViewT< VALUE_TYPE, PERMUTATION > const & c )
{ MATRIX_MATRIX_KERNEL_RAJA( a.size( 0 ), a.size( 1 ), b.size( 1 ), a( i, k ), b( k, j ), c( i, j ) ); }

template< typename PERMUTATION, typename POLICY >
void MatrixMatrixRAJA< PERMUTATION, POLICY >::
fortranSliceKernel( ArraySliceT< VALUE_TYPE const, PERMUTATION > const a,
                    ArraySliceT< VALUE_TYPE const, PERMUTATION > const b,
                    ArraySliceT< VALUE_TYPE, PERMUTATION > const c )
{ MATRIX_MATRIX_KERNEL_RAJA( a.size( 0 ), a.size( 1 ), b.size( 1 ), a( i, k ), b( k, j ), c( i, j ) ); }

template< typename PERMUTATION, typename POLICY >
void MatrixMatrixRAJA< PERMUTATION, POLICY >::
subscriptViewKernel( ArrayViewT< VALUE_TYPE const, PERMUTATION > const & a,
                     ArrayViewT< VALUE_TYPE const, PERMUTATION > const & b,
                     ArrayViewT< VALUE_TYPE, PERMUTATION > const & c )
{ MATRIX_MATRIX_KERNEL_RAJA( a.size( 0 ), a.size( 1 ), b.size( 1 ), a[ i ][ k ], b[ k ][ j ], c[ i ][ j ] ); }

template< typename PERMUTATION, typename POLICY >
void MatrixMatrixRAJA< PERMUTATION, POLICY >::
subscriptSliceKernel( ArraySliceT< VALUE_TYPE const, PERMUTATION > const a,
                      ArraySliceT< VALUE_TYPE const, PERMUTATION > const b,
                      ArraySliceT< VALUE_TYPE, PERMUTATION > const c )
{ MATRIX_MATRIX_KERNEL_RAJA( a.size( 0 ), a.size( 1 ), b.size( 1 ), a[ i ][ k ], b[ k ][ j ], c[ i ][ j ] ); }

template< typename PERMUTATION, typename POLICY >
void MatrixMatrixRAJA< PERMUTATION, POLICY >::
RAJAViewKernel( RajaView< VALUE_TYPE const, PERMUTATION > const & a,
                RajaView< VALUE_TYPE const, PERMUTATION > const & b,
                RajaView< VALUE_TYPE, PERMUTATION > const & c )
{ MATRIX_MATRIX_KERNEL_RAJA( getRAJAViewLayout( a ).sizes[ 0 ], getRAJAViewLayout( a ).sizes[ 1 ], getRAJAViewLayout( b ).sizes[ 1 ], a( i, k ), b( k, j ), c( i, j ) ); }

template< typename POLICY >
void RAJAPointerKernelHelper( RAJA::PERM_IJ,
                              INDEX_TYPE const N,
                              INDEX_TYPE const L,
                              INDEX_TYPE const M,
                              VALUE_TYPE const * const LVARRAY_RESTRICT a,
                              VALUE_TYPE const * const LVARRAY_RESTRICT b,
                              VALUE_TYPE * const LVARRAY_RESTRICT c )
{
  MATRIX_MATRIX_KERNEL_RAJA( N, L, M,
                             a[ ACCESS_IJ( N, L, i, k ) ],
                             b[ ACCESS_IJ( L, M, k, j ) ],
                             c[ ACCESS_IJ( N, M, i, j ) ] );
}

template< typename POLICY >
void RAJAPointerKernelHelper( RAJA::PERM_JI,
                              INDEX_TYPE const N,
                              INDEX_TYPE const L,
                              INDEX_TYPE const M,
                              VALUE_TYPE const * const LVARRAY_RESTRICT a,
                              VALUE_TYPE const * const LVARRAY_RESTRICT b,
                              VALUE_TYPE * const LVARRAY_RESTRICT c )
{
  MATRIX_MATRIX_KERNEL_RAJA( N, L, M,
                             a[ ACCESS_JI( N, L, i, k ) ],
                             b[ ACCESS_JI( L, M, k, j ) ],
                             c[ ACCESS_JI( N, M, i, j ) ] );
}

template< typename PERMUTATION, typename POLICY >
void MatrixMatrixRAJA< PERMUTATION, POLICY >::
pointerKernel( INDEX_TYPE const N,
               INDEX_TYPE const L,
               INDEX_TYPE const M,
               VALUE_TYPE const * const LVARRAY_RESTRICT a,
               VALUE_TYPE const * const LVARRAY_RESTRICT b,
               VALUE_TYPE * const LVARRAY_RESTRICT c )
{ return RAJAPointerKernelHelper< POLICY >( PERMUTATION {}, N, L, M, a, b, c ); }


template class MatrixMatrixNative< RAJA::PERM_IJ >;
template class MatrixMatrixNative< RAJA::PERM_JI >;

template class MatrixMatrixRAJA< RAJA::PERM_IJ, serialPolicy >;
template class MatrixMatrixRAJA< RAJA::PERM_JI, serialPolicy >;

#if defined(RAJA_ENABLE_OPENMP)

template class MatrixMatrixRAJA< RAJA::PERM_IJ, parallelHostPolicy >;
template class MatrixMatrixRAJA< RAJA::PERM_JI, parallelHostPolicy >;

#endif

#if defined(LVARRAY_USE_CUDA) && defined(LVARRAY_USE_CHAI)

template class MatrixMatrixRAJA< RAJA::PERM_IJ, RAJA::cuda_exec< THREADS_PER_BLOCK > >;
template class MatrixMatrixRAJA< RAJA::PERM_JI, RAJA::cuda_exec< THREADS_PER_BLOCK > >;

#endif

} // namespace benchmarking
} // namespace LvArray
