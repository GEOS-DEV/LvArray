/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "benchmarkMatrixVectorKernels.hpp"

namespace LvArray
{
namespace benchmarking
{

#define MATRIX_VECTOR_KERNEL( N, M, a_ij, b_j, c_i ) \
  for( INDEX_TYPE i = 0; i < N; ++i ) \
  { \
    for( INDEX_TYPE j = 0; j < M; ++j ) \
    { \
      c_i = c_i + a_ij * b_j; \
    } \
  } \
  return

#define MATRIX_VECTOR_KERNEL_RAJA( N, M, a_ij, b_j, c_i ) \
  forall< POLICY >( N, [=] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i ) \
  { \
    for( INDEX_TYPE j = 0; j < M; ++j ) \
    { \
      c_i = c_i + a_ij * b_j; \
    } \
  } ); \
  return

template< typename PERMUTATION >
void MatrixVectorNative< PERMUTATION >::
fortranArrayKernel( ArrayT< VALUE_TYPE, PERMUTATION > const & a,
                    ArrayT< VALUE_TYPE, RAJA::PERM_I > const & b,
                    ArrayT< VALUE_TYPE, RAJA::PERM_I > const & c )
{ MATRIX_VECTOR_KERNEL( a.size( 0 ), a.size( 1 ), a( i, j ), b( j ), c( i ) ); }

template< typename PERMUTATION >
void MatrixVectorNative< PERMUTATION >::
fortranViewKernel( ArrayViewT< VALUE_TYPE const, PERMUTATION > const & a,
                   ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & b,
                   ArrayViewT< VALUE_TYPE, RAJA::PERM_I > const & c )
{ MATRIX_VECTOR_KERNEL( a.size( 0 ), a.size( 1 ), a( i, j ), b( j ), c( i ) ); }

template< typename PERMUTATION >
void MatrixVectorNative< PERMUTATION >::
fortranSliceKernel( ArraySliceT< VALUE_TYPE const, PERMUTATION > const a,
                    ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const b,
                    ArraySliceT< VALUE_TYPE, RAJA::PERM_I > const c )
{ MATRIX_VECTOR_KERNEL( a.size( 0 ), a.size( 1 ), a( i, j ), b( j ), c( i ) ); }

template< typename PERMUTATION >
void MatrixVectorNative< PERMUTATION >::
subscriptArrayKernel( ArrayT< VALUE_TYPE, PERMUTATION > const & a,
                      ArrayT< VALUE_TYPE, RAJA::PERM_I > const & b,
                      ArrayT< VALUE_TYPE, RAJA::PERM_I > const & c )
{ MATRIX_VECTOR_KERNEL( a.size( 0 ), a.size( 1 ), a[ i ][ j ], b[ j ], c[ i ] ); }

template< typename PERMUTATION >
void MatrixVectorNative< PERMUTATION >::
subscriptViewKernel( ArrayViewT< VALUE_TYPE const, PERMUTATION > const & a,
                     ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & b,
                     ArrayViewT< VALUE_TYPE, RAJA::PERM_I > const & c )
{ MATRIX_VECTOR_KERNEL( a.size( 0 ), a.size( 1 ), a[ i ][ j ], b[ j ], c[ i ] ); }

template< typename PERMUTATION >
void MatrixVectorNative< PERMUTATION >::
subscriptSliceKernel( ArraySliceT< VALUE_TYPE const, PERMUTATION > const a,
                      ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const b,
                      ArraySliceT< VALUE_TYPE, RAJA::PERM_I > const c )
{ MATRIX_VECTOR_KERNEL( a.size( 0 ), a.size( 1 ), a[ i ][ j ], b[ j ], c[ i ] ); }

template< typename PERMUTATION >
void MatrixVectorNative< PERMUTATION >::
RAJAViewKernel( RajaView< VALUE_TYPE const, PERMUTATION > const & a,
                RajaView< VALUE_TYPE const, RAJA::PERM_I > const & b,
                RajaView< VALUE_TYPE, RAJA::PERM_I > const & c )
{ MATRIX_VECTOR_KERNEL( getRAJAViewLayout( a ).sizes[ 0 ], getRAJAViewLayout( a ).sizes[ 1 ], a( i, j ), b( j ), c( i ) ); }

template<>
void MatrixVectorNative< RAJA::PERM_IJ >::
pointerKernel( INDEX_TYPE const N,
               INDEX_TYPE const M,
               VALUE_TYPE const * const LVARRAY_RESTRICT a,
               VALUE_TYPE const * const LVARRAY_RESTRICT b,
               VALUE_TYPE * const LVARRAY_RESTRICT c )
{ MATRIX_VECTOR_KERNEL( N, M, a[ ACCESS_IJ( N, M, i, j ) ], b[ j ], c[ i ] ); }

template<>
void MatrixVectorNative< RAJA::PERM_JI >::
pointerKernel( INDEX_TYPE const N,
               INDEX_TYPE const M,
               VALUE_TYPE const * const LVARRAY_RESTRICT a,
               VALUE_TYPE const * const LVARRAY_RESTRICT b,
               VALUE_TYPE * const LVARRAY_RESTRICT c )
{ MATRIX_VECTOR_KERNEL( N, M, a[ ACCESS_JI( N, M, i, j ) ], b[ j ], c[ i ] ); }


template< typename PERMUTATION, typename POLICY >
void MatrixVectorRAJA< PERMUTATION, POLICY >::
fortranViewKernel( ArrayViewT< VALUE_TYPE const, PERMUTATION > const & a,
                   ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & b,
                   ArrayViewT< VALUE_TYPE, RAJA::PERM_I > const & c )
{ MATRIX_VECTOR_KERNEL_RAJA( a.size( 0 ), a.size( 1 ), a( i, j ), b( j ), c( i ) ); }

template< typename PERMUTATION, typename POLICY >
void MatrixVectorRAJA< PERMUTATION, POLICY >::
fortranSliceKernel( ArraySliceT< VALUE_TYPE const, PERMUTATION > const a,
                    ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const b,
                    ArraySliceT< VALUE_TYPE, RAJA::PERM_I > const c )
{ MATRIX_VECTOR_KERNEL_RAJA( a.size( 0 ), a.size( 1 ), a( i, j ), b( j ), c( i ) ); }

template< typename PERMUTATION, typename POLICY >
void MatrixVectorRAJA< PERMUTATION, POLICY >::
subscriptViewKernel( ArrayViewT< VALUE_TYPE const, PERMUTATION > const & a,
                     ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & b,
                     ArrayViewT< VALUE_TYPE, RAJA::PERM_I > const & c )
{ MATRIX_VECTOR_KERNEL_RAJA( a.size( 0 ), a.size( 1 ), a[ i ][ j ], b[ j ], c[ i ] ); }

template< typename PERMUTATION, typename POLICY >
void MatrixVectorRAJA< PERMUTATION, POLICY >::
subscriptSliceKernel( ArraySliceT< VALUE_TYPE const, PERMUTATION > const a,
                      ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const b,
                      ArraySliceT< VALUE_TYPE, RAJA::PERM_I > const c )
{ MATRIX_VECTOR_KERNEL_RAJA( a.size( 0 ), a.size( 1 ), a[ i ][ j ], b[ j ], c[ i ] ); }

template< typename PERMUTATION, typename POLICY >
void MatrixVectorRAJA< PERMUTATION, POLICY >::
RAJAViewKernel( RajaView< VALUE_TYPE const, PERMUTATION > const & a,
                RajaView< VALUE_TYPE const, RAJA::PERM_I > const & b,
                RajaView< VALUE_TYPE, RAJA::PERM_I > const & c )
{ MATRIX_VECTOR_KERNEL_RAJA( getRAJAViewLayout( a ).sizes[ 0 ], getRAJAViewLayout( a ).sizes[ 1 ], a( i, j ), b( j ), c( i ) ); }


template< typename POLICY >
void RAJAPointerKernelHelper( RAJA::PERM_IJ,
                              INDEX_TYPE const N,
                              INDEX_TYPE const M,
                              VALUE_TYPE const * const LVARRAY_RESTRICT a,
                              VALUE_TYPE const * const LVARRAY_RESTRICT b,
                              VALUE_TYPE * const LVARRAY_RESTRICT c )
{ MATRIX_VECTOR_KERNEL_RAJA( N, M, a[ ACCESS_IJ( N, M, i, j ) ], b[ j ], c[ i ] ); }

template< typename POLICY >
void RAJAPointerKernelHelper( RAJA::PERM_JI,
                              INDEX_TYPE const N,
                              INDEX_TYPE const M,
                              VALUE_TYPE const * const LVARRAY_RESTRICT a,
                              VALUE_TYPE const * const LVARRAY_RESTRICT b,
                              VALUE_TYPE * const LVARRAY_RESTRICT c )
{ MATRIX_VECTOR_KERNEL_RAJA( N, M, a[ ACCESS_JI( N, M, i, j ) ], b[ j ], c[ i ] ); }

template< typename PERMUTATION, typename POLICY >
void MatrixVectorRAJA< PERMUTATION, POLICY >::
pointerKernel( INDEX_TYPE const N,
               INDEX_TYPE const M,
               VALUE_TYPE const * const LVARRAY_RESTRICT a,
               VALUE_TYPE const * const LVARRAY_RESTRICT b,
               VALUE_TYPE * const LVARRAY_RESTRICT c )
{ return RAJAPointerKernelHelper< POLICY >( PERMUTATION {}, N, M, a, b, c ); }


template class MatrixVectorNative< RAJA::PERM_IJ >;
template class MatrixVectorNative< RAJA::PERM_JI >;

template class MatrixVectorRAJA< RAJA::PERM_IJ, serialPolicy >;
template class MatrixVectorRAJA< RAJA::PERM_JI, serialPolicy >;

#if defined(RAJA_ENABLE_OPENMP)

template class MatrixVectorRAJA< RAJA::PERM_IJ, parallelHostPolicy >;
template class MatrixVectorRAJA< RAJA::PERM_JI, parallelHostPolicy >;

#endif

#if defined(LVARRAY_USE_CUDA) && defined(LVARRAY_USE_CHAI)

template class MatrixVectorRAJA< RAJA::PERM_IJ, RAJA::cuda_exec< THREADS_PER_BLOCK > >;
template class MatrixVectorRAJA< RAJA::PERM_JI, RAJA::cuda_exec< THREADS_PER_BLOCK > >;

#endif

} // namespace benchmarking
} // namespace LvArray
