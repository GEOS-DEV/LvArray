/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "benchmarkOuterProductKernels.hpp"

namespace LvArray
{
namespace benchmarking
{

#define OUTER_PRODUCT_KERNEL( N, M, a_i, b_j, c_ij ) \
  for( INDEX_TYPE i = 0; i < N; ++i ) \
  { \
    for( INDEX_TYPE j = 0; j < M; ++j ) \
    { \
      c_ij = c_ij + a_i * b_j; \
    } \
  } \
  return

#define OUTER_PRODUCT_KERNEL_RAJA( N, M, a_i, b_j, c_ij ) \
  forall< POLICY >( N, [=] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i ) \
  { \
    for( INDEX_TYPE j = 0; j < M; ++j ) \
    { \
      c_ij = c_ij + a_i * b_j; \
    } \
  } ); \
  return

template< typename PERMUTATION >
void OuterProductNative< PERMUTATION >::
fortranArrayKernel( ArrayT< VALUE_TYPE, RAJA::PERM_I > const & a,
                    ArrayT< VALUE_TYPE, RAJA::PERM_I > const & b,
                    ArrayT< VALUE_TYPE, PERMUTATION > const & c )
{ OUTER_PRODUCT_KERNEL( c.size( 0 ), c.size( 1 ), a( i ), b( j ), c( i, j ) ); }

template< typename PERMUTATION >
void OuterProductNative< PERMUTATION >::
fortranViewKernel( ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & a,
                   ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & b,
                   ArrayViewT< VALUE_TYPE, PERMUTATION > const & c )
{ OUTER_PRODUCT_KERNEL( c.size( 0 ), c.size( 1 ), a( i ), b( j ), c( i, j ) ); }

template< typename PERMUTATION >
void OuterProductNative< PERMUTATION >::
fortranSliceKernel( ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a,
                    ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const b,
                    ArraySliceT< VALUE_TYPE, PERMUTATION > const c )
{ OUTER_PRODUCT_KERNEL( c.size( 0 ), c.size( 1 ), a( i ), b( j ), c( i, j ) ); }

template< typename PERMUTATION >
void OuterProductNative< PERMUTATION >::
subscriptArrayKernel( ArrayT< VALUE_TYPE, RAJA::PERM_I > const & a,
                      ArrayT< VALUE_TYPE, RAJA::PERM_I > const & b,
                      ArrayT< VALUE_TYPE, PERMUTATION > const & c )
{ OUTER_PRODUCT_KERNEL( c.size( 0 ), c.size( 1 ), a[ i ], b[ j ], c[ i ][ j ] ); }

template< typename PERMUTATION >
void OuterProductNative< PERMUTATION >::
subscriptViewKernel( ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & a,
                     ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & b,
                     ArrayViewT< VALUE_TYPE, PERMUTATION > const & c )
{ OUTER_PRODUCT_KERNEL( c.size( 0 ), c.size( 1 ), a[ i ], b[ j ], c[ i ][ j ] ); }

template< typename PERMUTATION >
void OuterProductNative< PERMUTATION >::
subscriptSliceKernel( ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a,
                      ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const b,
                      ArraySliceT< VALUE_TYPE, PERMUTATION > const c )
{ OUTER_PRODUCT_KERNEL( c.size( 0 ), c.size( 1 ), a[ i ], b[ j ], c[ i ][ j ] ); }

template< typename PERMUTATION >
void OuterProductNative< PERMUTATION >::
RAJAViewKernel( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                RajaView< VALUE_TYPE const, RAJA::PERM_I > const & b,
                RajaView< VALUE_TYPE, PERMUTATION > const & c )
{ OUTER_PRODUCT_KERNEL( getRAJAViewLayout( c ).sizes[ 0 ], getRAJAViewLayout( c ).sizes[ 1 ], a( i ), b( j ), c( i, j ) ); }

template<>
void OuterProductNative< RAJA::PERM_IJ >::
pointerKernel( INDEX_TYPE const N,
               INDEX_TYPE const M,
               VALUE_TYPE const * const LVARRAY_RESTRICT a,
               VALUE_TYPE const * const LVARRAY_RESTRICT b,
               VALUE_TYPE * const LVARRAY_RESTRICT c )
{ OUTER_PRODUCT_KERNEL( N, M, a[ i ], b[ j ], c[ ACCESS_IJ( N, M, i, j ) ] ); }

template<>
void OuterProductNative< RAJA::PERM_JI >::
pointerKernel( INDEX_TYPE const N,
               INDEX_TYPE const M,
               VALUE_TYPE const * const LVARRAY_RESTRICT a,
               VALUE_TYPE const * const LVARRAY_RESTRICT b,
               VALUE_TYPE * const LVARRAY_RESTRICT c )
{ OUTER_PRODUCT_KERNEL( N, M, a[ i ], b[ j ], c[ ACCESS_JI( N, M, i, j ) ] ); }

template< typename PERMUTATION, typename POLICY >
void OuterProductRAJA< PERMUTATION, POLICY >::
fortranViewKernel( ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & a,
                   ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & b,
                   ArrayViewT< VALUE_TYPE, PERMUTATION > const & c )
{ OUTER_PRODUCT_KERNEL_RAJA( c.size( 0 ), c.size( 1 ), a( i ), b( j ), c( i, j ) ); }

template< typename PERMUTATION, typename POLICY >
void OuterProductRAJA< PERMUTATION, POLICY >::
fortranSliceKernel( ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a,
                    ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const b,
                    ArraySliceT< VALUE_TYPE, PERMUTATION > const c )
{ OUTER_PRODUCT_KERNEL_RAJA( c.size( 0 ), c.size( 1 ), a( i ), b( j ), c( i, j ) ); }

template< typename PERMUTATION, typename POLICY >
void OuterProductRAJA< PERMUTATION, POLICY >::
subscriptViewKernel( ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & a,
                     ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & b,
                     ArrayViewT< VALUE_TYPE, PERMUTATION > const & c )
{ OUTER_PRODUCT_KERNEL_RAJA( c.size( 0 ), c.size( 1 ), a[ i ], b[ j ], c[ i ][ j ] ); }

template< typename PERMUTATION, typename POLICY >
void OuterProductRAJA< PERMUTATION, POLICY >::
subscriptSliceKernel( ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a,
                      ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const b,
                      ArraySliceT< VALUE_TYPE, PERMUTATION > const c )
{ OUTER_PRODUCT_KERNEL_RAJA( c.size( 0 ), c.size( 1 ), a[ i ], b[ j ], c[ i ][ j ] ); }

template< typename PERMUTATION, typename POLICY >
void OuterProductRAJA< PERMUTATION, POLICY >::
RAJAViewKernel( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                RajaView< VALUE_TYPE const, RAJA::PERM_I > const & b,
                RajaView< VALUE_TYPE, PERMUTATION > const & c )
{ OUTER_PRODUCT_KERNEL_RAJA( getRAJAViewLayout( c ).sizes[ 0 ], getRAJAViewLayout( c ).sizes[ 1 ], a( i ), b( j ), c( i, j ) ); }


template< typename POLICY >
void RAJAPointerKernelHelper( RAJA::PERM_IJ,
                              INDEX_TYPE const N,
                              INDEX_TYPE const M,
                              VALUE_TYPE const * const LVARRAY_RESTRICT a,
                              VALUE_TYPE const * const LVARRAY_RESTRICT b,
                              VALUE_TYPE * const LVARRAY_RESTRICT c )
{ OUTER_PRODUCT_KERNEL_RAJA( N, M, a[ i ], b[ j ], c[ ACCESS_IJ( N, M, i, j ) ] ); }

template< typename POLICY >
void RAJAPointerKernelHelper( RAJA::PERM_JI,
                              INDEX_TYPE const N,
                              INDEX_TYPE const M,
                              VALUE_TYPE const * const LVARRAY_RESTRICT a,
                              VALUE_TYPE const * const LVARRAY_RESTRICT b,
                              VALUE_TYPE * const LVARRAY_RESTRICT c )
{ OUTER_PRODUCT_KERNEL_RAJA( N, M, a[ i ], b[ j ], c[ ACCESS_JI( N, M, i, j ) ] ); }

template< typename PERMUTATION, typename POLICY >
void OuterProductRAJA< PERMUTATION, POLICY >::
pointerKernel( INDEX_TYPE const N,
               INDEX_TYPE const M,
               VALUE_TYPE const * const LVARRAY_RESTRICT a,
               VALUE_TYPE const * const LVARRAY_RESTRICT b,
               VALUE_TYPE * const LVARRAY_RESTRICT c )
{ return RAJAPointerKernelHelper< POLICY >( PERMUTATION {}, N, M, a, b, c ); }


template class OuterProductNative< RAJA::PERM_IJ >;
template class OuterProductNative< RAJA::PERM_JI >;

template class OuterProductRAJA< RAJA::PERM_IJ, serialPolicy >;
template class OuterProductRAJA< RAJA::PERM_JI, serialPolicy >;

#if defined(RAJA_ENABLE_OPENMP)
template class OuterProductRAJA< RAJA::PERM_IJ, parallelHostPolicy >;
template class OuterProductRAJA< RAJA::PERM_JI, parallelHostPolicy >;
#endif

#if defined(LVARRAY_USE_CUDA) && defined(LVARRAY_USE_CHAI)
template class OuterProductRAJA< RAJA::PERM_IJ, RAJA::cuda_exec< THREADS_PER_BLOCK > >;
template class OuterProductRAJA< RAJA::PERM_JI, RAJA::cuda_exec< THREADS_PER_BLOCK > >;
#endif

#undef OUTER_PRODUCT_KERNEL
#undef OUTER_PRODUCT_KERNEL_RAJA

} // namespace benchmarking
} // namespace LvArray
