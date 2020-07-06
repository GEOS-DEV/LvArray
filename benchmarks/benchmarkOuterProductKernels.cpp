/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2019, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * LLNL-CODE-746361
 *
 * All rights reserved. See COPYRIGHT for details.
 *
 * This file is part of the GEOSX Simulation Framework.
 *
 * GEOSX is a free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License (as published by the
 * Free Software Foundation) version 2.1 dated February 1999.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
fortranArrayKernel( Array< VALUE_TYPE, RAJA::PERM_I > const & a,
                    Array< VALUE_TYPE, RAJA::PERM_I > const & b,
                    Array< VALUE_TYPE, PERMUTATION > const & c )
{ OUTER_PRODUCT_KERNEL( c.size( 0 ), c.size( 1 ), a( i ), b( j ), c( i, j ) ); }

template< typename PERMUTATION >
void OuterProductNative< PERMUTATION >::
fortranViewKernel( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                   ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b,
                   ArrayView< VALUE_TYPE, PERMUTATION > const & c )
{ OUTER_PRODUCT_KERNEL( c.size( 0 ), c.size( 1 ), a( i ), b( j ), c( i, j ) ); }

template< typename PERMUTATION >
void OuterProductNative< PERMUTATION >::
fortranSliceKernel( ArraySlice< VALUE_TYPE const, RAJA::PERM_I > const a,
                    ArraySlice< VALUE_TYPE const, RAJA::PERM_I > const b,
                    ArraySlice< VALUE_TYPE, PERMUTATION > const c )
{ OUTER_PRODUCT_KERNEL( c.size( 0 ), c.size( 1 ), a( i ), b( j ), c( i, j ) ); }

template< typename PERMUTATION >
void OuterProductNative< PERMUTATION >::
subscriptArrayKernel( Array< VALUE_TYPE, RAJA::PERM_I > const & a,
                      Array< VALUE_TYPE, RAJA::PERM_I > const & b,
                      Array< VALUE_TYPE, PERMUTATION > const & c )
{ OUTER_PRODUCT_KERNEL( c.size( 0 ), c.size( 1 ), a[ i ], b[ j ], c[ i ][ j ] ); }

template< typename PERMUTATION >
void OuterProductNative< PERMUTATION >::
subscriptViewKernel( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                     ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b,
                     ArrayView< VALUE_TYPE, PERMUTATION > const & c )
{ OUTER_PRODUCT_KERNEL( c.size( 0 ), c.size( 1 ), a[ i ], b[ j ], c[ i ][ j ] ); }

template< typename PERMUTATION >
void OuterProductNative< PERMUTATION >::
subscriptSliceKernel( ArraySlice< VALUE_TYPE const, RAJA::PERM_I > const a,
                      ArraySlice< VALUE_TYPE const, RAJA::PERM_I > const b,
                      ArraySlice< VALUE_TYPE, PERMUTATION > const c )
{ OUTER_PRODUCT_KERNEL( c.size( 0 ), c.size( 1 ), a[ i ], b[ j ], c[ i ][ j ] ); }

template< typename PERMUTATION >
void OuterProductNative< PERMUTATION >::
RAJAViewKernel( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                RajaView< VALUE_TYPE const, RAJA::PERM_I > const & b,
                RajaView< VALUE_TYPE, PERMUTATION > const & c )
{ OUTER_PRODUCT_KERNEL( c.layout.sizes[ 0 ], c.layout.sizes[ 1 ], a( i ), b( j ), c( i, j ) ); }

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
fortranViewKernel( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                   ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b,
                   ArrayView< VALUE_TYPE, PERMUTATION > const & c )
{ OUTER_PRODUCT_KERNEL_RAJA( c.size( 0 ), c.size( 1 ), a( i ), b( j ), c( i, j ) ); }

template< typename PERMUTATION, typename POLICY >
void OuterProductRAJA< PERMUTATION, POLICY >::
fortranSliceKernel( ArraySlice< VALUE_TYPE const, RAJA::PERM_I > const a,
                    ArraySlice< VALUE_TYPE const, RAJA::PERM_I > const b,
                    ArraySlice< VALUE_TYPE, PERMUTATION > const c )
{ OUTER_PRODUCT_KERNEL_RAJA( c.size( 0 ), c.size( 1 ), a( i ), b( j ), c( i, j ) ); }

template< typename PERMUTATION, typename POLICY >
void OuterProductRAJA< PERMUTATION, POLICY >::
subscriptViewKernel( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                     ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b,
                     ArrayView< VALUE_TYPE, PERMUTATION > const & c )
{ OUTER_PRODUCT_KERNEL_RAJA( c.size( 0 ), c.size( 1 ), a[ i ], b[ j ], c[ i ][ j ] ); }

template< typename PERMUTATION, typename POLICY >
void OuterProductRAJA< PERMUTATION, POLICY >::
subscriptSliceKernel( ArraySlice< VALUE_TYPE const, RAJA::PERM_I > const a,
                      ArraySlice< VALUE_TYPE const, RAJA::PERM_I > const b,
                      ArraySlice< VALUE_TYPE, PERMUTATION > const c )
{ OUTER_PRODUCT_KERNEL_RAJA( c.size( 0 ), c.size( 1 ), a[ i ], b[ j ], c[ i ][ j ] ); }

template< typename PERMUTATION, typename POLICY >
void OuterProductRAJA< PERMUTATION, POLICY >::
RAJAViewKernel( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                RajaView< VALUE_TYPE const, RAJA::PERM_I > const & b,
                RajaView< VALUE_TYPE, PERMUTATION > const & c )
{ OUTER_PRODUCT_KERNEL_RAJA( c.layout.sizes[ 0 ], c.layout.sizes[ 1 ], a( i ), b( j ), c( i, j ) ); }


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

#if defined(USE_OPENMP)
template class OuterProductRAJA< RAJA::PERM_IJ, parallelHostPolicy >;
template class OuterProductRAJA< RAJA::PERM_JI, parallelHostPolicy >;
#endif

#if defined(USE_CUDA) && defined(USE_CHAI)
template class OuterProductRAJA< RAJA::PERM_IJ, RAJA::cuda_exec< THREADS_PER_BLOCK > >;
template class OuterProductRAJA< RAJA::PERM_JI, RAJA::cuda_exec< THREADS_PER_BLOCK > >;
#endif

#undef OUTER_PRODUCT_KERNEL
#undef OUTER_PRODUCT_KERNEL_RAJA

} // namespace benchmarking
} // namespace LvArray
