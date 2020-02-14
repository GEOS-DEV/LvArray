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

#define OUTER_PRODUCT_KERNEL( a_i, b_j, c_ij ) \
  for( INDEX_TYPE i = 0 ; i < N ; ++i ) \
  { \
    for( INDEX_TYPE j = 0 ; j < M ; ++j ) \
    { \
      c_ij += a_i * b_j; \
    } \
  } \
  return

#define OUTER_PRODUCT_KERNEL_RAJA( a_i, b_j, c_ij ) \
  RAJA::forall< POLICY >( RAJA::TypedRangeSegment< INDEX_TYPE >( 0, N ), \
                          [=] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i ) \
  { \
    for( INDEX_TYPE j = 0 ; j < M ; ++j ) \
    { \
      c_ij += a_i * b_j; \
    } \
  } \
                          ); \
  return

template< typename PERMUTATION >
void OuterProductNative< PERMUTATION >::
fortran( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
         ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b,
         ArrayView< VALUE_TYPE, PERMUTATION > const & c,
         INDEX_TYPE const N,
         INDEX_TYPE const M )
{ OUTER_PRODUCT_KERNEL( a( i ), b( j ), c( i, j ) ); }

template< typename PERMUTATION >
void OuterProductNative< PERMUTATION >::
subscript( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
           ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b,
           ArrayView< VALUE_TYPE, PERMUTATION > const & c,
           INDEX_TYPE const N,
           INDEX_TYPE const M )
{ OUTER_PRODUCT_KERNEL( a[ i ], b[ j ], c[ i ][ j ] ); }

template< typename PERMUTATION >
void OuterProductNative< PERMUTATION >::
rajaView( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a,
          RajaView< VALUE_TYPE const, RAJA::PERM_I > const & b,
          RajaView< VALUE_TYPE, PERMUTATION > const & c,
          INDEX_TYPE const N,
          INDEX_TYPE const M )
{ OUTER_PRODUCT_KERNEL( a( i ), b( j ), c( i, j ) ); }

template<>
void OuterProductNative< RAJA::PERM_IJ >::
pointer( VALUE_TYPE const * const LVARRAY_RESTRICT a,
         VALUE_TYPE const * const LVARRAY_RESTRICT b,
         VALUE_TYPE * const LVARRAY_RESTRICT c,
         INDEX_TYPE const N,
         INDEX_TYPE const M )
{ OUTER_PRODUCT_KERNEL( a[ i ], b[ j ], c[ M * i + j ] ); }

template<>
void OuterProductNative< RAJA::PERM_JI >::
pointer( VALUE_TYPE const * const LVARRAY_RESTRICT a,
         VALUE_TYPE const * const LVARRAY_RESTRICT b,
         VALUE_TYPE * const LVARRAY_RESTRICT c,
         INDEX_TYPE const N,
         INDEX_TYPE const M )
{ OUTER_PRODUCT_KERNEL( a[ i ], b[ j ], c[ N * j + i ] ); }

template< typename PERMUTATION, typename POLICY >
void OuterProductRAJA< PERMUTATION, POLICY >::
fortran( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
         ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b,
         ArrayView< VALUE_TYPE, PERMUTATION > const & c,
         INDEX_TYPE const N,
         INDEX_TYPE const M )
{ OUTER_PRODUCT_KERNEL_RAJA( a( i ), b( j ), c( i, j ) ); }

template< typename PERMUTATION, typename POLICY >
void OuterProductRAJA< PERMUTATION, POLICY >::
subscript( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
           ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b,
           ArrayView< VALUE_TYPE, PERMUTATION > const & c,
           INDEX_TYPE const N,
           INDEX_TYPE const M )
{ OUTER_PRODUCT_KERNEL_RAJA( a[ i ], b[ j ], c[ i ][ j ] ); }

template< typename PERMUTATION, typename POLICY >
void OuterProductRAJA< PERMUTATION, POLICY >::
rajaView( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a,
          RajaView< VALUE_TYPE const, RAJA::PERM_I > const & b,
          RajaView< VALUE_TYPE, PERMUTATION > const & c,
          INDEX_TYPE const N,
          INDEX_TYPE const M )
{ OUTER_PRODUCT_KERNEL_RAJA( a( i ), b( j ), c( i, j ) ); }

template< typename POLICY >
void pointerRajaHelper( RAJA::PERM_IJ,
                        VALUE_TYPE const * const LVARRAY_RESTRICT a,
                        VALUE_TYPE const * const LVARRAY_RESTRICT b,
                        VALUE_TYPE * const LVARRAY_RESTRICT c,
                        INDEX_TYPE const N,
                        INDEX_TYPE const M )
{ OUTER_PRODUCT_KERNEL_RAJA( a[ i ], b[ j ], c[ M * i + j ] ); }

template< typename POLICY >
void pointerRajaHelper( RAJA::PERM_JI,
                        VALUE_TYPE const * const LVARRAY_RESTRICT a,
                        VALUE_TYPE const * const LVARRAY_RESTRICT b,
                        VALUE_TYPE * const LVARRAY_RESTRICT c,
                        INDEX_TYPE const N,
                        INDEX_TYPE const M )
{ OUTER_PRODUCT_KERNEL_RAJA( a[ i ], b[ j ], c[ N * j + i ] ); }

template< typename PERMUTATION, typename POLICY >
void OuterProductRAJA< PERMUTATION, POLICY >::
pointer( VALUE_TYPE const * const LVARRAY_RESTRICT a,
         VALUE_TYPE const * const LVARRAY_RESTRICT b,
         VALUE_TYPE * const LVARRAY_RESTRICT c,
         INDEX_TYPE const N,
         INDEX_TYPE const M )
{ return pointerRajaHelper< POLICY >( PERMUTATION {}, a, b, c, N, M ); }


template struct OuterProductNative< RAJA::PERM_IJ >;
template struct OuterProductNative< RAJA::PERM_JI >;

template struct OuterProductRAJA< RAJA::PERM_IJ, serialPolicy >;
template struct OuterProductRAJA< RAJA::PERM_JI, serialPolicy >;

#if defined(USE_OPENMP)
template struct OuterProductRAJA< RAJA::PERM_IJ, parallelHostPolicy >;
template struct OuterProductRAJA< RAJA::PERM_JI, parallelHostPolicy >;
#endif

#if defined(USE_CUDA)
template struct OuterProductRAJA< RAJA::PERM_IJ, RAJA::cuda_exec< THREADS_PER_BLOCK > >;
template struct OuterProductRAJA< RAJA::PERM_JI, RAJA::cuda_exec< THREADS_PER_BLOCK > >;
#endif

} // namespace benchmarking
} // namespace LvArray
