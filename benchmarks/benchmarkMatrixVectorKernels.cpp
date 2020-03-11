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
#include "benchmarkMatrixVectorKernels.hpp"

namespace LvArray
{
namespace benchmarking
{

#define MATRIX_VECTOR_KERNEL( a_ij, b_j, c_i ) \
  for( INDEX_TYPE i = 0; i < N; ++i ) \
  { \
    for( INDEX_TYPE j = 0; j < M; ++j ) \
    { \
      c_i += a_ij * b_j; \
    } \
  } \
  return

#define MATRIX_VECTOR_KERNEL_RAJA( a_ij, b_j, c_i ) \
  RAJA::forall< POLICY >( RAJA::TypedRangeSegment< INDEX_TYPE >( 0, N ), \
                          [=] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i ) \
  { \
    for( INDEX_TYPE j = 0; j < M; ++j ) \
    { \
      c_i += a_ij * b_j; \
    } \
  } \
                          ); \
  return

template< typename PERMUTATION >
void MatrixVectorNative< PERMUTATION >::
fortran( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
         ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b,
         ArrayView< VALUE_TYPE, RAJA::PERM_I > const & c,
         INDEX_TYPE const N,
         INDEX_TYPE const M )
{ MATRIX_VECTOR_KERNEL( a( i, j ), b( j ), c( i ) ); }

template< typename PERMUTATION >
void MatrixVectorNative< PERMUTATION >::
subscript( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
           ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b,
           ArrayView< VALUE_TYPE, RAJA::PERM_I > const & c,
           INDEX_TYPE const N,
           INDEX_TYPE const M )
{ MATRIX_VECTOR_KERNEL( a[ i ][ j ], b[ j ], c[ i ] ); }

template< typename PERMUTATION >
void MatrixVectorNative< PERMUTATION >::
rajaView( RajaView< VALUE_TYPE const, PERMUTATION > const & a,
          RajaView< VALUE_TYPE const, RAJA::PERM_I > const & b,
          RajaView< VALUE_TYPE, RAJA::PERM_I > const & c,
          INDEX_TYPE const N,
          INDEX_TYPE const M )
{ MATRIX_VECTOR_KERNEL( a( i, j ), b( j ), c( i ) ); }

template<>
void MatrixVectorNative< RAJA::PERM_IJ >::
pointer( VALUE_TYPE const * const LVARRAY_RESTRICT a,
         VALUE_TYPE const * const LVARRAY_RESTRICT b,
         VALUE_TYPE * const LVARRAY_RESTRICT c,
         INDEX_TYPE const N,
         INDEX_TYPE const M )
{ MATRIX_VECTOR_KERNEL( a[ ACCESS_IJ( N, M, i, j ) ], b[ j ], c[ i ] ); }

template<>
void MatrixVectorNative< RAJA::PERM_JI >::
pointer( VALUE_TYPE const * const LVARRAY_RESTRICT a,
         VALUE_TYPE const * const LVARRAY_RESTRICT b,
         VALUE_TYPE * const LVARRAY_RESTRICT c,
         INDEX_TYPE const N,
         INDEX_TYPE const M )
{ MATRIX_VECTOR_KERNEL( a[ ACCESS_JI( N, M, i, j ) ], b[ j ], c[ i ] ); }

template< typename PERMUTATION, typename POLICY >
void MatrixVectorRAJA< PERMUTATION, POLICY >::
fortran( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
         ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b,
         ArrayView< VALUE_TYPE, RAJA::PERM_I > const & c,
         INDEX_TYPE const N,
         INDEX_TYPE const M )
{ MATRIX_VECTOR_KERNEL_RAJA( a( i, j ), b( j ), c( i ) ); }

template< typename PERMUTATION, typename POLICY >
void MatrixVectorRAJA< PERMUTATION, POLICY >::
subscript( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
           ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b,
           ArrayView< VALUE_TYPE, RAJA::PERM_I > const & c,
           INDEX_TYPE const N,
           INDEX_TYPE const M )
{ MATRIX_VECTOR_KERNEL_RAJA( a[ i ][ j ], b[ j ], c[ i ] ); }

template< typename PERMUTATION, typename POLICY >
void MatrixVectorRAJA< PERMUTATION, POLICY >::
rajaView( RajaView< VALUE_TYPE const, PERMUTATION > const & a,
          RajaView< VALUE_TYPE const, RAJA::PERM_I > const & b,
          RajaView< VALUE_TYPE, RAJA::PERM_I > const & c,
          INDEX_TYPE const N,
          INDEX_TYPE const M )
{ MATRIX_VECTOR_KERNEL_RAJA( a( i, j ), b( j ), c( i ) ); }

template< typename POLICY >
void pointerRajaHelper( RAJA::PERM_IJ,
                        VALUE_TYPE const * const LVARRAY_RESTRICT a,
                        VALUE_TYPE const * const LVARRAY_RESTRICT b,
                        VALUE_TYPE * const LVARRAY_RESTRICT c,
                        INDEX_TYPE const N,
                        INDEX_TYPE const M )
{ MATRIX_VECTOR_KERNEL_RAJA( a[ ACCESS_IJ( N, M, i, j ) ], b[ j ], c[ i ] ); }

template< typename POLICY >
void pointerRajaHelper( RAJA::PERM_JI,
                        VALUE_TYPE const * const LVARRAY_RESTRICT a,
                        VALUE_TYPE const * const LVARRAY_RESTRICT b,
                        VALUE_TYPE * const LVARRAY_RESTRICT c,
                        INDEX_TYPE const N,
                        INDEX_TYPE const M )
{ MATRIX_VECTOR_KERNEL_RAJA( a[ ACCESS_JI( N, M, i, j ) ], b[ j ], c[ i ] ); }

template< typename PERMUTATION, typename POLICY >
void MatrixVectorRAJA< PERMUTATION, POLICY >::
pointer( VALUE_TYPE const * const LVARRAY_RESTRICT a,
         VALUE_TYPE const * const LVARRAY_RESTRICT b,
         VALUE_TYPE * const LVARRAY_RESTRICT c,
         INDEX_TYPE const N,
         INDEX_TYPE const M )
{ return pointerRajaHelper< POLICY >( PERMUTATION {}, a, b, c, N, M ); }


template struct MatrixVectorNative< RAJA::PERM_IJ >;
template struct MatrixVectorNative< RAJA::PERM_JI >;

template struct MatrixVectorRAJA< RAJA::PERM_IJ, serialPolicy >;
template struct MatrixVectorRAJA< RAJA::PERM_JI, serialPolicy >;

#if defined(USE_OPENMP)

template struct MatrixVectorRAJA< RAJA::PERM_IJ, parallelHostPolicy >;
template struct MatrixVectorRAJA< RAJA::PERM_JI, parallelHostPolicy >;

#endif

#if defined(USE_CUDA)

template struct MatrixVectorRAJA< RAJA::PERM_IJ, RAJA::cuda_exec< THREADS_PER_BLOCK > >;
template struct MatrixVectorRAJA< RAJA::PERM_JI, RAJA::cuda_exec< THREADS_PER_BLOCK > >;

#endif

} // namespace benchmarking
} // namespace LvArray
