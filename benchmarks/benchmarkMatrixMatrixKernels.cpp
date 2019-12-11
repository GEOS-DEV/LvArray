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
#include "benchmarkMatrixMatrixKernels.hpp"

namespace LvArray
{
namespace benchmarking
{

#define MATRIX_MATRIX_KERNEL( a_ik, b_kj, c_ij ) \
  for( INDEX_TYPE i = 0 ; i < N ; ++i ) \
  { \
    for( INDEX_TYPE j = 0 ; j < M ; ++j ) \
    { \
      VALUE_TYPE dot = 0; \
      for( INDEX_TYPE k = 0 ; k < P ; ++k ) \
      { \
        dot += a_ik * b_kj; \
      } \
      c_ij += dot; \
    } \
  } \
  return

#define MATRIX_MATRIX_KERNEL_RAJA( a_ik, b_kj, c_ij ) \
  RAJA::forall< POLICY >( RAJA::TypedRangeSegment< INDEX_TYPE >( 0, N ), \
                          [=] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i ) \
  { \
    for( INDEX_TYPE j = 0 ; j < M ; ++j ) \
    { \
      VALUE_TYPE dot = 0; \
      for( INDEX_TYPE k = 0 ; k < P ; ++k ) \
      { \
        dot += a_ik * b_kj; \
      } \
      c_ij += dot; \
    } \
  } \
                          ); \
  return

template< typename PERMUTATION >
void MatrixMatrixNative< PERMUTATION >::
fortran( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
         ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
         ArrayView< VALUE_TYPE, PERMUTATION > const & c,
         INDEX_TYPE const N,
         INDEX_TYPE const M,
         INDEX_TYPE const P )
{ MATRIX_MATRIX_KERNEL( a( i, k ), b( k, j ), c( i, j ) ); }

template< typename PERMUTATION >
void MatrixMatrixNative< PERMUTATION >::
subscript( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
           ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
           ArrayView< VALUE_TYPE, PERMUTATION > const & c,
           INDEX_TYPE const N,
           INDEX_TYPE const M,
           INDEX_TYPE const P )
{ MATRIX_MATRIX_KERNEL( a[ i ][ k ], b[ k ][ j ], c[ i ][ j ] ); }

template< typename PERMUTATION >
void MatrixMatrixNative< PERMUTATION >::
rajaView( RajaView< VALUE_TYPE const, PERMUTATION > const & a,
          RajaView< VALUE_TYPE const, PERMUTATION > const & b,
          RajaView< VALUE_TYPE, PERMUTATION > const & c,
          INDEX_TYPE const N,
          INDEX_TYPE const M,
          INDEX_TYPE const P )
{ MATRIX_MATRIX_KERNEL( a( i, k ), b( k, j ), c( i, j ) ); }

template<>
void MatrixMatrixNative< RAJA::PERM_IJ >::
pointer( VALUE_TYPE const * const restrict a,
         VALUE_TYPE const * const restrict b,
         VALUE_TYPE * const restrict c,
         INDEX_TYPE const N,
         INDEX_TYPE const M,
         INDEX_TYPE const P )
{
  MATRIX_MATRIX_KERNEL( a[ ACCESS_IJ( N, P, i, k ) ],
                        b[ ACCESS_IJ( L, M, k, j ) ],
                        c[ ACCESS_IJ( N, M, i, j ) ] );
}

template<>
void MatrixMatrixNative< RAJA::PERM_JI >::
pointer( VALUE_TYPE const * const restrict a,
         VALUE_TYPE const * const restrict b,
         VALUE_TYPE * const restrict c,
         INDEX_TYPE const N,
         INDEX_TYPE const M,
         INDEX_TYPE const P )
{
  MATRIX_MATRIX_KERNEL( a[ ACCESS_JI( N, P, i, k ) ],
                        b[ ACCESS_JI( P, M, k, j ) ],
                        c[ ACCESS_JI( N, M, i, j ) ] );
}

template< typename PERMUTATION, typename POLICY >
void MatrixMatrixRAJA< PERMUTATION, POLICY >::
fortran( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
         ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
         ArrayView< VALUE_TYPE, PERMUTATION > const & c,
         INDEX_TYPE const N,
         INDEX_TYPE const M,
         INDEX_TYPE const P )
{ MATRIX_MATRIX_KERNEL_RAJA( a( i, k ), b( k, j ), c( i, j ) ); }

template< typename PERMUTATION, typename POLICY >
void MatrixMatrixRAJA< PERMUTATION, POLICY >::
subscript( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
           ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
           ArrayView< VALUE_TYPE, PERMUTATION > const & c,
           INDEX_TYPE const N,
           INDEX_TYPE const M,
           INDEX_TYPE const P )
{ MATRIX_MATRIX_KERNEL_RAJA( a[ i ][ k ], b[ k ][ j ], c[ i ][ j ] ); }

template< typename PERMUTATION, typename POLICY >
void MatrixMatrixRAJA< PERMUTATION, POLICY >::
rajaView( RajaView< VALUE_TYPE const, PERMUTATION > const & a,
          RajaView< VALUE_TYPE const, PERMUTATION > const & b,
          RajaView< VALUE_TYPE, PERMUTATION > const & c,
          INDEX_TYPE const N,
          INDEX_TYPE const M,
          INDEX_TYPE const P )
{ MATRIX_MATRIX_KERNEL_RAJA( a( i, k ), b( k, j ), c( i, j ) ); }

template< typename POLICY >
void pointerRajaHelper( RAJA::PERM_IJ,
                        VALUE_TYPE const * const restrict a,
                        VALUE_TYPE const * const restrict b,
                        VALUE_TYPE * const restrict c,
                        INDEX_TYPE const N,
                        INDEX_TYPE const M,
                        INDEX_TYPE const P )
{
  MATRIX_MATRIX_KERNEL_RAJA( a[ ACCESS_IJ( N, P, i, k ) ],
                             b[ ACCESS_IJ( P, M, k, j ) ],
                             c[ ACCESS_IJ( N, M, i, j ) ] );
}

template< typename POLICY >
void pointerRajaHelper( RAJA::PERM_JI,
                        VALUE_TYPE const * const restrict a,
                        VALUE_TYPE const * const restrict b,
                        VALUE_TYPE * const restrict c,
                        INDEX_TYPE const N,
                        INDEX_TYPE const M,
                        INDEX_TYPE const P )
{
  MATRIX_MATRIX_KERNEL_RAJA( a[ ACCESS_JI( N, P, i, k ) ],
                             b[ ACCESS_JI( P, M, k, j ) ],
                             c[ ACCESS_JI( N, M, i, j ) ] );
}

template< typename PERMUTATION, typename POLICY >
void MatrixMatrixRAJA< PERMUTATION, POLICY >::
pointer( VALUE_TYPE const * const restrict a,
         VALUE_TYPE const * const restrict b,
         VALUE_TYPE * const restrict c,
         INDEX_TYPE const N,
         INDEX_TYPE const M,
         INDEX_TYPE const P )
{ return pointerRajaHelper< POLICY >( PERMUTATION {}, a, b, c, N, M, P ); }


template struct MatrixMatrixNative< RAJA::PERM_IJ >;
template struct MatrixMatrixNative< RAJA::PERM_JI >;

template struct MatrixMatrixRAJA< RAJA::PERM_IJ, serialPolicy >;
template struct MatrixMatrixRAJA< RAJA::PERM_JI, serialPolicy >;

#if defined(USE_OPENMP)

template struct MatrixMatrixRAJA< RAJA::PERM_IJ, parallelHostPolicy >;
template struct MatrixMatrixRAJA< RAJA::PERM_JI, parallelHostPolicy >;

#endif

#if defined(USE_CUDA)

template struct MatrixMatrixRAJA< RAJA::PERM_IJ, RAJA::cuda_exec< THREADS_PER_BLOCK > >;
template struct MatrixMatrixRAJA< RAJA::PERM_JI, RAJA::cuda_exec< THREADS_PER_BLOCK > >;

#endif

} // namespace benchmarking
} // namespace LvArray
