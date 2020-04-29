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
#include "benchmarkArray1DR2TensorMultiplicationKernels.hpp"

namespace LvArray
{
namespace benchmarking
{


#define OUTER_LOOP( BODY ) \
  for( INDEX_TYPE i = 0; i < N; ++i ) \
  { \
    BODY \
  } \
  return


#define RAJA_OUTER_LOOP( BODY ) \
  RAJA::forall< POLICY >( RAJA::TypedRangeSegment< INDEX_TYPE >( 0, N ), \
                          [=] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i ) \
  { \
    BODY \
  } ); \
  return


#define INNER_LOOP( a_ijl, b_ilk, c_ijk ) \
  for( INDEX_TYPE j = 0; j < 3; ++j ) \
  { \
    for( INDEX_TYPE k = 0; k < 3; ++k ) \
    { \
      VALUE_TYPE dot = 0; \
      for( INDEX_TYPE l = 0; l < 3; ++l ) \
      { \
        dot += a_ijl * b_ilk; \
      } \
      c_ijk += dot; \
    } \
  }


#define KERNEL( a_ijl, b_ilk, c_ijk ) \
  OUTER_LOOP( INNER_LOOP( a_ijl, b_ilk, c_ijk ) )


#define RAJA_KERNEL( a_ijl, b_ilk, c_ijk ) \
  RAJA_OUTER_LOOP( INNER_LOOP( a_ijl, b_ilk, c_ijk ) )


template< int USD >
RAJA_INLINE LVARRAY_HOST_DEVICE constexpr
void R2TensorMultiply( ArraySlice< VALUE_TYPE const, 2, USD > const & a,
                       ArraySlice< VALUE_TYPE const, 2, USD > const & b,
                       ArraySlice< VALUE_TYPE, 2, USD > const & c )
{ INNER_LOOP( a( j, l ), b( l, k ), c( j, k ) ) }


template< typename PERMUTATION >
void Array1DR2TensorMultiplicationNative< PERMUTATION >::
fortran( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
         ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
         ArrayView< VALUE_TYPE, PERMUTATION > const & c,
         INDEX_TYPE const N )
{ KERNEL( a( i, j, l ), b( i, l, k ), c( i, j, k ) ); }


template< typename PERMUTATION >
void Array1DR2TensorMultiplicationNative< PERMUTATION >::
subscript( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
           ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
           ArrayView< VALUE_TYPE, PERMUTATION > const & c,
           INDEX_TYPE const N )
{ KERNEL( a[ i ][ j ][ l ], b[ i ][ l ][ k ], c[ i ][ j ][ k ] ); }


template< typename PERMUTATION >
void Array1DR2TensorMultiplicationNative< PERMUTATION >::
tensorAbstraction( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
                   ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
                   ArrayView< VALUE_TYPE, PERMUTATION > const & c,
                   INDEX_TYPE const N )
{ OUTER_LOOP( R2TensorMultiply( a[ i ], b[ i ], c[ i ] ); ); }


template< typename PERMUTATION >
void Array1DR2TensorMultiplicationNative< PERMUTATION >::
rajaView( RajaView< VALUE_TYPE const, PERMUTATION > const & a,
          RajaView< VALUE_TYPE const, PERMUTATION > const & b,
          RajaView< VALUE_TYPE, PERMUTATION > const & c,
          INDEX_TYPE const N )
{ KERNEL( a( i, j, l ), b( i, l, k ), c( i, j, k ) ); }


template<>
void Array1DR2TensorMultiplicationNative< RAJA::PERM_IJK >::
pointer( VALUE_TYPE const * const LVARRAY_RESTRICT a,
         VALUE_TYPE const * const LVARRAY_RESTRICT b,
         VALUE_TYPE * const LVARRAY_RESTRICT c,
         INDEX_TYPE const N )
{
  KERNEL( a[ ACCESS_IJK( N, 3, 3, i, j, l ) ],
          b[ ACCESS_IJK( N, 3, 3, i, l, k ) ],
          c[ ACCESS_IJK( N, 3, 3, i, j, k ) ] );
}


template<>
void Array1DR2TensorMultiplicationNative< RAJA::PERM_KJI >::
pointer( VALUE_TYPE const * const LVARRAY_RESTRICT a,
         VALUE_TYPE const * const LVARRAY_RESTRICT b,
         VALUE_TYPE * const LVARRAY_RESTRICT c,
         INDEX_TYPE const N )
{
  KERNEL( a[ ACCESS_KJI( N, 3, 3, i, j, l ) ],
          b[ ACCESS_KJI( N, 3, 3, i, l, k ) ],
          c[ ACCESS_KJI( N, 3, 3, i, j, k ) ] );
}


template< typename PERMUTATION, typename POLICY >
void Array1DR2TensorMultiplicationRaja< PERMUTATION, POLICY >::
fortran( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
         ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
         ArrayView< VALUE_TYPE, PERMUTATION > const & c,
         INDEX_TYPE const N )
{ RAJA_KERNEL( a( i, j, l ), b( i, l, k ), c( i, j, k ) ); }


template< typename PERMUTATION, typename POLICY >
void Array1DR2TensorMultiplicationRaja< PERMUTATION, POLICY >::
subscript( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
           ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
           ArrayView< VALUE_TYPE, PERMUTATION > const & c,
           INDEX_TYPE const N )
{ RAJA_KERNEL( a[ i ][ j ][ l ], b[ i ][ l ][ k ], c[ i ][ j ][ k ] ); }


template< typename PERMUTATION, typename POLICY >
void Array1DR2TensorMultiplicationRaja< PERMUTATION, POLICY >::
tensorAbstraction( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
                   ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
                   ArrayView< VALUE_TYPE, PERMUTATION > const & c,
                   INDEX_TYPE const N )
{ RAJA_OUTER_LOOP( R2TensorMultiply( a[ i ], b[ i ], c[ i ] ); ); }


template< typename PERMUTATION, typename POLICY >
void Array1DR2TensorMultiplicationRaja< PERMUTATION, POLICY >::
rajaView( RajaView< VALUE_TYPE const, PERMUTATION > const & a,
          RajaView< VALUE_TYPE const, PERMUTATION > const & b,
          RajaView< VALUE_TYPE, PERMUTATION > const & c,
          INDEX_TYPE const N )
{ RAJA_KERNEL( a( i, j, l ), b( i, l, k ), c( i, j, k ) ); }


template< typename POLICY >
void pointerRajaHelper( RAJA::PERM_IJK,
                        VALUE_TYPE const * const LVARRAY_RESTRICT a,
                        VALUE_TYPE const * const LVARRAY_RESTRICT b,
                        VALUE_TYPE * const LVARRAY_RESTRICT c,
                        INDEX_TYPE const N )
{
  RAJA_KERNEL( a[ ACCESS_IJK( N, 3, 3, i, j, l ) ],
               b[ ACCESS_IJK( N, 3, 3, i, l, k ) ],
               c[ ACCESS_IJK( N, 3, 3, i, j, k ) ] );
}


template< typename POLICY >
void pointerRajaHelper( RAJA::PERM_KJI,
                        VALUE_TYPE const * const LVARRAY_RESTRICT a,
                        VALUE_TYPE const * const LVARRAY_RESTRICT b,
                        VALUE_TYPE * const LVARRAY_RESTRICT c,
                        INDEX_TYPE const N )
{
  RAJA_KERNEL( a[ ACCESS_KJI( N, 3, 3, i, j, l ) ],
               b[ ACCESS_KJI( N, 3, 3, i, l, k ) ],
               c[ ACCESS_KJI( N, 3, 3, i, j, k ) ] );
}


template< typename PERMUTATION, typename POLICY >
void Array1DR2TensorMultiplicationRaja< PERMUTATION, POLICY >::
pointer( VALUE_TYPE const * const LVARRAY_RESTRICT a,
         VALUE_TYPE const * const LVARRAY_RESTRICT b,
         VALUE_TYPE * const LVARRAY_RESTRICT c,
         INDEX_TYPE const N )
{ return pointerRajaHelper< POLICY >( PERMUTATION {}, a, b, c, N ); }


template struct Array1DR2TensorMultiplicationNative< RAJA::PERM_IJK >;
template struct Array1DR2TensorMultiplicationNative< RAJA::PERM_KJI >;

template struct Array1DR2TensorMultiplicationRaja< RAJA::PERM_IJK, serialPolicy >;
template struct Array1DR2TensorMultiplicationRaja< RAJA::PERM_KJI, serialPolicy >;

#if defined(USE_OPENMP)
template struct Array1DR2TensorMultiplicationRaja< RAJA::PERM_IJK, parallelHostPolicy >;
template struct Array1DR2TensorMultiplicationRaja< RAJA::PERM_KJI, parallelHostPolicy >;
#endif

#if defined(USE_CUDA)
template struct Array1DR2TensorMultiplicationRaja< RAJA::PERM_IJK, RAJA::cuda_exec< THREADS_PER_BLOCK > >;
template struct Array1DR2TensorMultiplicationRaja< RAJA::PERM_KJI, RAJA::cuda_exec< THREADS_PER_BLOCK > >;
#endif

} // namespace benchmarking
} // namespace LvArray
