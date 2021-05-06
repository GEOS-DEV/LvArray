/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

#include "benchmarkArray1DR2TensorMultiplicationKernels.hpp"

namespace LvArray
{
namespace benchmarking
{

#define OUTER_LOOP( N, BODY ) \
  for( INDEX_TYPE i = 0; i < N; ++i ) \
  { \
    BODY \
  } \
  return


#define RAJA_OUTER_LOOP( N, BODY ) \
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
        dot = dot + a_ijl * b_ilk; \
      } \
      c_ijk += dot; \
    } \
  }


#define KERNEL( N, a_ijl, b_ilk, c_ijk ) \
  OUTER_LOOP( N, INNER_LOOP( a_ijl, b_ilk, c_ijk ) )


#define RAJA_KERNEL( N, a_ijl, b_ilk, c_ijk ) \
  RAJA_OUTER_LOOP( N, INNER_LOOP( a_ijl, b_ilk, c_ijk ) )


template< typename VALUE_TYPE_CONST, int USD >
inline LVARRAY_HOST_DEVICE constexpr
void R2TensorMultiplyFortran( LvArray::ArraySlice< VALUE_TYPE_CONST, 2, USD, INDEX_TYPE > const a,
                              LvArray::ArraySlice< VALUE_TYPE_CONST, 2, USD, INDEX_TYPE > const b,
                              LvArray::ArraySlice< VALUE_TYPE, 2, USD, INDEX_TYPE > const c )
{ INNER_LOOP( a( j, l ), b( l, k ), c( j, k ) ) }

template< typename VALUE_TYPE_CONST, int USD >
RAJA_INLINE LVARRAY_HOST_DEVICE constexpr
void R2TensorMultiplySubscript( LvArray::ArraySlice< VALUE_TYPE_CONST, 2, USD, INDEX_TYPE > const a,
                                LvArray::ArraySlice< VALUE_TYPE_CONST, 2, USD, INDEX_TYPE > const b,
                                LvArray::ArraySlice< VALUE_TYPE, 2, USD, INDEX_TYPE > const c )
{ INNER_LOOP( a[ j ][ l ], b[ l ][ k ], c[ j ][ k ] ) }


template< typename PERMUTATION >
void ArrayOfR2TensorsNative< PERMUTATION >::
fortranArrayKernel( ArrayT< VALUE_TYPE, PERMUTATION > const & a,
                    ArrayT< VALUE_TYPE, PERMUTATION > const & b,
                    ArrayT< VALUE_TYPE, PERMUTATION > const & c )
{ KERNEL( a.size( 0 ), a( i, j, l ), b( i, l, k ), c( i, j, k ) ); }

template< typename PERMUTATION >
void ArrayOfR2TensorsNative< PERMUTATION >::
fortranViewKernel( ArrayViewT< VALUE_TYPE const, PERMUTATION > const & a,
                   ArrayViewT< VALUE_TYPE const, PERMUTATION > const & b,
                   ArrayViewT< VALUE_TYPE, PERMUTATION > const & c )
{ KERNEL( a.size( 0 ), a( i, j, l ), b( i, l, k ), c( i, j, k ) ); }

template< typename PERMUTATION >
void ArrayOfR2TensorsNative< PERMUTATION >::
fortranSliceKernel( ArraySliceT< VALUE_TYPE const, PERMUTATION > const a,
                    ArraySliceT< VALUE_TYPE const, PERMUTATION > const b,
                    ArraySliceT< VALUE_TYPE, PERMUTATION > const c )
{ KERNEL( a.size( 0 ), a( i, j, l ), b( i, l, k ), c( i, j, k ) );}

template< typename PERMUTATION >
void ArrayOfR2TensorsNative< PERMUTATION >::
subscriptArrayKernel( ArrayT< VALUE_TYPE, PERMUTATION > const & a,
                      ArrayT< VALUE_TYPE, PERMUTATION > const & b,
                      ArrayT< VALUE_TYPE, PERMUTATION > const & c )
{ KERNEL( a.size( 0 ), a[ i ][ j ][ l ], b[ i ][ l ][ k ], c[ i ][ j ][ k ] ); }

template< typename PERMUTATION >
void ArrayOfR2TensorsNative< PERMUTATION >::
subscriptViewKernel( ArrayViewT< VALUE_TYPE const, PERMUTATION > const & a,
                     ArrayViewT< VALUE_TYPE const, PERMUTATION > const & b,
                     ArrayViewT< VALUE_TYPE, PERMUTATION > const & c )
{ KERNEL( a.size( 0 ), a[ i ][ j ][ l ], b[ i ][ l ][ k ], c[ i ][ j ][ k ] ); }

template< typename PERMUTATION >
void ArrayOfR2TensorsNative< PERMUTATION >::
subscriptSliceKernel( ArraySliceT< VALUE_TYPE const, PERMUTATION > const a,
                      ArraySliceT< VALUE_TYPE const, PERMUTATION > const b,
                      ArraySliceT< VALUE_TYPE, PERMUTATION > const c )
{ KERNEL( a.size( 0 ), a[ i ][ j ][ l ], b[ i ][ l ][ k ], c[ i ][ j ][ k ] ); }

template< typename PERMUTATION >
void ArrayOfR2TensorsNative< PERMUTATION >::
tensorAbstractionFortranArrayKernel( ArrayT< VALUE_TYPE, PERMUTATION > const & a,
                                     ArrayT< VALUE_TYPE, PERMUTATION > const & b,
                                     ArrayT< VALUE_TYPE, PERMUTATION > const & c )
{ OUTER_LOOP( a.size( 0 ), R2TensorMultiplyFortran( a[ i ], b[ i ], c[ i ] ); ); }

template< typename PERMUTATION >
void ArrayOfR2TensorsNative< PERMUTATION >::
tensorAbstractionFortranViewKernel( ArrayViewT< VALUE_TYPE const, PERMUTATION > const & a,
                                    ArrayViewT< VALUE_TYPE const, PERMUTATION > const & b,
                                    ArrayViewT< VALUE_TYPE, PERMUTATION > const & c )
{ OUTER_LOOP( a.size( 0 ), R2TensorMultiplyFortran( a[ i ], b[ i ], c[ i ] ); ); }

template< typename PERMUTATION >
void ArrayOfR2TensorsNative< PERMUTATION >::
tensorAbstractionFortranSliceKernel( ArraySliceT< VALUE_TYPE const, PERMUTATION > const a,
                                     ArraySliceT< VALUE_TYPE const, PERMUTATION > const b,
                                     ArraySliceT< VALUE_TYPE, PERMUTATION > const c )
{ OUTER_LOOP( a.size( 0 ), R2TensorMultiplyFortran( a[ i ], b[ i ], c[ i ] ); ); }

template< typename PERMUTATION >
void ArrayOfR2TensorsNative< PERMUTATION >::
tensorAbstractionSubscriptArrayKernel( ArrayT< VALUE_TYPE, PERMUTATION > const & a,
                                       ArrayT< VALUE_TYPE, PERMUTATION > const & b,
                                       ArrayT< VALUE_TYPE, PERMUTATION > const & c )
{ OUTER_LOOP( a.size( 0 ), R2TensorMultiplySubscript( a[ i ], b[ i ], c[ i ] ); ); }

template< typename PERMUTATION >
void ArrayOfR2TensorsNative< PERMUTATION >::
tensorAbstractionSubscriptViewKernel( ArrayViewT< VALUE_TYPE const, PERMUTATION > const & a,
                                      ArrayViewT< VALUE_TYPE const, PERMUTATION > const & b,
                                      ArrayViewT< VALUE_TYPE, PERMUTATION > const & c )
{ OUTER_LOOP( a.size( 0 ), R2TensorMultiplySubscript( a[ i ], b[ i ], c[ i ] ); ); }

template< typename PERMUTATION >
void ArrayOfR2TensorsNative< PERMUTATION >::
tensorAbstractionSubscriptSliceKernel( ArraySliceT< VALUE_TYPE const, PERMUTATION > const a,
                                       ArraySliceT< VALUE_TYPE const, PERMUTATION > const b,
                                       ArraySliceT< VALUE_TYPE, PERMUTATION > const c )
{ OUTER_LOOP( a.size( 0 ), R2TensorMultiplySubscript( a[ i ], b[ i ], c[ i ] ); ); }

template< typename PERMUTATION >
void ArrayOfR2TensorsNative< PERMUTATION >::
RAJAViewKernel( RajaView< VALUE_TYPE const, PERMUTATION > const & a,
                RajaView< VALUE_TYPE const, PERMUTATION > const & b,
                RajaView< VALUE_TYPE, PERMUTATION > const & c )
{ KERNEL( getRAJAViewLayout( a ).sizes[ 0 ], a( i, j, l ), b( i, l, k ), c( i, j, k ) ); }

template<>
void ArrayOfR2TensorsNative< RAJA::PERM_IJK >::
pointerKernel( INDEX_TYPE const N,
               VALUE_TYPE const * const LVARRAY_RESTRICT a,
               VALUE_TYPE const * const LVARRAY_RESTRICT b,
               VALUE_TYPE * const LVARRAY_RESTRICT c )
{
  KERNEL( N,
          a[ ACCESS_IJK( N, 3, 3, i, j, l ) ],
          b[ ACCESS_IJK( N, 3, 3, i, l, k ) ],
          c[ ACCESS_IJK( N, 3, 3, i, j, k ) ] );
}

template<>
void ArrayOfR2TensorsNative< RAJA::PERM_KJI >::
pointerKernel( INDEX_TYPE const N,
               VALUE_TYPE const * const LVARRAY_RESTRICT a,
               VALUE_TYPE const * const LVARRAY_RESTRICT b,
               VALUE_TYPE * const LVARRAY_RESTRICT c )
{
  KERNEL( N,
          a[ ACCESS_KJI( N, 3, 3, i, j, l ) ],
          b[ ACCESS_KJI( N, 3, 3, i, l, k ) ],
          c[ ACCESS_KJI( N, 3, 3, i, j, k ) ] );
}

template< typename PERMUTATION, typename POLICY >
void ArrayOfR2TensorsRAJA< PERMUTATION, POLICY >::
fortranViewKernel( ArrayViewT< VALUE_TYPE const, PERMUTATION > const & a,
                   ArrayViewT< VALUE_TYPE const, PERMUTATION > const & b,
                   ArrayViewT< VALUE_TYPE, PERMUTATION > const & c )
{ RAJA_KERNEL( a.size( 0 ), a( i, j, l ), b( i, l, k ), c( i, j, k ) ); }

template< typename PERMUTATION, typename POLICY >
void ArrayOfR2TensorsRAJA< PERMUTATION, POLICY >::
fortranSliceKernel( ArraySliceT< VALUE_TYPE const, PERMUTATION > const a,
                    ArraySliceT< VALUE_TYPE const, PERMUTATION > const b,
                    ArraySliceT< VALUE_TYPE, PERMUTATION > const c )
{ RAJA_KERNEL( a.size( 0 ), a( i, j, l ), b( i, l, k ), c( i, j, k ) ); }

template< typename PERMUTATION, typename POLICY >
void ArrayOfR2TensorsRAJA< PERMUTATION, POLICY >::
subscriptViewKernel( ArrayViewT< VALUE_TYPE const, PERMUTATION > const & a,
                     ArrayViewT< VALUE_TYPE const, PERMUTATION > const & b,
                     ArrayViewT< VALUE_TYPE, PERMUTATION > const & c )
{ RAJA_KERNEL( a.size( 0 ), a[ i ][ j ][ l ], b[ i ][ l ][ k ], c[ i ][ j ][ k ] ); }

template< typename PERMUTATION, typename POLICY >
void ArrayOfR2TensorsRAJA< PERMUTATION, POLICY >::
subscriptSliceKernel( ArraySliceT< VALUE_TYPE const, PERMUTATION > const a,
                      ArraySliceT< VALUE_TYPE const, PERMUTATION > const b,
                      ArraySliceT< VALUE_TYPE, PERMUTATION > const c )
{ RAJA_KERNEL( a.size( 0 ), a[ i ][ j ][ l ], b[ i ][ l ][ k ], c[ i ][ j ][ k ] ); }

template< typename PERMUTATION, typename POLICY >
void ArrayOfR2TensorsRAJA< PERMUTATION, POLICY >::
tensorAbstractionFortranViewKernel( ArrayViewT< VALUE_TYPE const, PERMUTATION > const & a,
                                    ArrayViewT< VALUE_TYPE const, PERMUTATION > const & b,
                                    ArrayViewT< VALUE_TYPE, PERMUTATION > const & c )
{ RAJA_OUTER_LOOP( a.size( 0 ), R2TensorMultiplyFortran( a[ i ], b[ i ], c[ i ] ); ); }

template< typename PERMUTATION, typename POLICY >
void ArrayOfR2TensorsRAJA< PERMUTATION, POLICY >::
tensorAbstractionFortranSliceKernel( ArraySliceT< VALUE_TYPE const, PERMUTATION > const a,
                                     ArraySliceT< VALUE_TYPE const, PERMUTATION > const b,
                                     ArraySliceT< VALUE_TYPE, PERMUTATION > const c )
{ RAJA_OUTER_LOOP( a.size( 0 ), R2TensorMultiplyFortran( a[ i ], b[ i ], c[ i ] ); ); }

template< typename PERMUTATION, typename POLICY >
void ArrayOfR2TensorsRAJA< PERMUTATION, POLICY >::
tensorAbstractionSubscriptViewKernel( ArrayViewT< VALUE_TYPE const, PERMUTATION > const & a,
                                      ArrayViewT< VALUE_TYPE const, PERMUTATION > const & b,
                                      ArrayViewT< VALUE_TYPE, PERMUTATION > const & c )
{ RAJA_OUTER_LOOP( a.size( 0 ), R2TensorMultiplySubscript( a[ i ], b[ i ], c[ i ] ); ); }

template< typename PERMUTATION, typename POLICY >
void ArrayOfR2TensorsRAJA< PERMUTATION, POLICY >::
tensorAbstractionSubscriptSliceKernel( ArraySliceT< VALUE_TYPE const, PERMUTATION > const a,
                                       ArraySliceT< VALUE_TYPE const, PERMUTATION > const b,
                                       ArraySliceT< VALUE_TYPE, PERMUTATION > const c )
{ RAJA_OUTER_LOOP( a.size( 0 ), R2TensorMultiplySubscript( a[ i ], b[ i ], c[ i ] ); ); }

template< typename PERMUTATION, typename POLICY >
void ArrayOfR2TensorsRAJA< PERMUTATION, POLICY >::
RAJAViewKernel( RajaView< VALUE_TYPE const, PERMUTATION > const & a,
                RajaView< VALUE_TYPE const, PERMUTATION > const & b,
                RajaView< VALUE_TYPE, PERMUTATION > const & c )
{ RAJA_KERNEL( getRAJAViewLayout( a ).sizes[ 0 ], a( i, j, l ), b( i, l, k ), c( i, j, k ) ); }

template< typename POLICY >
void pointerRajaHelper( RAJA::PERM_IJK,
                        INDEX_TYPE const N,
                        VALUE_TYPE const * const LVARRAY_RESTRICT a,
                        VALUE_TYPE const * const LVARRAY_RESTRICT b,
                        VALUE_TYPE * const LVARRAY_RESTRICT c )
{
  RAJA_KERNEL( N,
               a[ ACCESS_IJK( N, 3, 3, i, j, l ) ],
               b[ ACCESS_IJK( N, 3, 3, i, l, k ) ],
               c[ ACCESS_IJK( N, 3, 3, i, j, k ) ] );
}


template< typename POLICY >
void pointerRajaHelper( RAJA::PERM_KJI,
                        INDEX_TYPE const N,
                        VALUE_TYPE const * const LVARRAY_RESTRICT a,
                        VALUE_TYPE const * const LVARRAY_RESTRICT b,
                        VALUE_TYPE * const LVARRAY_RESTRICT c )
{
  RAJA_KERNEL( N,
               a[ ACCESS_KJI( N, 3, 3, i, j, l ) ],
               b[ ACCESS_KJI( N, 3, 3, i, l, k ) ],
               c[ ACCESS_KJI( N, 3, 3, i, j, k ) ] );
}


template< typename PERMUTATION, typename POLICY >
void ArrayOfR2TensorsRAJA< PERMUTATION, POLICY >::
pointerKernel( INDEX_TYPE const N,
               VALUE_TYPE const * const LVARRAY_RESTRICT a,
               VALUE_TYPE const * const LVARRAY_RESTRICT b,
               VALUE_TYPE * const LVARRAY_RESTRICT c )
{ return pointerRajaHelper< POLICY >( PERMUTATION {}, N, a, b, c ); }


template class ArrayOfR2TensorsNative< RAJA::PERM_IJK >;
template class ArrayOfR2TensorsNative< RAJA::PERM_KJI >;

template class ArrayOfR2TensorsRAJA< RAJA::PERM_IJK, serialPolicy >;
template class ArrayOfR2TensorsRAJA< RAJA::PERM_KJI, serialPolicy >;

#if defined(RAJA_ENABLE_OPENMP)
template class ArrayOfR2TensorsRAJA< RAJA::PERM_IJK, parallelHostPolicy >;
template class ArrayOfR2TensorsRAJA< RAJA::PERM_KJI, parallelHostPolicy >;
#endif

#if defined(LVARRAY_USE_CUDA) && defined(LVARRAY_USE_CHAI)
template class ArrayOfR2TensorsRAJA< RAJA::PERM_IJK, RAJA::cuda_exec< THREADS_PER_BLOCK > >;
template class ArrayOfR2TensorsRAJA< RAJA::PERM_KJI, RAJA::cuda_exec< THREADS_PER_BLOCK > >;
#endif


#undef OUTER_LOOP
#undef RAJA_OUTER_LOOP
#undef INNER_LOOP
#undef KERNEL
#undef RAJA_KERNEL

} // namespace benchmarking
} // namespace LvArray
