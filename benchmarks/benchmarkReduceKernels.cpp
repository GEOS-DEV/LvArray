/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "benchmarkReduceKernels.hpp"

namespace LvArray
{
namespace benchmarking
{

#define REDUCE_KERNEL( N, a_i ) \
  VALUE_TYPE sum = 0; \
  for( INDEX_TYPE i = 0; i < N; ++i ) \
  { \
    sum += a_i; \
  } \
  return sum

#define REDUCE_KERNEL_RAJA( N, a_i ) \
  RAJA::ReduceSum< typename RAJAHelper< POLICY >::ReducePolicy, VALUE_TYPE > sum( 0 ); \
  forall< POLICY >( N, [sum, a] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i ) \
  { \
    sum += a_i; \
  } ); \
  return sum.get()

VALUE_TYPE ReduceNative::fortranArrayKernel( ArrayT< VALUE_TYPE, RAJA::PERM_I > const & a )
{ REDUCE_KERNEL( a.size(), a( i ) ); }

VALUE_TYPE ReduceNative::fortranViewKernel( ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & a )
{ REDUCE_KERNEL( a.size(), a( i ) ); }

VALUE_TYPE ReduceNative::fortranSliceKernel( ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a )
{ REDUCE_KERNEL( a.size(), a( i ) ); }

VALUE_TYPE ReduceNative::subscriptArrayKernel( ArrayT< VALUE_TYPE, RAJA::PERM_I > const & a )
{ REDUCE_KERNEL( a.size(), a[ i ] ); }

VALUE_TYPE ReduceNative::subscriptViewKernel( ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & a )
{ REDUCE_KERNEL( a.size(), a[ i ] ); }

VALUE_TYPE ReduceNative::subscriptSliceKernel( ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a )
{ REDUCE_KERNEL( a.size(), a[ i ] ); }

VALUE_TYPE ReduceNative::RAJAViewKernel( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a )
{ REDUCE_KERNEL( getRAJAViewLayout( a ).sizes[ 0 ], a( i ) ); }

VALUE_TYPE ReduceNative::pointerKernel( VALUE_TYPE const * const LVARRAY_RESTRICT a,
                                        INDEX_TYPE const N )
{ REDUCE_KERNEL( N, a[ i ] ); }

template< class POLICY >
VALUE_TYPE ReduceRAJA< POLICY >::fortranViewKernel( ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & a )
{ REDUCE_KERNEL_RAJA( a.size(), a( i ) ); }

template< class POLICY >
VALUE_TYPE ReduceRAJA< POLICY >::fortranSliceKernel( ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a )
{ REDUCE_KERNEL_RAJA( a.size(), a( i ) ); }

template< class POLICY >
VALUE_TYPE ReduceRAJA< POLICY >::subscriptViewKernel( ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & a )
{ REDUCE_KERNEL_RAJA( a.size(), a[ i ] ); }

template< class POLICY >
VALUE_TYPE ReduceRAJA< POLICY >::subscriptSliceKernel( ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a )
{ REDUCE_KERNEL_RAJA( a.size(), a[ i ] ); }

template< class POLICY >
VALUE_TYPE ReduceRAJA< POLICY >::RAJAViewKernel( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a )
{ REDUCE_KERNEL_RAJA( getRAJAViewLayout( a ).sizes[ 0 ], a( i ) ); }

template< class POLICY >
VALUE_TYPE ReduceRAJA< POLICY >::pointerKernel( VALUE_TYPE const * const LVARRAY_RESTRICT a,
                                                INDEX_TYPE const N )
{ REDUCE_KERNEL_RAJA( N, a[ i ] ); }

template class ReduceRAJA< serialPolicy >;

#if defined(RAJA_ENABLE_OPENMP)
template class ReduceRAJA< parallelHostPolicy >;
#endif

#if defined(LVARRAY_USE_CUDA) && defined(LVARRAY_USE_CHAI)
template class ReduceRAJA< RAJA::cuda_exec< THREADS_PER_BLOCK > >;
#endif

#undef REDUCE_KERNEL
#undef REDUCE_KERNEL_RAJA

} // namespace benchmarking
} // namespace LvArray
