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

VALUE_TYPE ReduceNative::fortranArrayKernel( Array< VALUE_TYPE, RAJA::PERM_I > const & a )
{ REDUCE_KERNEL( a.size(), a( i ) ); }

VALUE_TYPE ReduceNative::fortranViewKernel( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a )
{ REDUCE_KERNEL( a.size(), a( i ) ); }

VALUE_TYPE ReduceNative::fortranSliceKernel( ArraySlice< VALUE_TYPE const, RAJA::PERM_I > const a )
{ REDUCE_KERNEL( a.size(), a( i ) ); }

VALUE_TYPE ReduceNative::subscriptArrayKernel( Array< VALUE_TYPE, RAJA::PERM_I > const & a )
{ REDUCE_KERNEL( a.size(), a[ i ] ); }

VALUE_TYPE ReduceNative::subscriptViewKernel( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a )
{ REDUCE_KERNEL( a.size(), a[ i ] ); }

VALUE_TYPE ReduceNative::subscriptSliceKernel( ArraySlice< VALUE_TYPE const, RAJA::PERM_I > const a )
{ REDUCE_KERNEL( a.size(), a[ i ] ); }

VALUE_TYPE ReduceNative::RAJAViewKernel( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a )
{ REDUCE_KERNEL( a.layout.sizes[ 0 ], a( i ) ); }

VALUE_TYPE ReduceNative::pointerKernel( VALUE_TYPE const * const LVARRAY_RESTRICT a,
                                        INDEX_TYPE const N )
{ REDUCE_KERNEL( N, a[ i ] ); }

template< class POLICY >
VALUE_TYPE ReduceRAJA< POLICY >::fortranViewKernel( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a )
{ REDUCE_KERNEL_RAJA( a.size(), a( i ) ); }

template< class POLICY >
VALUE_TYPE ReduceRAJA< POLICY >::fortranSliceKernel( ArraySlice< VALUE_TYPE const, RAJA::PERM_I > const a )
{ REDUCE_KERNEL_RAJA( a.size(), a( i ) ); }

template< class POLICY >
VALUE_TYPE ReduceRAJA< POLICY >::subscriptViewKernel( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a )
{ REDUCE_KERNEL_RAJA( a.size(), a[ i ] ); }

template< class POLICY >
VALUE_TYPE ReduceRAJA< POLICY >::subscriptSliceKernel( ArraySlice< VALUE_TYPE const, RAJA::PERM_I > const a )
{ REDUCE_KERNEL_RAJA( a.size(), a[ i ] ); }

template< class POLICY >
VALUE_TYPE ReduceRAJA< POLICY >::RAJAViewKernel( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a )
{ REDUCE_KERNEL_RAJA( a.layout.sizes[ 0 ], a( i ) ); }

template< class POLICY >
VALUE_TYPE ReduceRAJA< POLICY >::pointerKernel( VALUE_TYPE const * const LVARRAY_RESTRICT a,
                                                INDEX_TYPE const N )
{ REDUCE_KERNEL_RAJA( N, a[ i ] ); }

template class ReduceRAJA< serialPolicy >;

#if defined(USE_OPENMP)
template class ReduceRAJA< parallelHostPolicy >;
#endif

#if defined(USE_CUDA)
template class ReduceRAJA< RAJA::cuda_exec< THREADS_PER_BLOCK > >;
#endif

#undef REDUCE_KERNEL
#undef REDUCE_KERNEL_RAJA

} // namespace benchmarking
} // namespace LvArray
