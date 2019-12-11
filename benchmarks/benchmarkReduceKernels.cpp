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

#define REDUCE_KERNEL( a_i ) \
  VALUE_TYPE sum = 0; \
  for( INDEX_TYPE i = 0 ; i < N ; ++i ) \
  { \
    sum += a_i; \
  } \
  return sum

#define REDUCE_KERNEL_RAJA( a_i ) \
  RAJA::ReduceSum< typename RAJAHelper< POLICY >::ReducePolicy, VALUE_TYPE > sum( 0 ); \
  RAJA::forall< POLICY >( RAJA::TypedRangeSegment< INDEX_TYPE >( 0, N ), \
                          [sum, a] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i ) { \
    sum += a_i; \
  } \
                          ); \
  return sum.get()


VALUE_TYPE ReduceNative::fortran( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                                  INDEX_TYPE const N )
{ REDUCE_KERNEL( a( i ) ); }

VALUE_TYPE ReduceNative::subscript( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                                    INDEX_TYPE const N )
{ REDUCE_KERNEL( a[ i ] ); }

VALUE_TYPE ReduceNative::rajaView( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                                   INDEX_TYPE const N )
{ REDUCE_KERNEL( a( i ) ); }

VALUE_TYPE ReduceNative::pointer( VALUE_TYPE const * const restrict a,
                                  INDEX_TYPE const N )
{ REDUCE_KERNEL( a[ i ] ); }

template< class POLICY >
VALUE_TYPE ReduceRAJA< POLICY >::fortran( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                                          INDEX_TYPE const N )
{ REDUCE_KERNEL_RAJA( a( i ) ); }

template< class POLICY >
VALUE_TYPE ReduceRAJA< POLICY >::subscript( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                                            INDEX_TYPE const N )
{ REDUCE_KERNEL_RAJA( a[ i ] ); }

template< class POLICY >
VALUE_TYPE ReduceRAJA< POLICY >::rajaView( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                                           INDEX_TYPE const N )
{ REDUCE_KERNEL_RAJA( a( i ) ); }

template< class POLICY >
VALUE_TYPE ReduceRAJA< POLICY >::pointer( VALUE_TYPE const * const restrict a,
                                          INDEX_TYPE const N )
{ REDUCE_KERNEL_RAJA( a[ i ] ); }

template struct ReduceRAJA< serialPolicy >;

#if defined(USE_OPENMP)
template struct ReduceRAJA< parallelHostPolicy >;
#endif

#if defined(USE_CUDA)
template struct ReduceRAJA< RAJA::cuda_exec< THREADS_PER_BLOCK > >;
#endif

} // namespace benchmarking
} // namespace LvArray
