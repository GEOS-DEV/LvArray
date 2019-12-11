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
#include "benchmarkInnerProductKernels.hpp"

namespace LvArray
{
namespace benchmarking
{

#define INNER_PRODUCT_KERNEL( a_i, b_i ) \
  VALUE_TYPE sum = 0; \
  for( INDEX_TYPE i = 0 ; i < N ; ++i ) \
  { \
    sum += a_i * b_i; \
  } \
  return sum

#define INNER_PRODUCT_KERNEL_RAJA( a_i, b_i ) \
  RAJA::ReduceSum< typename RAJAHelper< POLICY >::ReducePolicy, VALUE_TYPE > sum( 0 ); \
  RAJA::forall< POLICY >( RAJA::TypedRangeSegment< INDEX_TYPE >( 0, N ), \
                          [sum, a, b] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i ) { \
    sum += a_i * b_i; \
  } \
                          ); \
  return sum.get()

VALUE_TYPE InnerProductNative::fortran( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                                        ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b,
                                        INDEX_TYPE const N )
{ INNER_PRODUCT_KERNEL( a( i ), b( i ) ); }

VALUE_TYPE InnerProductNative::subscript( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                                          ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b,
                                          INDEX_TYPE const N )
{ INNER_PRODUCT_KERNEL( a[ i ], b[ i ] ); }

VALUE_TYPE InnerProductNative::rajaView( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                                         RajaView< VALUE_TYPE const, RAJA::PERM_I > const & b,
                                         INDEX_TYPE const N )
{ INNER_PRODUCT_KERNEL( a( i ), b( i ) ); }

VALUE_TYPE InnerProductNative::pointer( VALUE_TYPE const * const restrict a,
                                        VALUE_TYPE const * const restrict b,
                                        INDEX_TYPE const N )
{ INNER_PRODUCT_KERNEL( a[ i ], b[ i ] ); }

template< class POLICY >
VALUE_TYPE InnerProductRAJA< POLICY >::fortran( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                                                ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b,
                                                INDEX_TYPE const N )
{ INNER_PRODUCT_KERNEL_RAJA( a( i ), b( i ) ); }

template< class POLICY >
VALUE_TYPE InnerProductRAJA< POLICY >::subscript( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                                                  ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b,
                                                  INDEX_TYPE const N )
{ INNER_PRODUCT_KERNEL_RAJA( a[ i ], b[ i ] ); }

template< class POLICY >
VALUE_TYPE InnerProductRAJA< POLICY >::rajaView( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                                                 RajaView< VALUE_TYPE const, RAJA::PERM_I > const & b,
                                                 INDEX_TYPE const N )
{ INNER_PRODUCT_KERNEL_RAJA( a( i ), b( i ) ); }

template< class POLICY >
VALUE_TYPE InnerProductRAJA< POLICY >::pointer( VALUE_TYPE const * const restrict a,
                                                VALUE_TYPE const * const restrict b,
                                                INDEX_TYPE const N )
{ INNER_PRODUCT_KERNEL_RAJA( a[ i ], b[ i ] ); }

template struct InnerProductRAJA< serialPolicy >;

#if defined(USE_OPENMP)
template struct InnerProductRAJA< parallelHostPolicy >;
#endif

#if defined(USE_CUDA)
template struct InnerProductRAJA< RAJA::cuda_exec< THREADS_PER_BLOCK > >;
#endif


} // namespace benchmarking
} // namespace LvArray
