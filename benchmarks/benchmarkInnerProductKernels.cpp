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

#define INNER_PRODUCT_KERNEL( N, a_i, b_i ) \
  VALUE_TYPE sum = 0; \
  for( INDEX_TYPE i = 0; i < N; ++i ) \
  { \
    sum = sum + a_i * b_i; \
  } \
  return sum

#define INNER_PRODUCT_KERNEL_RAJA( N, a_i, b_i ) \
  RAJA::ReduceSum< typename RAJAHelper< POLICY >::ReducePolicy, VALUE_TYPE > sum( 0 ); \
  forall< POLICY >( N, [sum, a, b] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i ) \
  { \
    sum += a_i * b_i; \
  } ); \
  return sum.get()

VALUE_TYPE InnerProductNative::
  fortranArrayKernel( Array< VALUE_TYPE, RAJA::PERM_I > const & a,
                      Array< VALUE_TYPE, RAJA::PERM_I > const & b )
{ INNER_PRODUCT_KERNEL( a.size(), a( i ), b( i ) ); }

VALUE_TYPE InnerProductNative::
  fortranViewKernel( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                     ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b )
{ INNER_PRODUCT_KERNEL( a.size(), a( i ), b( i ) ); }

VALUE_TYPE InnerProductNative::
  fortranSliceKernel( ArraySlice< VALUE_TYPE const, RAJA::PERM_I > const a,
                      ArraySlice< VALUE_TYPE const, RAJA::PERM_I > const b )
{ INNER_PRODUCT_KERNEL( a.size(), a( i ), b( i ) ); }

VALUE_TYPE InnerProductNative::
  subscriptArrayKernel( Array< VALUE_TYPE, RAJA::PERM_I > const & a,
                        Array< VALUE_TYPE, RAJA::PERM_I > const & b )
{ INNER_PRODUCT_KERNEL( a.size(), a[ i ], b[ i ] ); }

VALUE_TYPE InnerProductNative::
  subscriptViewKernel( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                       ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b )
{ INNER_PRODUCT_KERNEL( a.size(), a[ i ], b[ i ] ); }

VALUE_TYPE InnerProductNative::
  subscriptSliceKernel( ArraySlice< VALUE_TYPE const, RAJA::PERM_I > const a,
                        ArraySlice< VALUE_TYPE const, RAJA::PERM_I > const b )
{ INNER_PRODUCT_KERNEL( a.size(), a[ i ], b[ i ] ); }

VALUE_TYPE InnerProductNative::
  RAJAViewKernel( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                  RajaView< VALUE_TYPE const, RAJA::PERM_I > const & b )
{ INNER_PRODUCT_KERNEL( a.layout.sizes[ 0 ], a( i ), b( i ) ); }

VALUE_TYPE InnerProductNative::
  pointerKernel( INDEX_TYPE const N,
                 VALUE_TYPE const * const LVARRAY_RESTRICT a,
                 VALUE_TYPE const * const LVARRAY_RESTRICT b )
{ INNER_PRODUCT_KERNEL( N, a[ i ], b[ i ] ); }

template< typename POLICY >
VALUE_TYPE InnerProductRAJA< POLICY >::
fortranViewKernel( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                   ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b )
{ INNER_PRODUCT_KERNEL_RAJA( a.size(), a( i ), b( i ) ); }

template< typename POLICY >
VALUE_TYPE InnerProductRAJA< POLICY >::
fortranSliceKernel( ArraySlice< VALUE_TYPE const, RAJA::PERM_I > const a,
                    ArraySlice< VALUE_TYPE const, RAJA::PERM_I > const b )
{ INNER_PRODUCT_KERNEL_RAJA( a.size(), a( i ), b( i ) ); }

template< typename POLICY >
VALUE_TYPE InnerProductRAJA< POLICY >::
subscriptViewKernel( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                     ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b )
{ INNER_PRODUCT_KERNEL_RAJA( a.size(), a[ i ], b[ i ] ); }

template< typename POLICY >
VALUE_TYPE InnerProductRAJA< POLICY >::
subscriptSliceKernel( ArraySlice< VALUE_TYPE const, RAJA::PERM_I > const a,
                      ArraySlice< VALUE_TYPE const, RAJA::PERM_I > const b )
{ INNER_PRODUCT_KERNEL_RAJA( a.size(), a[ i ], b[ i ] ); }

template< typename POLICY >
VALUE_TYPE InnerProductRAJA< POLICY >::
RAJAViewKernel( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                RajaView< VALUE_TYPE const, RAJA::PERM_I > const & b )
{ INNER_PRODUCT_KERNEL_RAJA( a.layout.sizes[ 0 ], a( i ), b( i ) ); }

template< typename POLICY >
VALUE_TYPE InnerProductRAJA< POLICY >::
pointerKernel( INDEX_TYPE const N,
               VALUE_TYPE const * const LVARRAY_RESTRICT a,
               VALUE_TYPE const * const LVARRAY_RESTRICT b )
{ INNER_PRODUCT_KERNEL_RAJA( N, a[ i ], b[ i ] ); }

template class InnerProductRAJA< serialPolicy >;

#if defined(USE_OPENMP)
template class InnerProductRAJA< parallelHostPolicy >;
#endif

#if defined(USE_CUDA) && defined(USE_CHAI)
template class InnerProductRAJA< RAJA::cuda_exec< THREADS_PER_BLOCK > >;
#endif

} // namespace benchmarking
} // namespace LvArray
