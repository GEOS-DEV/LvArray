/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
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
  fortranArrayKernel( ArrayT< VALUE_TYPE, RAJA::PERM_I > const & a,
                      ArrayT< VALUE_TYPE, RAJA::PERM_I > const & b )
{ INNER_PRODUCT_KERNEL( a.size(), a( i ), b( i ) ); }

VALUE_TYPE InnerProductNative::
  fortranViewKernel( ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & a,
                     ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & b )
{ INNER_PRODUCT_KERNEL( a.size(), a( i ), b( i ) ); }

VALUE_TYPE InnerProductNative::
  fortranSliceKernel( ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a,
                      ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const b )
{ INNER_PRODUCT_KERNEL( a.size(), a( i ), b( i ) ); }

VALUE_TYPE InnerProductNative::
  subscriptArrayKernel( ArrayT< VALUE_TYPE, RAJA::PERM_I > const & a,
                        ArrayT< VALUE_TYPE, RAJA::PERM_I > const & b )
{ INNER_PRODUCT_KERNEL( a.size(), a[ i ], b[ i ] ); }

VALUE_TYPE InnerProductNative::
  subscriptViewKernel( ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & a,
                       ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & b )
{ INNER_PRODUCT_KERNEL( a.size(), a[ i ], b[ i ] ); }

VALUE_TYPE InnerProductNative::
  subscriptSliceKernel( ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a,
                        ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const b )
{ INNER_PRODUCT_KERNEL( a.size(), a[ i ], b[ i ] ); }

VALUE_TYPE InnerProductNative::
  RAJAViewKernel( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                  RajaView< VALUE_TYPE const, RAJA::PERM_I > const & b )
{ INNER_PRODUCT_KERNEL( getRAJAViewLayout( a ).sizes[ 0 ], a( i ), b( i ) ); }

VALUE_TYPE InnerProductNative::
  pointerKernel( INDEX_TYPE const N,
                 VALUE_TYPE const * const LVARRAY_RESTRICT a,
                 VALUE_TYPE const * const LVARRAY_RESTRICT b )
{ INNER_PRODUCT_KERNEL( N, a[ i ], b[ i ] ); }

template< typename POLICY >
VALUE_TYPE InnerProductRAJA< POLICY >::
fortranViewKernel( ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & a,
                   ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & b )
{ INNER_PRODUCT_KERNEL_RAJA( a.size(), a( i ), b( i ) ); }

template< typename POLICY >
VALUE_TYPE InnerProductRAJA< POLICY >::
fortranSliceKernel( ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a,
                    ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const b )
{ INNER_PRODUCT_KERNEL_RAJA( a.size(), a( i ), b( i ) ); }

template< typename POLICY >
VALUE_TYPE InnerProductRAJA< POLICY >::
subscriptViewKernel( ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & a,
                     ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & b )
{ INNER_PRODUCT_KERNEL_RAJA( a.size(), a[ i ], b[ i ] ); }

template< typename POLICY >
VALUE_TYPE InnerProductRAJA< POLICY >::
subscriptSliceKernel( ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a,
                      ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const b )
{ INNER_PRODUCT_KERNEL_RAJA( a.size(), a[ i ], b[ i ] ); }

template< typename POLICY >
VALUE_TYPE InnerProductRAJA< POLICY >::
RAJAViewKernel( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                RajaView< VALUE_TYPE const, RAJA::PERM_I > const & b )
{ INNER_PRODUCT_KERNEL_RAJA( getRAJAViewLayout( a ).sizes[ 0 ], a( i ), b( i ) ); }

template< typename POLICY >
VALUE_TYPE InnerProductRAJA< POLICY >::
pointerKernel( INDEX_TYPE const N,
               VALUE_TYPE const * const LVARRAY_RESTRICT a,
               VALUE_TYPE const * const LVARRAY_RESTRICT b )
{ INNER_PRODUCT_KERNEL_RAJA( N, a[ i ], b[ i ] ); }

template class InnerProductRAJA< serialPolicy >;

#if defined(RAJA_ENABLE_OPENMP)
template class InnerProductRAJA< parallelHostPolicy >;
#endif

#if defined(LVARRAY_USE_CUDA) && defined(LVARRAY_USE_CHAI)
template class InnerProductRAJA< RAJA::cuda_exec< THREADS_PER_BLOCK > >;
#endif

} // namespace benchmarking
} // namespace LvArray
