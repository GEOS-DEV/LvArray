/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

#pragma once

// Source includes
#include "benchmarkHelpers.hpp"

// TPL includes
#include <benchmark/benchmark.h>

namespace LvArray
{
namespace benchmarking
{

using VALUE_TYPE = double;
constexpr unsigned long THREADS_PER_BLOCK = 256;

#define TIMING_LOOP( KERNEL ) \
  for( auto _ : this->m_state ) \
  { \
    LVARRAY_UNUSED_VARIABLE( _ ); \
    KERNEL; \
    ::benchmark::ClobberMemory(); \
  } \

template< typename PERMUTATION >
class MatrixMatrixNative
{
public:
  MatrixMatrixNative( ::benchmark::State & state,
                      char const * const callingFunction,
                      ResultsMap< VALUE_TYPE, 3 > & results ):
    m_state( state ),
    m_callingFunction( callingFunction ),
    m_results( results ),
    m_a( state.range( 0 ), state.range( 1 ) ),
    m_b( m_a.size( 1 ), state.range( 2 ) ),
    m_c( m_a.size( 0 ), m_b.size( 1 ) )
  {
    int iter = 0;
    initialize( m_a.toSlice(), iter );
    initialize( m_b.toSlice(), iter );
  }

  ~MatrixMatrixNative()
  {
    VALUE_TYPE const result = reduce( m_c ) / INDEX_TYPE( m_state.iterations() );
    registerResult( m_results, { m_a.size( 0 ), m_a.size( 1 ), m_b.size( 1 ) }, result, m_callingFunction );
    m_state.counters[ "OPS" ] = ::benchmark::Counter( 2 * m_a.size() * m_b.size(
                                                        1 ), ::benchmark::Counter::kIsIterationInvariantRate, ::benchmark::Counter::OneK::kIs1000 );
  }

  void fortranArray() const
  { TIMING_LOOP( fortranArrayKernel( m_a, m_b, m_c ); ) }

  void fortranView() const
  {
    ArrayViewT< VALUE_TYPE const, PERMUTATION > const a = m_a.toViewConst();
    ArrayViewT< VALUE_TYPE const, PERMUTATION > const b = m_b.toViewConst();
    ArrayViewT< VALUE_TYPE, PERMUTATION > const c = m_c.toView();
    TIMING_LOOP( fortranViewKernel( a, b, c ); )
  }

  void fortranSlice() const
  {
    ArraySliceT< VALUE_TYPE const, PERMUTATION > const a = m_a.toSliceConst();
    ArraySliceT< VALUE_TYPE const, PERMUTATION > const b = m_b.toSliceConst();
    ArraySliceT< VALUE_TYPE, PERMUTATION > const c = m_c.toSlice();
    TIMING_LOOP( fortranSliceKernel( a, b, c ); )
  }

  void subscriptArray() const
  { TIMING_LOOP( subscriptArrayKernel( m_a, m_b, m_c ); ) }

  void subscriptView() const
  {
    ArrayViewT< VALUE_TYPE const, PERMUTATION > const a = m_a.toViewConst();
    ArrayViewT< VALUE_TYPE const, PERMUTATION > const b = m_b.toViewConst();
    ArrayViewT< VALUE_TYPE, PERMUTATION > const c = m_c.toView();
    TIMING_LOOP( subscriptViewKernel( a, b, c ); )
  }

  void subscriptSlice() const
  {
    ArraySliceT< VALUE_TYPE const, PERMUTATION > const a = m_a.toSliceConst();
    ArraySliceT< VALUE_TYPE const, PERMUTATION > const b = m_b.toSliceConst();
    ArraySliceT< VALUE_TYPE, PERMUTATION > const c = m_c.toSlice();
    TIMING_LOOP( subscriptSliceKernel( a, b, c ); )
  }

  void RAJAView() const
  {
    constexpr int NDIM = typeManipulation::getDimension< PERMUTATION >;
    constexpr std::array< camp::idx_t, NDIM > const permutation = RAJA::as_array< PERMUTATION >::get();

    std::array< INDEX_TYPE, NDIM > a_sizes;
    for( int i = 0; i < NDIM; ++i )
    {
      a_sizes[ i ] = m_a.dims()[ i ];
    }
    RajaView< VALUE_TYPE const, PERMUTATION > const a = RajaView< VALUE_TYPE const, PERMUTATION >( m_a.data(), RAJA::make_permuted_layout( a_sizes, permutation ) );

    std::array< INDEX_TYPE, NDIM > b_sizes;
    for( int i = 0; i < NDIM; ++i )
    {
      b_sizes[ i ] = m_b.dims()[ i ];
    }
    RajaView< VALUE_TYPE const, PERMUTATION > const b = RajaView< VALUE_TYPE const, PERMUTATION >( m_b.data(), RAJA::make_permuted_layout( b_sizes, permutation ) );

    std::array< INDEX_TYPE, NDIM > c_sizes;
    for( int i = 0; i < NDIM; ++i )
    {
      c_sizes[ i ] = m_c.dims()[ i ];
    }
    RajaView< VALUE_TYPE, PERMUTATION > const c = RajaView< VALUE_TYPE, PERMUTATION >( m_c.data(), RAJA::make_permuted_layout( c_sizes, permutation ) );
    TIMING_LOOP( RAJAViewKernel( a, b, c ); )
  }

  void pointer() const
  {
    VALUE_TYPE const * const LVARRAY_RESTRICT a = m_a.data();
    VALUE_TYPE const * const LVARRAY_RESTRICT b = m_b.data();
    VALUE_TYPE * const LVARRAY_RESTRICT c = m_c.data();
    TIMING_LOOP( pointerKernel( m_a.size( 0 ), m_a.size( 1 ), m_b.size( 1 ), a, b, c ); )
  }

protected:
  ::benchmark::State & m_state;
  std::string const m_callingFunction;
  ResultsMap< VALUE_TYPE, 3 > & m_results;
  ArrayT< VALUE_TYPE, PERMUTATION > m_a;
  ArrayT< VALUE_TYPE, PERMUTATION > m_b;
  ArrayT< VALUE_TYPE, PERMUTATION > m_c;

private:
  static void fortranArrayKernel( ArrayT< VALUE_TYPE, PERMUTATION > const & a,
                                  ArrayT< VALUE_TYPE, PERMUTATION > const & b,
                                  ArrayT< VALUE_TYPE, PERMUTATION > const & c );

  static void fortranViewKernel( ArrayViewT< VALUE_TYPE const, PERMUTATION > const & a,
                                 ArrayViewT< VALUE_TYPE const, PERMUTATION > const & b,
                                 ArrayViewT< VALUE_TYPE, PERMUTATION > const & c );

  static void fortranSliceKernel( ArraySliceT< VALUE_TYPE const, PERMUTATION > const a,
                                  ArraySliceT< VALUE_TYPE const, PERMUTATION > const b,
                                  ArraySliceT< VALUE_TYPE, PERMUTATION > const c );

  static void subscriptArrayKernel( ArrayT< VALUE_TYPE, PERMUTATION > const & a,
                                    ArrayT< VALUE_TYPE, PERMUTATION > const & b,
                                    ArrayT< VALUE_TYPE, PERMUTATION > const & c );

  static void subscriptViewKernel( ArrayViewT< VALUE_TYPE const, PERMUTATION > const & a,
                                   ArrayViewT< VALUE_TYPE const, PERMUTATION > const & b,
                                   ArrayViewT< VALUE_TYPE, PERMUTATION > const & c );

  static void subscriptSliceKernel( ArraySliceT< VALUE_TYPE const, PERMUTATION > const a,
                                    ArraySliceT< VALUE_TYPE const, PERMUTATION > const b,
                                    ArraySliceT< VALUE_TYPE, PERMUTATION > const c );

  static void RAJAViewKernel( RajaView< VALUE_TYPE const, PERMUTATION > const & a,
                              RajaView< VALUE_TYPE const, PERMUTATION > const & b,
                              RajaView< VALUE_TYPE, PERMUTATION > const & c );

  static void pointerKernel( INDEX_TYPE const N,
                             INDEX_TYPE const L,
                             INDEX_TYPE const M,
                             VALUE_TYPE const * const LVARRAY_RESTRICT a,
                             VALUE_TYPE const * const LVARRAY_RESTRICT b,
                             VALUE_TYPE * const LVARRAY_RESTRICT c );
};

template< typename PERMUTATION, typename POLICY >
class MatrixMatrixRAJA : public MatrixMatrixNative< PERMUTATION >
{
public:
  MatrixMatrixRAJA( ::benchmark::State & state,
                    char const * const callingFunction,
                    ResultsMap< VALUE_TYPE, 3 > & results ):
    MatrixMatrixNative< PERMUTATION >( state, callingFunction, results )
  {
    this->m_a.move( RAJAHelper< POLICY >::space, false );
    this->m_b.move( RAJAHelper< POLICY >::space, false );
    this->m_c.move( RAJAHelper< POLICY >::space, true );
  }

  ~MatrixMatrixRAJA()
  { this->m_c.move( MemorySpace::host, false ); }

  void fortranView() const
  {
    ArrayViewT< VALUE_TYPE const, PERMUTATION > const a = this->m_a.toViewConst();
    ArrayViewT< VALUE_TYPE const, PERMUTATION > const b = this->m_b.toViewConst();
    ArrayViewT< VALUE_TYPE, PERMUTATION > const c = this->m_c.toView();
    TIMING_LOOP( fortranViewKernel( a, b, c ); )
  }

  void fortranSlice() const
  {
    ArraySliceT< VALUE_TYPE const, PERMUTATION > const a = this->m_a.toSliceConst();
    ArraySliceT< VALUE_TYPE const, PERMUTATION > const b = this->m_b.toSliceConst();
    ArraySliceT< VALUE_TYPE, PERMUTATION > const c = this->m_c.toSlice();
    TIMING_LOOP( fortranSliceKernel( a, b, c ); )
  }

  void subscriptView() const
  {
    ArrayViewT< VALUE_TYPE const, PERMUTATION > const a = this->m_a.toViewConst();
    ArrayViewT< VALUE_TYPE const, PERMUTATION > const b = this->m_b.toViewConst();
    ArrayViewT< VALUE_TYPE, PERMUTATION > const c = this->m_c.toView();
    TIMING_LOOP( subscriptViewKernel( a, b, c ); )
  }

  void subscriptSlice() const
  {
    ArraySliceT< VALUE_TYPE const, PERMUTATION > const a = this->m_a.toSliceConst();
    ArraySliceT< VALUE_TYPE const, PERMUTATION > const b = this->m_b.toSliceConst();
    ArraySliceT< VALUE_TYPE, PERMUTATION > const c = this->m_c.toSlice();
    TIMING_LOOP( subscriptSliceKernel( a, b, c ); )
  }

  void RAJAView() const
  {
    constexpr int NDIM = typeManipulation::getDimension< PERMUTATION >;
    constexpr std::array< camp::idx_t, NDIM > const permutation = RAJA::as_array< PERMUTATION >::get();

    std::array< INDEX_TYPE, NDIM > a_sizes;
    for( int i = 0; i < NDIM; ++i )
    {
      a_sizes[ i ] = this->m_a.dims()[ i ];
    }
    RajaView< VALUE_TYPE const, PERMUTATION > const a = RajaView< VALUE_TYPE const, PERMUTATION >( this->m_a.data(), RAJA::make_permuted_layout( a_sizes, permutation ) );

    std::array< INDEX_TYPE, NDIM > b_sizes;
    for( int i = 0; i < NDIM; ++i )
    {
      b_sizes[ i ] = this->m_b.dims()[ i ];
    }
    RajaView< VALUE_TYPE const, PERMUTATION > const b = RajaView< VALUE_TYPE const, PERMUTATION >( this->m_b.data(), RAJA::make_permuted_layout( b_sizes, permutation ) );

    std::array< INDEX_TYPE, NDIM > c_sizes;
    for( int i = 0; i < NDIM; ++i )
    {
      c_sizes[ i ] = this->m_c.dims()[ i ];
    }
    RajaView< VALUE_TYPE, PERMUTATION > const c = RajaView< VALUE_TYPE, PERMUTATION >( this->m_c.data(), RAJA::make_permuted_layout( c_sizes, permutation ) );
    TIMING_LOOP( RAJAViewKernel( a, b, c ); )
  }

  void pointer() const
  {
    VALUE_TYPE const * const LVARRAY_RESTRICT a = this->m_a.data();
    VALUE_TYPE const * const LVARRAY_RESTRICT b = this->m_b.data();
    VALUE_TYPE * const LVARRAY_RESTRICT c = this->m_c.data();
    TIMING_LOOP( pointerKernel( this->m_a.size( 0 ), this->m_a.size( 1 ), this->m_b.size( 1 ), a, b, c ); )
  }

// Should be private but nvcc demands they're public.
public:
  static void fortranViewKernel( ArrayViewT< VALUE_TYPE const, PERMUTATION > const & a,
                                 ArrayViewT< VALUE_TYPE const, PERMUTATION > const & b,
                                 ArrayViewT< VALUE_TYPE, PERMUTATION > const & c );

  static void fortranSliceKernel( ArraySliceT< VALUE_TYPE const, PERMUTATION > const a,
                                  ArraySliceT< VALUE_TYPE const, PERMUTATION > const b,
                                  ArraySliceT< VALUE_TYPE, PERMUTATION > const c );

  static void subscriptViewKernel( ArrayViewT< VALUE_TYPE const, PERMUTATION > const & a,
                                   ArrayViewT< VALUE_TYPE const, PERMUTATION > const & b,
                                   ArrayViewT< VALUE_TYPE, PERMUTATION > const & c );

  static void subscriptSliceKernel( ArraySliceT< VALUE_TYPE const, PERMUTATION > const a,
                                    ArraySliceT< VALUE_TYPE const, PERMUTATION > const b,
                                    ArraySliceT< VALUE_TYPE, PERMUTATION > const c );

  static void RAJAViewKernel( RajaView< VALUE_TYPE const, PERMUTATION > const & a,
                              RajaView< VALUE_TYPE const, PERMUTATION > const & b,
                              RajaView< VALUE_TYPE, PERMUTATION > const & c );

  static void pointerKernel( INDEX_TYPE const N,
                             INDEX_TYPE const L,
                             INDEX_TYPE const M,
                             VALUE_TYPE const * const LVARRAY_RESTRICT a,
                             VALUE_TYPE const * const LVARRAY_RESTRICT b,
                             VALUE_TYPE * const LVARRAY_RESTRICT c );
};

#undef TIMING_LOOP

} // namespace benchmarking
} // namespace LvArray
