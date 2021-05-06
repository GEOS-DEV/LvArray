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
  for( auto _ : m_state ) \
  { \
    LVARRAY_UNUSED_VARIABLE( _ ); \
    m_sum += KERNEL; \
    ::benchmark::DoNotOptimize( m_sum ); \
    ::benchmark::ClobberMemory(); \
  } \

class InnerProductNative
{
public:

  InnerProductNative( ::benchmark::State & state,
                      char const * const callingFunction,
                      ResultsMap< VALUE_TYPE, 1 > & results ):
    m_state( state ),
    m_callingFunction( callingFunction ),
    m_results( results ),
    m_a( state.range( 0 ) ),
    m_b( state.range( 0 ) ),
    m_sum( 0 )
  {
    int iter = 0;
    initialize( m_a.toSlice(), iter );
    initialize( m_b.toSlice(), iter );
  }

  ~InnerProductNative()
  {
    registerResult( m_results, { m_a.size() }, m_sum / INDEX_TYPE( m_state.iterations() ), m_callingFunction );
    m_state.counters[ "OPS "] = ::benchmark::Counter( 2 * m_a.size(), ::benchmark::Counter::kIsIterationInvariantRate, ::benchmark::Counter::OneK::kIs1000 );
  }

  void fortranArray()
  { TIMING_LOOP( fortranArrayKernel( m_a, m_b ) ); }

  void fortranView()
  {
    ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const a = m_a.toViewConst();
    ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const b = m_b.toViewConst();
    TIMING_LOOP( fortranViewKernel( a, b ) );
  }

  void fortranSlice()
  {
    ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a = m_a.toSliceConst();
    ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const b = m_b.toSliceConst();
    TIMING_LOOP( fortranSliceKernel( a, b ) );
  }

  void subscriptArray()
  { TIMING_LOOP( subscriptArrayKernel( m_a, m_b ) ); }

  void subscriptView()
  {
    ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const a = m_a.toViewConst();
    ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const b = m_b.toViewConst();
    TIMING_LOOP( subscriptViewKernel( a, b ) );
  }

  void subscriptSlice()
  {
    ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a = m_a.toSliceConst();
    ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const b = m_b.toSliceConst();
    TIMING_LOOP( subscriptSliceKernel( a, b ) );
  }

  void RAJAView()
  {
    RajaView< VALUE_TYPE const, RAJA::PERM_I > const a = makeRajaView( m_a );
    RajaView< VALUE_TYPE const, RAJA::PERM_I > const b = makeRajaView( m_b );
    TIMING_LOOP( RAJAViewKernel( a, b ) );
  }

  void pointer()
  {
    VALUE_TYPE const * const LVARRAY_RESTRICT a = m_a.data();
    VALUE_TYPE const * const LVARRAY_RESTRICT b = m_b.data();
    TIMING_LOOP( pointerKernel( m_a.size(), a, b ) );
  }

protected:

  ::benchmark::State & m_state;
  std::string const m_callingFunction;
  ResultsMap< VALUE_TYPE, 1 > & m_results;
  ArrayT< VALUE_TYPE, RAJA::PERM_I > m_a;
  ArrayT< VALUE_TYPE, RAJA::PERM_I > m_b;
  VALUE_TYPE m_sum = 0;

private:
  static VALUE_TYPE fortranArrayKernel( ArrayT< VALUE_TYPE, RAJA::PERM_I > const & a,
                                        ArrayT< VALUE_TYPE, RAJA::PERM_I > const & b );

  static VALUE_TYPE fortranViewKernel( ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & a,
                                       ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & b );

  static VALUE_TYPE fortranSliceKernel( ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a,
                                        ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const b );

  static VALUE_TYPE subscriptArrayKernel( ArrayT< VALUE_TYPE, RAJA::PERM_I > const & a,
                                          ArrayT< VALUE_TYPE, RAJA::PERM_I > const & b );

  static VALUE_TYPE subscriptViewKernel( ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & a,
                                         ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & b );

  static VALUE_TYPE subscriptSliceKernel( ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a,
                                          ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const b );

  static VALUE_TYPE RAJAViewKernel( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                                    RajaView< VALUE_TYPE const, RAJA::PERM_I > const & b );

  static VALUE_TYPE pointerKernel( INDEX_TYPE const N,
                                   VALUE_TYPE const * const LVARRAY_RESTRICT a,
                                   VALUE_TYPE const * const LVARRAY_RESTRICT b );
};

template< typename POLICY >
class InnerProductRAJA : public InnerProductNative
{
public:

  InnerProductRAJA( ::benchmark::State & state,
                    char const * const callingFunction,
                    ResultsMap< VALUE_TYPE, 1 > & results ):
    InnerProductNative( state, callingFunction, results )
  {}

  void fortranView()
  {
    ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const a = m_a.toViewConst();
    ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const b = m_b.toViewConst();
    TIMING_LOOP( fortranViewKernel( a, b ) );
  }

  void fortranSlice()
  {
    ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a = m_a.toSliceConst();
    ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const b = m_b.toSliceConst();
    TIMING_LOOP( fortranSliceKernel( a, b ) );
  }

  void subscriptView()
  {
    ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const a = m_a.toViewConst();
    ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const b = m_b.toViewConst();
    TIMING_LOOP( subscriptViewKernel( a, b ) );
  }

  void subscriptSlice()
  {
    ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a = m_a.toSliceConst();
    ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const b = m_b.toSliceConst();
    TIMING_LOOP( subscriptSliceKernel( a, b ) );
  }

  void RAJAView()
  {
    RajaView< VALUE_TYPE const, RAJA::PERM_I > const a = makeRajaView( m_a );
    RajaView< VALUE_TYPE const, RAJA::PERM_I > const b = makeRajaView( m_b );
    TIMING_LOOP( RAJAViewKernel( a, b ) );
  }

  void pointer()
  {
    VALUE_TYPE const * const LVARRAY_RESTRICT a = m_a.data();
    VALUE_TYPE const * const LVARRAY_RESTRICT b = m_b.data();
    TIMING_LOOP( pointerKernel( m_a.size(), a, b ) );
  }

// Should be private but nvcc demands they're public.
public:
  static VALUE_TYPE fortranViewKernel( ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & a,
                                       ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & b );

  static VALUE_TYPE fortranSliceKernel( ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a,
                                        ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const b );

  static VALUE_TYPE subscriptViewKernel( ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & a,
                                         ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & b );

  static VALUE_TYPE subscriptSliceKernel( ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a,
                                          ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const b );

  static VALUE_TYPE RAJAViewKernel( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                                    RajaView< VALUE_TYPE const, RAJA::PERM_I > const & b );

  static VALUE_TYPE pointerKernel( INDEX_TYPE const N,
                                   VALUE_TYPE const * const LVARRAY_RESTRICT a,
                                   VALUE_TYPE const * const LVARRAY_RESTRICT b );
};

#undef TIMING_LOOP

} // namespace benchmarking
} // namespace LvArray
