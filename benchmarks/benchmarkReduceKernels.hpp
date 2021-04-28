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

class ReduceNative
{
public:

  ReduceNative( ::benchmark::State & state, char const * const callingFunction, ResultsMap< VALUE_TYPE, 1 > & results ):
    m_state( state ),
    m_callingFunction( callingFunction ),
    m_results( results ),
    m_array( state.range( 0 ) ),
    m_sum( 0 )
  {
    int iter = 0;
    initialize( m_array.toSlice(), iter );
  }

  ~ReduceNative()
  {
    registerResult( m_results, { m_array.size() }, m_sum / INDEX_TYPE( m_state.iterations() ), m_callingFunction );
    m_state.counters[ "OPS "] = ::benchmark::Counter( m_array.size(), ::benchmark::Counter::kIsIterationInvariantRate, ::benchmark::Counter::OneK::kIs1000 );
  }

  void fortranArray()
  { TIMING_LOOP( fortranArrayKernel( m_array ) ); }

  void fortranView()
  {
    ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const view = m_array.toViewConst();
    TIMING_LOOP( fortranViewKernel( view ) );
  }

  void fortranSlice()
  {
    ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const slice = m_array.toSliceConst();
    TIMING_LOOP( fortranSliceKernel( slice ) );
  }

  void subscriptArray()
  { TIMING_LOOP( subscriptArrayKernel( m_array ) ); }

  void subscriptView()
  {
    ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const view = m_array.toViewConst();
    TIMING_LOOP( subscriptViewKernel( view ) );
  }

  void subscriptSlice()
  {
    ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const slice = m_array.toSliceConst();
    TIMING_LOOP( subscriptSliceKernel( slice ) );
  }

  void RAJAView()
  {
    RajaView< VALUE_TYPE const, RAJA::PERM_I > const view = makeRajaView( m_array );
    TIMING_LOOP( RAJAViewKernel( view ) );
  }

  void pointer()
  {
    VALUE_TYPE const * const LVARRAY_RESTRICT ptr = m_array.data();
    TIMING_LOOP( pointerKernel( ptr, m_array.size() ) );
  }

protected:

  ::benchmark::State & m_state;
  std::string const m_callingFunction;
  ResultsMap< VALUE_TYPE, 1 > & m_results;
  ArrayT< VALUE_TYPE, RAJA::PERM_I > m_array;
  VALUE_TYPE m_sum = 0;

private:

  static VALUE_TYPE fortranArrayKernel( ArrayT< VALUE_TYPE, RAJA::PERM_I > const & a );

  static VALUE_TYPE fortranViewKernel( ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & a );

  static VALUE_TYPE fortranSliceKernel( ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a );

  static VALUE_TYPE subscriptArrayKernel( ArrayT< VALUE_TYPE, RAJA::PERM_I > const & a );

  static VALUE_TYPE subscriptViewKernel( ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & a );

  static VALUE_TYPE subscriptSliceKernel( ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a );

  static VALUE_TYPE subscriptKernel( ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & a );

  static VALUE_TYPE RAJAViewKernel( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a );

  static VALUE_TYPE pointerKernel( VALUE_TYPE const * const LVARRAY_RESTRICT a,
                                   INDEX_TYPE const N );

};

template< typename POLICY >
class ReduceRAJA : public ReduceNative
{
public:

  ReduceRAJA( ::benchmark::State & state, char const * const callingFunction, ResultsMap< VALUE_TYPE, 1 > & results ):
    ReduceNative( state, callingFunction, results )
  { m_array.move( RAJAHelper< POLICY >::space ); }

  void fortranView()
  {
    ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const view = m_array.toViewConst();
    TIMING_LOOP( fortranViewKernel( view ) );
  }

  void fortranSlice()
  {
    ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const slice = m_array.toSliceConst();
    TIMING_LOOP( fortranSliceKernel( slice ) );
  }

  void subscriptView()
  {
    ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const view = m_array.toViewConst();
    TIMING_LOOP( subscriptViewKernel( view ) );
  }

  void subscriptSlice()
  {
    ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const slice = m_array.toSliceConst();
    TIMING_LOOP( subscriptSliceKernel( slice ) );
  }

  void RAJAView()
  {
    RajaView< VALUE_TYPE const, RAJA::PERM_I > const view = makeRajaView( m_array );
    TIMING_LOOP( RAJAViewKernel( view ) );
  }

  void pointer()
  {
    VALUE_TYPE const * const LVARRAY_RESTRICT ptr = m_array.data();
    TIMING_LOOP( pointerKernel( ptr, m_array.size() ) );
  }

// Should be private but nvcc demands they're public.
public:
  static VALUE_TYPE fortranViewKernel( ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & a );

  static VALUE_TYPE fortranSliceKernel( ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a );

  static VALUE_TYPE subscriptViewKernel( ArrayViewT< VALUE_TYPE const, RAJA::PERM_I > const & a );

  static VALUE_TYPE subscriptSliceKernel( ArraySliceT< VALUE_TYPE const, RAJA::PERM_I > const a );

  static VALUE_TYPE RAJAViewKernel( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a );

  static VALUE_TYPE pointerKernel( VALUE_TYPE const * const LVARRAY_RESTRICT a,
                                   INDEX_TYPE const N );
};

#undef TIMING_LOOP

} // namespace benchmarking
} // namespace LvArray
