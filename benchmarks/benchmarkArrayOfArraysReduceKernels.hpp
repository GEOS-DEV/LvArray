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

class ArrayOfArraysReduce
{
public:

  ArrayOfArraysReduce( ::benchmark::State & state,
                       char const * const callingFunction,
                       ResultsMap< VALUE_TYPE, 1 > & results ):
    m_state( state ),
    m_callingFunction( callingFunction ),
    m_results( results ),
    m_array(),
    m_vector(),
    m_sum( 0 )
  {
    int iter = 0;
    for( std::ptrdiff_t i = 0; i < state.range( 0 ); ++i )
    {
      m_array.appendArray( i + 1 );
      initialize( m_array[ i ], iter );

      iter = 0;
      m_vector.emplace_back( i + 1 );
      for( INDEX_TYPE j = 0; j < i + 1; ++j )
      {
        m_vector[ i ][ j ] = m_array[ i ][ j ];
      }
    }
  }

  ~ArrayOfArraysReduce()
  {
    registerResult( m_results, { m_array.size() }, m_sum / INDEX_TYPE( m_state.iterations() ), m_callingFunction );
    INDEX_TYPE const totalSize = ( m_array.size() * ( m_array.size() + 1 ) ) / 2;
    m_state.counters[ "OPS "] = ::benchmark::Counter( totalSize,
                                                      ::benchmark::Counter::kIsIterationInvariantRate,
                                                      ::benchmark::Counter::OneK::kIs1000 );
  }

  void reduceFortranArray()
  { TIMING_LOOP( reduceFortranArray( m_array ) ); }

  void reduceSubscriptArray()
  { TIMING_LOOP( reduceSubscriptArray( m_array ) ); }

  void reduceFortranView()
  {
    ArrayOfArraysViewT< VALUE_TYPE const, true > const view = m_array.toViewConst();
    TIMING_LOOP( reduceFortranView( view ) );
  }

  void reduceSubscriptView()
  {
    ArrayOfArraysViewT< VALUE_TYPE const, true > const view = m_array.toViewConst();
    TIMING_LOOP( reduceSubscriptView( view ) );
  }

  void reduceVector()
  { TIMING_LOOP( reduceVector( m_vector ) ); }

private:
  static VALUE_TYPE reduceFortranArray( ArrayOfArraysT< VALUE_TYPE > const & a );
  static VALUE_TYPE reduceSubscriptArray( ArrayOfArraysT< VALUE_TYPE > const & a );
  static VALUE_TYPE reduceFortranView( ArrayOfArraysViewT< VALUE_TYPE const, true > const & a );
  static VALUE_TYPE reduceSubscriptView( ArrayOfArraysViewT< VALUE_TYPE const, true > const & a );

  static VALUE_TYPE reduceVector( std::vector< std::vector< VALUE_TYPE > > const & a );

  ::benchmark::State & m_state;
  std::string const m_callingFunction;
  ResultsMap< VALUE_TYPE, 1 > & m_results;
  ArrayOfArraysT< VALUE_TYPE > m_array;
  std::vector< std::vector< VALUE_TYPE > > m_vector;
  VALUE_TYPE m_sum = 0;
};

#undef TIMING_LOOP

} // namespace benchmarking
} // namespace LvArray
