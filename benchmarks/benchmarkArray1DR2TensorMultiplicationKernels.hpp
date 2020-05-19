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
    benchmark::ClobberMemory(); \
  } \


template< typename PERMUTATION >
class ArrayOfR2TensorsNative
{
public:
  ArrayOfR2TensorsNative( ::benchmark::State & state,
                          char const * const callingFunction,
                          ResultsMap< VALUE_TYPE, 1 > & results ):
    m_state( state ),
    m_callingFunction( callingFunction ),
    m_results( results )
  {
    INDEX_TYPE const N = m_state.range( 0 );

    m_a.resizeWithoutInitializationOrDestruction( N, 3, 3 );
    m_b.resizeWithoutInitializationOrDestruction( N, 3, 3 );
    m_c.resize( N, 3, 3 );

    int iter = 0;
    initialize( m_a, iter );
    initialize( m_b, iter );
  }

  ~ArrayOfR2TensorsNative()
  {
    m_state.counters[ "OPS" ] = ::benchmark::Counter( 3 * m_a.size(), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000 );
    VALUE_TYPE const result = reduce( m_c ) / INDEX_TYPE( m_state.iterations() );
    registerResult( m_results, { m_state.range( 0 ) }, result, m_callingFunction );
  }

  void fortranArray() const
  { TIMING_LOOP( fortranArrayKernel( m_a, m_b, m_c ) ); }

  void fortranView() const
  {
    ArrayView< VALUE_TYPE const, PERMUTATION > const & a = m_a;
    ArrayView< VALUE_TYPE const, PERMUTATION > const & b = m_b;
    ArrayView< VALUE_TYPE, PERMUTATION > const & c = m_c;
    TIMING_LOOP( fortranViewKernel( a, b, c ) );
  }

  void fortranSlice() const
  {
    ArraySlice< VALUE_TYPE const, PERMUTATION > const a = m_a;
    ArraySlice< VALUE_TYPE const, PERMUTATION > const b = m_b;
    ArraySlice< VALUE_TYPE, PERMUTATION > const c = m_c;
    TIMING_LOOP( fortranSliceKernel( a, b, c ) );
  }

  void subscriptArray() const
  { TIMING_LOOP( subscriptArrayKernel( m_a, m_b, m_c ) ); }

  void subscriptView() const
  {
    ArrayView< VALUE_TYPE const, PERMUTATION > const & a = m_a;
    ArrayView< VALUE_TYPE const, PERMUTATION > const & b = m_b;
    ArrayView< VALUE_TYPE, PERMUTATION > const & c = m_c;
    TIMING_LOOP( subscriptViewKernel( a, b, c ) );
  }

  void subscriptSlice() const
  {
    ArraySlice< VALUE_TYPE const, PERMUTATION > const a = m_a;
    ArraySlice< VALUE_TYPE const, PERMUTATION > const b = m_b;
    ArraySlice< VALUE_TYPE, PERMUTATION > const c = m_c;
    TIMING_LOOP( subscriptSliceKernel( a, b, c ) );
  }

  void tensorAbstractionFortranArray() const
  { TIMING_LOOP( tensorAbstractionFortranArrayKernel( m_a, m_b, m_c ) ); }

  void tensorAbstractionFortranView() const
  {
    ArrayView< VALUE_TYPE const, PERMUTATION > const & a = m_a;
    ArrayView< VALUE_TYPE const, PERMUTATION > const & b = m_b;
    ArrayView< VALUE_TYPE, PERMUTATION > const & c = m_c;
    TIMING_LOOP( tensorAbstractionFortranViewKernel( a, b, c ) );
  }

  void tensorAbstractionFortranSlice() const
  {
    ArraySlice< VALUE_TYPE const, PERMUTATION > const a = m_a;
    ArraySlice< VALUE_TYPE const, PERMUTATION > const b = m_b;
    ArraySlice< VALUE_TYPE, PERMUTATION > const c = m_c;
    TIMING_LOOP( tensorAbstractionFortranSliceKernel( a, b, c ) );
  }

  void tensorAbstractionSubscriptArray() const
  { TIMING_LOOP( tensorAbstractionSubscriptArrayKernel( m_a, m_b, m_c ) ); }

  void tensorAbstractionSubscriptView() const
  {
    ArrayView< VALUE_TYPE const, PERMUTATION > const & a = m_a;
    ArrayView< VALUE_TYPE const, PERMUTATION > const & b = m_b;
    ArrayView< VALUE_TYPE, PERMUTATION > const & c = m_c;
    TIMING_LOOP( tensorAbstractionSubscriptViewKernel( a, b, c ) );
  }

  void tensorAbstractionSubscriptSlice() const
  {
    ArraySlice< VALUE_TYPE const, PERMUTATION > const a = m_a;
    ArraySlice< VALUE_TYPE const, PERMUTATION > const b = m_b;
    ArraySlice< VALUE_TYPE, PERMUTATION > const c = m_c;
    TIMING_LOOP( tensorAbstractionSubscriptSliceKernel( a, b, c ) );
  }

  void RAJAView() const
  {
    RajaView< VALUE_TYPE const, PERMUTATION > const a = makeRajaView( m_a );
    RajaView< VALUE_TYPE const, PERMUTATION > const b = makeRajaView( m_b );
    RajaView< VALUE_TYPE, PERMUTATION > const c = makeRajaView( m_c );
    TIMING_LOOP( RAJAViewKernel( a, b, c ) );
  }

  void pointer() const
  {
    VALUE_TYPE const * const LVARRAY_RESTRICT a = m_a.data();
    VALUE_TYPE const * const LVARRAY_RESTRICT b = m_b.data();
    VALUE_TYPE * const LVARRAY_RESTRICT c = m_c.data();
    TIMING_LOOP( pointerKernel( m_a.size( 0 ), a, b, c ) );
  }

protected:
  ::benchmark::State & m_state;
  std::string const m_callingFunction;
  ResultsMap< VALUE_TYPE, 1 > & m_results;

  Array< VALUE_TYPE, PERMUTATION > m_a;
  Array< VALUE_TYPE, PERMUTATION > m_b;
  Array< VALUE_TYPE, PERMUTATION > m_c;

private:

  static void fortranArrayKernel( Array< VALUE_TYPE, PERMUTATION > const & a,
                                  Array< VALUE_TYPE, PERMUTATION > const & b,
                                  Array< VALUE_TYPE, PERMUTATION > const & c );

  static void fortranViewKernel( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
                                 ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
                                 ArrayView< VALUE_TYPE, PERMUTATION > const & c );

  static void fortranSliceKernel( ArraySlice< VALUE_TYPE const, PERMUTATION > const a,
                                  ArraySlice< VALUE_TYPE const, PERMUTATION > const b,
                                  ArraySlice< VALUE_TYPE, PERMUTATION > const c );

  static void subscriptArrayKernel( Array< VALUE_TYPE, PERMUTATION > const & a,
                                    Array< VALUE_TYPE, PERMUTATION > const & b,
                                    Array< VALUE_TYPE, PERMUTATION > const & c );

  static void subscriptViewKernel( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
                                   ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
                                   ArrayView< VALUE_TYPE, PERMUTATION > const & c );

  static void subscriptSliceKernel( ArraySlice< VALUE_TYPE const, PERMUTATION > const a,
                                    ArraySlice< VALUE_TYPE const, PERMUTATION > const b,
                                    ArraySlice< VALUE_TYPE, PERMUTATION > const c );

  static void tensorAbstractionFortranArrayKernel( Array< VALUE_TYPE, PERMUTATION > const & a,
                                                   Array< VALUE_TYPE, PERMUTATION > const & b,
                                                   Array< VALUE_TYPE, PERMUTATION > const & c );

  static void tensorAbstractionFortranViewKernel( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
                                                  ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
                                                  ArrayView< VALUE_TYPE, PERMUTATION > const & c );

  static void tensorAbstractionFortranSliceKernel( ArraySlice< VALUE_TYPE const, PERMUTATION > const a,
                                                   ArraySlice< VALUE_TYPE const, PERMUTATION > const b,
                                                   ArraySlice< VALUE_TYPE, PERMUTATION > const c );

  static void tensorAbstractionSubscriptArrayKernel( Array< VALUE_TYPE, PERMUTATION > const & a,
                                                     Array< VALUE_TYPE, PERMUTATION > const & b,
                                                     Array< VALUE_TYPE, PERMUTATION > const & c );

  static void tensorAbstractionSubscriptViewKernel( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
                                                    ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
                                                    ArrayView< VALUE_TYPE, PERMUTATION > const & c );

  static void tensorAbstractionSubscriptSliceKernel( ArraySlice< VALUE_TYPE const, PERMUTATION > const a,
                                                     ArraySlice< VALUE_TYPE const, PERMUTATION > const b,
                                                     ArraySlice< VALUE_TYPE, PERMUTATION > const c );

  static void RAJAViewKernel( RajaView< VALUE_TYPE const, PERMUTATION > const & a,
                              RajaView< VALUE_TYPE const, PERMUTATION > const & b,
                              RajaView< VALUE_TYPE, PERMUTATION > const & c );

  static void pointerKernel( INDEX_TYPE const N,
                             VALUE_TYPE const * const LVARRAY_RESTRICT a,
                             VALUE_TYPE const * const LVARRAY_RESTRICT b,
                             VALUE_TYPE * const LVARRAY_RESTRICT c );

};

template< typename PERMUTATION, typename POLICY >
class ArrayOfR2TensorsRAJA : private ArrayOfR2TensorsNative< PERMUTATION >
{
public:
  using ArrayOfR2TensorsNative< PERMUTATION >::m_a;
  using ArrayOfR2TensorsNative< PERMUTATION >::m_b;
  using ArrayOfR2TensorsNative< PERMUTATION >::m_c;

  ArrayOfR2TensorsRAJA( ::benchmark::State & state,
                        char const * const callingFunction,
                        ResultsMap< VALUE_TYPE, 1 > & results ):
    ArrayOfR2TensorsNative< PERMUTATION >( state, callingFunction, results )
  {
    this->m_a.move( RAJAHelper< POLICY >::space, false );
    this->m_b.move( RAJAHelper< POLICY >::space, false );
    this->m_c.move( RAJAHelper< POLICY >::space );
  }

  ~ArrayOfR2TensorsRAJA()
  { this->m_c.move( MemorySpace::CPU ); }

  void fortranView() const
  {
    ArrayView< VALUE_TYPE const, PERMUTATION > const & a = this->m_a;
    ArrayView< VALUE_TYPE const, PERMUTATION > const & b = this->m_b;
    ArrayView< VALUE_TYPE, PERMUTATION > const & c = this->m_c;
    TIMING_LOOP( fortranViewKernel( a, b, c ) );
  }

  void fortranSlice() const
  {
    ArraySlice< VALUE_TYPE const, PERMUTATION > const a = this->m_a;
    ArraySlice< VALUE_TYPE const, PERMUTATION > const b = this->m_b;
    ArraySlice< VALUE_TYPE, PERMUTATION > const c = this->m_c;
    TIMING_LOOP( fortranSliceKernel( a, b, c ) );
  }

  void subscriptView() const
  {
    ArrayView< VALUE_TYPE const, PERMUTATION > const & a = this->m_a;
    ArrayView< VALUE_TYPE const, PERMUTATION > const & b = this->m_b;
    ArrayView< VALUE_TYPE, PERMUTATION > const & c = this->m_c;
    TIMING_LOOP( subscriptViewKernel( a, b, c ) );
  }

  void subscriptSlice() const
  {
    ArraySlice< VALUE_TYPE const, PERMUTATION > const a = this->m_a;
    ArraySlice< VALUE_TYPE const, PERMUTATION > const b = this->m_b;
    ArraySlice< VALUE_TYPE, PERMUTATION > const c = this->m_c;
    TIMING_LOOP( subscriptSliceKernel( a, b, c ) );
  }

  void tensorAbstractionFortranView() const
  {
    ArrayView< VALUE_TYPE const, PERMUTATION > const & a = this->m_a;
    ArrayView< VALUE_TYPE const, PERMUTATION > const & b = this->m_b;
    ArrayView< VALUE_TYPE, PERMUTATION > const & c = this->m_c;
    TIMING_LOOP( tensorAbstractionFortranViewKernel( a, b, c ) );
  }

  void tensorAbstractionFortranSlice() const
  {
    ArraySlice< VALUE_TYPE const, PERMUTATION > const a = this->m_a;
    ArraySlice< VALUE_TYPE const, PERMUTATION > const b = this->m_b;
    ArraySlice< VALUE_TYPE, PERMUTATION > const c = this->m_c;
    TIMING_LOOP( tensorAbstractionFortranSliceKernel( a, b, c ) );
  }

  void tensorAbstractionSubscriptView() const
  {
    ArrayView< VALUE_TYPE const, PERMUTATION > const & a = this->m_a;
    ArrayView< VALUE_TYPE const, PERMUTATION > const & b = this->m_b;
    ArrayView< VALUE_TYPE, PERMUTATION > const & c = this->m_c;
    TIMING_LOOP( tensorAbstractionSubscriptViewKernel( a, b, c ) );
  }

  void tensorAbstractionSubscriptSlice() const
  {
    ArraySlice< VALUE_TYPE const, PERMUTATION > const a = this->m_a;
    ArraySlice< VALUE_TYPE const, PERMUTATION > const b = this->m_b;
    ArraySlice< VALUE_TYPE, PERMUTATION > const c = this->m_c;
    TIMING_LOOP( tensorAbstractionSubscriptSliceKernel( a, b, c ) );
  }

  void RAJAView() const
  {
    RajaView< VALUE_TYPE const, PERMUTATION > const a = makeRajaView( this->m_a );
    RajaView< VALUE_TYPE const, PERMUTATION > const b = makeRajaView( this->m_b );
    RajaView< VALUE_TYPE, PERMUTATION > const c = makeRajaView( this->m_c );
    TIMING_LOOP( RAJAViewKernel( a, b, c ) );
  }

  void pointer() const
  {
    VALUE_TYPE const * const LVARRAY_RESTRICT a = this->m_a.data();
    VALUE_TYPE const * const LVARRAY_RESTRICT b = this->m_b.data();
    VALUE_TYPE * const LVARRAY_RESTRICT c = this->m_c.data();
    TIMING_LOOP( pointerKernel( this->m_a.size( 0 ), a, b, c ) );
  }

// Should be private but nvcc demands they're public.
public:
  static void fortranViewKernel( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
                                 ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
                                 ArrayView< VALUE_TYPE, PERMUTATION > const & c );

  static void fortranSliceKernel( ArraySlice< VALUE_TYPE const, PERMUTATION > const a,
                                  ArraySlice< VALUE_TYPE const, PERMUTATION > const b,
                                  ArraySlice< VALUE_TYPE, PERMUTATION > const c );

  static void subscriptViewKernel( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
                                   ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
                                   ArrayView< VALUE_TYPE, PERMUTATION > const & c );

  static void subscriptSliceKernel( ArraySlice< VALUE_TYPE const, PERMUTATION > const a,
                                    ArraySlice< VALUE_TYPE const, PERMUTATION > const b,
                                    ArraySlice< VALUE_TYPE, PERMUTATION > const c );

  static void tensorAbstractionFortranViewKernel( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
                                                  ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
                                                  ArrayView< VALUE_TYPE, PERMUTATION > const & c );

  static void tensorAbstractionFortranSliceKernel( ArraySlice< VALUE_TYPE const, PERMUTATION > const a,
                                                   ArraySlice< VALUE_TYPE const, PERMUTATION > const b,
                                                   ArraySlice< VALUE_TYPE, PERMUTATION > const c );

  static void tensorAbstractionSubscriptViewKernel( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
                                                    ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
                                                    ArrayView< VALUE_TYPE, PERMUTATION > const & c );

  static void tensorAbstractionSubscriptSliceKernel( ArraySlice< VALUE_TYPE const, PERMUTATION > const a,
                                                     ArraySlice< VALUE_TYPE const, PERMUTATION > const b,
                                                     ArraySlice< VALUE_TYPE, PERMUTATION > const c );

  static void RAJAViewKernel( RajaView< VALUE_TYPE const, PERMUTATION > const & a,
                              RajaView< VALUE_TYPE const, PERMUTATION > const & b,
                              RajaView< VALUE_TYPE, PERMUTATION > const & c );

  static void pointerKernel( INDEX_TYPE const N,
                             VALUE_TYPE const * const LVARRAY_RESTRICT a,
                             VALUE_TYPE const * const LVARRAY_RESTRICT b,
                             VALUE_TYPE * const LVARRAY_RESTRICT c );
};

} // namespace benchmarking
} // namespace LvArray
