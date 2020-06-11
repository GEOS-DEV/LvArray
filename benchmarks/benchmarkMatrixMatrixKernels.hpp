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
    initialize( m_a, iter );
    initialize( m_b, iter );
  }

  ~MatrixMatrixNative()
  {
    VALUE_TYPE const result = reduce( m_c ) / INDEX_TYPE( m_state.iterations() );
    registerResult( m_results, { m_a.size( 0 ), m_a.size( 1 ), m_b.size( 1 ) }, result, m_callingFunction );
    m_state.counters[ "OPS "] = ::benchmark::Counter( 2 * m_a.size() * m_b.size(
                                                        1 ), ::benchmark::Counter::kIsIterationInvariantRate, ::benchmark::Counter::OneK::kIs1000 );
  }

  void fortranArray() const
  { TIMING_LOOP( fortranArrayKernel( m_a, m_b, m_c ); ) }

  void fortranView() const
  {
    ArrayView< VALUE_TYPE const, PERMUTATION > const & a = m_a.toViewConst();
    ArrayView< VALUE_TYPE const, PERMUTATION > const & b = m_b.toViewConst();
    ArrayView< VALUE_TYPE, PERMUTATION > const & c = m_c.toView();
    TIMING_LOOP( fortranViewKernel( a, b, c ); )
  }

  void fortranSlice() const
  {
    ArraySlice< VALUE_TYPE const, PERMUTATION > const & a = m_a.toSliceConst();
    ArraySlice< VALUE_TYPE const, PERMUTATION > const & b = m_b.toSliceConst();
    ArraySlice< VALUE_TYPE, PERMUTATION > const & c = m_c.toSlice();
    TIMING_LOOP( fortranSliceKernel( a, b, c ); )
  }

  void subscriptArray() const
  { TIMING_LOOP( subscriptArrayKernel( m_a, m_b, m_c ); ) }

  void subscriptView() const
  {
    ArrayView< VALUE_TYPE const, PERMUTATION > const & a = m_a.toViewConst();
    ArrayView< VALUE_TYPE const, PERMUTATION > const & b = m_b.toViewConst();
    ArrayView< VALUE_TYPE, PERMUTATION > const & c = m_c.toView();
    TIMING_LOOP( subscriptViewKernel( a, b, c ); )
  }

  void subscriptSlice() const
  {
    ArraySlice< VALUE_TYPE const, PERMUTATION > const & a = m_a.toSliceConst();
    ArraySlice< VALUE_TYPE const, PERMUTATION > const & b = m_b.toSliceConst();
    ArraySlice< VALUE_TYPE, PERMUTATION > const & c = m_c.toSlice();
    TIMING_LOOP( subscriptSliceKernel( a, b, c ); )
  }

  void RAJAView() const
  {
    RajaView< VALUE_TYPE const, PERMUTATION > const a = makeRajaView( m_a );
    RajaView< VALUE_TYPE const, PERMUTATION > const b = makeRajaView( m_b );
    RajaView< VALUE_TYPE, PERMUTATION > const c = makeRajaView( m_c );
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
  { this->m_c.move( RAJAHelper< POLICY >::space, false ); }

  void fortranView() const
  {
    ArrayView< VALUE_TYPE const, PERMUTATION > const & a = this->m_a.toViewConst();
    ArrayView< VALUE_TYPE const, PERMUTATION > const & b = this->m_b.toViewConst();
    ArrayView< VALUE_TYPE, PERMUTATION > const & c = this->m_c.toView();
    TIMING_LOOP( fortranViewKernel( a, b, c ); )
  }

  void fortranSlice() const
  {
    ArraySlice< VALUE_TYPE const, PERMUTATION > const & a = this->m_a.toSliceConst();
    ArraySlice< VALUE_TYPE const, PERMUTATION > const & b = this->m_b.toSliceConst();
    ArraySlice< VALUE_TYPE, PERMUTATION > const & c = this->m_c.toSlice();
    TIMING_LOOP( fortranSliceKernel( a, b, c ); )
  }

  void subscriptView() const
  {
    ArrayView< VALUE_TYPE const, PERMUTATION > const & a = this->m_a.toViewConst();
    ArrayView< VALUE_TYPE const, PERMUTATION > const & b = this->m_b.toViewConst();
    ArrayView< VALUE_TYPE, PERMUTATION > const & c = this->m_c.toView();
    TIMING_LOOP( subscriptViewKernel( a, b, c ); )
  }

  void subscriptSlice() const
  {
    ArraySlice< VALUE_TYPE const, PERMUTATION > const & a = this->m_a.toSliceConst();
    ArraySlice< VALUE_TYPE const, PERMUTATION > const & b = this->m_b.toSliceConst();
    ArraySlice< VALUE_TYPE, PERMUTATION > const & c = this->m_c.toSlice();
    TIMING_LOOP( subscriptSliceKernel( a, b, c ); )
  }

  void RAJAView() const
  {
    RajaView< VALUE_TYPE const, PERMUTATION > const a = makeRajaView( this->m_a );
    RajaView< VALUE_TYPE const, PERMUTATION > const b = makeRajaView( this->m_b );
    RajaView< VALUE_TYPE, PERMUTATION > const c = makeRajaView( this->m_c );
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
