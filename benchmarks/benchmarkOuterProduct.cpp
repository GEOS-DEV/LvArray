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
#include "benchmarkOuterProductKernels.hpp"

// TPL includes
#include <benchmark/benchmark.h>

// System includes
#include <utility>

namespace LvArray
{
namespace benchmarking
{


#define NATIVE_INIT \
  INDEX_TYPE const N = state.range( 0 ); \
  INDEX_TYPE const M = state.range( 1 ); \
  int iter = 0; \
  Array< VALUE_TYPE, RAJA::PERM_I > a( N ); \
  initialize( a, iter ); \
  Array< VALUE_TYPE, RAJA::PERM_I > b( M ); \
  initialize( b, iter ); \
  Array< VALUE_TYPE, PERMUTATION > c( N, M )


#define RAJA_INIT \
  using PERMUTATION = typename PERMUTATION_POLICY_PAIR::first_type; \
  using POLICY = typename PERMUTATION_POLICY_PAIR::second_type; \
  NATIVE_INIT; \
  a.move( RAJAHelper< POLICY >::space ); \
  b.move( RAJAHelper< POLICY >::space ); \
  c.move( RAJAHelper< POLICY >::space )


#define NATIVE_FINISH \
  state.SetBytesProcessed( INDEX_TYPE( state.iterations() ) * N * M * sizeof( VALUE_TYPE ) ); \
  VALUE_TYPE const result = reduce( c ) / INDEX_TYPE( state.iterations() ); \
  registerResult( resultsMap, { N, M }, result, __PRETTY_FUNCTION__ )


#define RAJA_FINISH \
  c.move( chai::CPU ); \
  NATIVE_FINISH


ResultsMap< 2 > resultsMap;

template< typename PERMUTATION >
void fortranNative( benchmark::State & state )
{
  NATIVE_INIT;
  ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & aView = a.toViewConst();
  ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & bView = b.toViewConst();
  ArrayView< VALUE_TYPE, PERMUTATION > const & cView = c.toView();
  for( auto _ : state )
  {
    OuterProductNative< PERMUTATION >::fortran( aView, bView, cView, N, M );
    benchmark::ClobberMemory();
  }

  NATIVE_FINISH;
}

template< typename PERMUTATION >
void subscriptNative( benchmark::State & state )
{
  NATIVE_INIT;
  ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & aView = a.toViewConst();
  ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & bView = b.toViewConst();
  ArrayView< VALUE_TYPE, PERMUTATION > const & cView = c.toView();
  for( auto _ : state )
  {
    OuterProductNative< PERMUTATION >::subscript( aView, bView, cView, N, M );
    benchmark::ClobberMemory();
  }

  NATIVE_FINISH;
}

template< typename PERMUTATION >
void rajaViewNative( benchmark::State & state )
{
  NATIVE_INIT;
  RajaView< VALUE_TYPE const, RAJA::PERM_I > const aView = makeRajaView( a );
  RajaView< VALUE_TYPE const, RAJA::PERM_I > const bView = makeRajaView( b );
  RajaView< VALUE_TYPE, PERMUTATION > const cView = makeRajaView( c );
  for( auto _ : state )
  {
    OuterProductNative< PERMUTATION >::rajaView( aView, bView, cView, N, M );
    benchmark::ClobberMemory();
  }

  NATIVE_FINISH;
}

template< typename PERMUTATION >
void pointerNative( benchmark::State & state )
{
  NATIVE_INIT;
  VALUE_TYPE const * const restrict aView = a.data();
  VALUE_TYPE const * const restrict bView = b.data();
  VALUE_TYPE * const restrict cView = c.data();
  for( auto _ : state )
  {
    OuterProductNative< PERMUTATION >::pointer( aView, bView, cView, N, M );
    benchmark::ClobberMemory();
  }

  NATIVE_FINISH;
}

template< typename PERMUTATION_POLICY_PAIR >
void fortranRAJA( benchmark::State & state )
{
  RAJA_INIT;
  ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & aView = a.toViewConst();
  ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & bView = b.toViewConst();
  ArrayView< VALUE_TYPE, PERMUTATION > const & cView = c.toView();
  for( auto _ : state )
  {
    OuterProductRAJA< PERMUTATION, POLICY >::fortran( aView, bView, cView, N, M );
    benchmark::ClobberMemory();
  }

  RAJA_FINISH;
}

template< typename PERMUTATION_POLICY_PAIR >
void subscriptRAJA( benchmark::State & state )
{
  RAJA_INIT;
  ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & aView = a.toViewConst();
  ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & bView = b.toViewConst();
  ArrayView< VALUE_TYPE, PERMUTATION > const & cView = c.toView();
  for( auto _ : state )
  {
    OuterProductRAJA< PERMUTATION, POLICY >::subscript( aView, bView, cView, N, M );
    benchmark::ClobberMemory();
  }

  RAJA_FINISH;
}

template< typename PERMUTATION_POLICY_PAIR >
void rajaViewRAJA( benchmark::State & state )
{
  RAJA_INIT;
  RajaView< VALUE_TYPE const, RAJA::PERM_I > const aView = makeRajaView( a );
  RajaView< VALUE_TYPE const, RAJA::PERM_I > const bView = makeRajaView( b );
  RajaView< VALUE_TYPE, PERMUTATION > const cView = makeRajaView( c );
  for( auto _ : state )
  {
    OuterProductRAJA< PERMUTATION, POLICY >::rajaView( aView, bView, cView, N, M );
    benchmark::ClobberMemory();
  }

  RAJA_FINISH;
}

template< typename PERMUTATION_POLICY_PAIR >
void pointerRAJA( benchmark::State & state )
{
  RAJA_INIT;
  VALUE_TYPE const * const restrict aView = a.data();
  VALUE_TYPE const * const restrict bView = b.data();
  VALUE_TYPE * const restrict cView = c.data();
  for( auto _ : state )
  {
    OuterProductRAJA< PERMUTATION, POLICY >::pointer( aView, bView, cView, N, M );
    benchmark::ClobberMemory();
  }

  RAJA_FINISH;
}

int const NUM_REPETITIONS = 3;
#define SERIAL_SIZE { \
    { (2 << 9) + 73, (2 << 9) - 71 } \
}

FOUR_BENCHMARK_TEMPLATES_ONE_TYPE( fortranNative, subscriptNative, rajaViewNative, pointerNative,
                                   RAJA::PERM_IJ, SERIAL_SIZE, NUM_REPETITIONS );

FOUR_BENCHMARK_TEMPLATES_ONE_TYPE( fortranNative, subscriptNative, rajaViewNative, pointerNative,
                                   RAJA::PERM_JI, SERIAL_SIZE, NUM_REPETITIONS );

FOUR_BENCHMARK_TEMPLATES_TWO_TYPES( fortranRAJA, subscriptRAJA, rajaViewRAJA, pointerRAJA,
                                    RAJA::PERM_IJ, serialPolicy, SERIAL_SIZE, NUM_REPETITIONS );

FOUR_BENCHMARK_TEMPLATES_TWO_TYPES( fortranRAJA, subscriptRAJA, rajaViewRAJA, pointerRAJA,
                                    RAJA::PERM_JI, serialPolicy, SERIAL_SIZE, NUM_REPETITIONS );

#if defined(USE_OPENMP)

#define OMP_SIZE SERIAL_SIZE

FOUR_BENCHMARK_TEMPLATES_TWO_TYPES( fortranRAJA, subscriptRAJA, rajaViewRAJA, pointerRAJA,
                                    RAJA::PERM_IJ, parallelHostPolicy, OMP_SIZE, NUM_REPETITIONS );

FOUR_BENCHMARK_TEMPLATES_TWO_TYPES( fortranRAJA, subscriptRAJA, rajaViewRAJA, pointerRAJA,
                                    RAJA::PERM_JI, parallelHostPolicy, OMP_SIZE, NUM_REPETITIONS );

#endif

#if defined(USE_CUDA)

#define CUDA_SIZE SERIAL_SIZE

FOUR_BENCHMARK_TEMPLATES_TWO_TYPES( fortranRAJA, subscriptRAJA, rajaViewRAJA, pointerRAJA,
                                    RAJA::PERM_IJ, RAJA::cuda_exec< THREADS_PER_BLOCK >, CUDA_SIZE, NUM_REPETITIONS );

FOUR_BENCHMARK_TEMPLATES_TWO_TYPES( fortranRAJA, subscriptRAJA, rajaViewRAJA, pointerRAJA,
                                    RAJA::PERM_JI, RAJA::cuda_exec< THREADS_PER_BLOCK >, CUDA_SIZE, NUM_REPETITIONS );

#endif

} // namespace benchmarking
} // namespace LvArray

int main( int argc, char * * argv )
{
  ::benchmark::Initialize( &argc, argv );
  if( ::benchmark::ReportUnrecognizedArguments( argc, argv ) )
    return 1;

  ::benchmark::RunSpecifiedBenchmarks();

  return LvArray::benchmarking::verifyResults( LvArray::benchmarking::resultsMap );
}
