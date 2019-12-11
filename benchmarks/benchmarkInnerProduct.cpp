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

// TPL includes
#include <benchmark/benchmark.h>


namespace LvArray
{
namespace benchmarking
{


#define NATIVE_INIT \
  INDEX_TYPE const N = state.range( 0 ); \
  int iter = 0; \
  Array< VALUE_TYPE, RAJA::PERM_I > a( N ); \
  initialize( a, iter ); \
  Array< VALUE_TYPE, RAJA::PERM_I > b( N ); \
  initialize( b, iter ); \
  VALUE_TYPE sum = 0


#define RAJA_INIT \
  NATIVE_INIT; \
  a.move( RAJAHelper< POLICY >::space ); \
  b.move( RAJAHelper< POLICY >::space )


#define FINISH \
  registerResult( resultsMap, { N }, sum / INDEX_TYPE( state.iterations() ), __PRETTY_FUNCTION__ ); \
  state.SetBytesProcessed( INDEX_TYPE( state.iterations() ) * N * sizeof( VALUE_TYPE ) )


ResultsMap< 1 > resultsMap;


void fortranNative( benchmark::State & state )
{
  NATIVE_INIT;
  ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & aView = a.toViewConst();
  ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & bView = b.toViewConst();
  for( auto _ : state )
  {
    sum += InnerProductNative::fortran( aView, bView, N );
    benchmark::DoNotOptimize( sum );
    benchmark::ClobberMemory();
  }

  FINISH;
}


void subscriptNative( benchmark::State & state )
{
  NATIVE_INIT;
  ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & aView = a.toViewConst();
  ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & bView = b.toViewConst();
  for( auto _ : state )
  {
    sum += InnerProductNative::subscript( aView, bView, N );
    benchmark::DoNotOptimize( sum );
    benchmark::ClobberMemory();
  }

  FINISH;
}


void rajaViewNative( benchmark::State & state )
{
  NATIVE_INIT;
  RajaView< VALUE_TYPE const, RAJA::PERM_I > const aView( a.data(), { a.size() } );
  RajaView< VALUE_TYPE const, RAJA::PERM_I > const bView( b.data(), { a.size() } );
  for( auto _ : state )
  {
    sum += InnerProductNative::rajaView( aView, bView, N );
    benchmark::DoNotOptimize( sum );
    benchmark::ClobberMemory();
  }

  FINISH;
}


void pointerNative( benchmark::State & state )
{
  NATIVE_INIT;
  VALUE_TYPE const * const restrict aPtr = a.data();
  VALUE_TYPE const * const restrict bPtr = b.data();
  for( auto _ : state )
  {
    sum += InnerProductNative::pointer( aPtr, bPtr, N );
    benchmark::DoNotOptimize( sum );
    benchmark::ClobberMemory();
  }

  FINISH;
}


template< typename POLICY >
void fortranRAJA( benchmark::State & state )
{
  NATIVE_INIT;
  ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & aView = a.toViewConst();
  ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & bView = b.toViewConst();
  for( auto _ : state )
  {
    sum += InnerProductRAJA< POLICY >::fortran( aView, bView, N );
    benchmark::DoNotOptimize( sum );
    benchmark::ClobberMemory();
  }

  FINISH;
}


template< typename POLICY >
void subscriptRAJA( benchmark::State & state )
{
  NATIVE_INIT;
  ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & aView = a.toViewConst();
  ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & bView = b.toViewConst();
  for( auto _ : state )
  {
    sum += InnerProductRAJA< POLICY >::subscript( aView, bView, N );
    benchmark::DoNotOptimize( sum );
    benchmark::ClobberMemory();
  }

  FINISH;
}


template< typename POLICY >
void rajaViewRAJA( benchmark::State & state )
{
  NATIVE_INIT;
  RajaView< VALUE_TYPE const, RAJA::PERM_I > const aView( a.data(), { a.size() } );
  RajaView< VALUE_TYPE const, RAJA::PERM_I > const bView( b.data(), { a.size() } );
  for( auto _ : state )
  {
    sum += InnerProductRAJA< POLICY >::rajaView( aView, bView, N );
    benchmark::DoNotOptimize( sum );
    benchmark::ClobberMemory();
  }

  FINISH;
}


template< typename POLICY >
void pointerRAJA( benchmark::State & state )
{
  NATIVE_INIT;
  VALUE_TYPE const * const restrict aPtr = a.data();
  VALUE_TYPE const * const restrict bPtr = b.data();
  for( auto _ : state )
  {
    sum += InnerProductRAJA< POLICY >::pointer( aPtr, bPtr, N );
    benchmark::DoNotOptimize( sum );
    benchmark::ClobberMemory();
  }

  FINISH;
}

int const NUM_REPETITIONS = 3;
#define SERIAL_SIZE { (2 << 20) + 573 \
}

DECLARE_BENCHMARK( fortranNative, SERIAL_SIZE, NUM_REPETITIONS );
DECLARE_BENCHMARK( subscriptNative, SERIAL_SIZE, NUM_REPETITIONS );
DECLARE_BENCHMARK( rajaViewNative, SERIAL_SIZE, NUM_REPETITIONS );
DECLARE_BENCHMARK( pointerNative, SERIAL_SIZE, NUM_REPETITIONS );

FOUR_BENCHMARK_TEMPLATES_ONE_TYPE( fortranRAJA, subscriptRAJA, rajaViewRAJA, pointerRAJA,
                                   serialPolicy, SERIAL_SIZE, NUM_REPETITIONS );

#if defined(USE_OPENMP)
#define OMP_SIZE SERIAL_SIZE
FOUR_BENCHMARK_TEMPLATES_ONE_TYPE( fortranRAJA, subscriptRAJA, rajaViewRAJA, pointerRAJA,
                                   parallelHostPolicy, OMP_SIZE, NUM_REPETITIONS );
#endif

#if defined(USE_CUDA)
#define CUDA_SIZE SERIAL_SIZE
FOUR_BENCHMARK_TEMPLATES_ONE_TYPE( fortranRAJA, subscriptRAJA, rajaViewRAJA, pointerRAJA,
                                   RAJA::cuda_exec< THREADS_PER_BLOCK >, SERIAL_SIZE, NUM_REPETITIONS );
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
