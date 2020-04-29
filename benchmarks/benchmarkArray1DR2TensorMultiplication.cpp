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
#include "benchmarkArray1DR2TensorMultiplicationKernels.hpp"

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
  int iter = 0; \
  Array< VALUE_TYPE, PERMUTATION > a; \
  a.resizeWithoutInitializationOrDestruction( N, 3, 3 ); \
  initialize( a, iter ); \
  Array< VALUE_TYPE, PERMUTATION > b; \
  b.resizeWithoutInitializationOrDestruction( N, 3, 3 ); \
  initialize( b, iter ); \
  Array< VALUE_TYPE, PERMUTATION > c( N, 3, 3 )


#define RAJA_INIT \
  using PERMUTATION = typename PERMUTATION_POLICY_PAIR::first_type; \
  using POLICY = typename PERMUTATION_POLICY_PAIR::second_type; \
  NATIVE_INIT; \
  a.move( RAJAHelper< POLICY >::space ); \
  b.move( RAJAHelper< POLICY >::space ); \
  c.move( RAJAHelper< POLICY >::space )


#define NATIVE_FINISH \
  state.SetBytesProcessed( INDEX_TYPE( state.iterations() ) * N * 3 * 3 * 3 * sizeof( VALUE_TYPE ) ); \
  VALUE_TYPE const result = reduce( c ) / INDEX_TYPE( state.iterations() ); \
  registerResult( resultsMap, { N }, result, __PRETTY_FUNCTION__ )


#define RAJA_FINISH \
  c.move( chai::CPU ); \
  NATIVE_FINISH


ResultsMap< 1 > resultsMap;


template< typename PERMUTATION >
void fortranNative( benchmark::State & state )
{
  NATIVE_INIT;
  ArrayView< VALUE_TYPE const, PERMUTATION > const & aView = a;
  ArrayView< VALUE_TYPE const, PERMUTATION > const & bView = b;
  ArrayView< VALUE_TYPE, PERMUTATION > const & cView = c;
  for( auto _ : state )
  {
    Array1DR2TensorMultiplicationNative< PERMUTATION >::fortran( aView, bView, cView, N );
    benchmark::ClobberMemory();
  }

  NATIVE_FINISH;
}


template< typename PERMUTATION >
void subscriptNative( benchmark::State & state )
{
  NATIVE_INIT;
  ArrayView< VALUE_TYPE const, PERMUTATION > const & aView = a;
  ArrayView< VALUE_TYPE const, PERMUTATION > const & bView = b;
  ArrayView< VALUE_TYPE, PERMUTATION > const & cView = c;
  for( auto _ : state )
  {
    Array1DR2TensorMultiplicationNative< PERMUTATION >::subscript( aView, bView, cView, N );
    benchmark::ClobberMemory();
  }

  NATIVE_FINISH;
}


template< typename PERMUTATION >
void tensorAbstractionNative( benchmark::State & state )
{
  NATIVE_INIT;
  ArrayView< VALUE_TYPE const, PERMUTATION > const & aView = a;
  ArrayView< VALUE_TYPE const, PERMUTATION > const & bView = b;
  ArrayView< VALUE_TYPE, PERMUTATION > const & cView = c;
  for( auto _ : state )
  {
    Array1DR2TensorMultiplicationNative< PERMUTATION >::tensorAbstraction( aView, bView, cView, N );
    benchmark::ClobberMemory();
  }

  NATIVE_FINISH;
}


template< typename PERMUTATION >
void rajaViewNative( benchmark::State & state )
{
  NATIVE_INIT;
  RajaView< VALUE_TYPE const, PERMUTATION > const aView = makeRajaView( a );
  RajaView< VALUE_TYPE const, PERMUTATION > const bView = makeRajaView( b );
  RajaView< VALUE_TYPE, PERMUTATION > const cView = makeRajaView( c );
  for( auto _ : state )
  {
    Array1DR2TensorMultiplicationNative< PERMUTATION >::rajaView( aView, bView, cView, N );
    benchmark::ClobberMemory();
  }

  NATIVE_FINISH;
}


template< typename PERMUTATION >
void pointerNative( benchmark::State & state )
{
  NATIVE_INIT;
  VALUE_TYPE const * const LVARRAY_RESTRICT aView = a.data();
  VALUE_TYPE const * const LVARRAY_RESTRICT bView = b.data();
  VALUE_TYPE * const LVARRAY_RESTRICT cView = c.data();
  for( auto _ : state )
  {
    Array1DR2TensorMultiplicationNative< PERMUTATION >::pointer( aView, bView, cView, N );
    benchmark::ClobberMemory();
  }

  NATIVE_FINISH;
}


template< typename PERMUTATION_POLICY_PAIR >
void fortranRaja( benchmark::State & state )
{
  RAJA_INIT;
  ArrayView< VALUE_TYPE const, PERMUTATION > const & aView = a;
  ArrayView< VALUE_TYPE const, PERMUTATION > const & bView = b;
  ArrayView< VALUE_TYPE, PERMUTATION > const & cView = c;
  for( auto _ : state )
  {
    Array1DR2TensorMultiplicationRaja< PERMUTATION, POLICY >::fortran( aView, bView, cView, N );
    benchmark::ClobberMemory();
  }

  RAJA_FINISH;
}


template< typename PERMUTATION_POLICY_PAIR >
void subscriptRaja( benchmark::State & state )
{
  RAJA_INIT;
  ArrayView< VALUE_TYPE const, PERMUTATION > const & aView = a;
  ArrayView< VALUE_TYPE const, PERMUTATION > const & bView = b;
  ArrayView< VALUE_TYPE, PERMUTATION > const & cView = c;
  for( auto _ : state )
  {
    Array1DR2TensorMultiplicationRaja< PERMUTATION, POLICY >::subscript( aView, bView, cView, N );
    benchmark::ClobberMemory();
  }

  RAJA_FINISH;
}


template< typename PERMUTATION_POLICY_PAIR >
void tensorAbstractionRaja( benchmark::State & state )
{
  RAJA_INIT;
  ArrayView< VALUE_TYPE const, PERMUTATION > const & aView = a;
  ArrayView< VALUE_TYPE const, PERMUTATION > const & bView = b;
  ArrayView< VALUE_TYPE, PERMUTATION > const & cView = c;
  for( auto _ : state )
  {
    Array1DR2TensorMultiplicationRaja< PERMUTATION, POLICY >::tensorAbstraction( aView, bView, cView, N );
    benchmark::ClobberMemory();
  }

  RAJA_FINISH;
}


template< typename PERMUTATION_POLICY_PAIR >
void rajaViewRaja( benchmark::State & state )
{
  RAJA_INIT;
  RajaView< VALUE_TYPE const, PERMUTATION > const aView = makeRajaView( a );
  RajaView< VALUE_TYPE const, PERMUTATION > const bView = makeRajaView( b );
  RajaView< VALUE_TYPE, PERMUTATION > const cView = makeRajaView( c );
  for( auto _ : state )
  {
    Array1DR2TensorMultiplicationRaja< PERMUTATION, POLICY >::rajaView( aView, bView, cView, N );
    benchmark::ClobberMemory();
  }

  RAJA_FINISH;
}


template< typename PERMUTATION_POLICY_PAIR >
void pointerRaja( benchmark::State & state )
{
  RAJA_INIT;
  VALUE_TYPE const * const LVARRAY_RESTRICT aView = a.data();
  VALUE_TYPE const * const LVARRAY_RESTRICT bView = b.data();
  VALUE_TYPE * const LVARRAY_RESTRICT cView = c.data();
  for( auto _ : state )
  {
    Array1DR2TensorMultiplicationRaja< PERMUTATION, POLICY >::pointer( aView, bView, cView, N );
    benchmark::ClobberMemory();
  }

  RAJA_FINISH;
}



int const NUM_REPETITIONS = 3;
#define SERIAL_SIZE ( (2 << 20) - 87 )

FIVE_BENCHMARK_TEMPLATES_ONE_TYPE( fortranNative, subscriptNative, tensorAbstractionNative, rajaViewNative,
                                   pointerNative, RAJA::PERM_IJK, { SERIAL_SIZE }, NUM_REPETITIONS );

FIVE_BENCHMARK_TEMPLATES_ONE_TYPE( fortranNative, subscriptNative, tensorAbstractionNative, rajaViewNative, pointerNative,
                                   RAJA::PERM_KJI, { SERIAL_SIZE }, NUM_REPETITIONS );

FIVE_BENCHMARK_TEMPLATES_TWO_TYPES( fortranRaja, subscriptRaja, tensorAbstractionRaja, rajaViewRaja,
                                    pointerRaja, RAJA::PERM_IJK, serialPolicy, { SERIAL_SIZE }, NUM_REPETITIONS );

FIVE_BENCHMARK_TEMPLATES_TWO_TYPES( fortranRaja, subscriptRaja, tensorAbstractionRaja, rajaViewRaja,
                                    pointerRaja, RAJA::PERM_KJI, serialPolicy, { SERIAL_SIZE }, NUM_REPETITIONS );

#if defined(USE_OPENMP)

#define OMP_SIZE ( (2 << 22) - 87 )

FIVE_BENCHMARK_TEMPLATES_TWO_TYPES( fortranRaja, subscriptRaja, tensorAbstractionRaja, rajaViewRaja,
                                    pointerRaja, RAJA::PERM_IJK, parallelHostPolicy, { OMP_SIZE }, NUM_REPETITIONS );

FIVE_BENCHMARK_TEMPLATES_TWO_TYPES( fortranRaja, subscriptRaja, tensorAbstractionRaja, rajaViewRaja,
                                    pointerRaja, RAJA::PERM_KJI, parallelHostPolicy, { OMP_SIZE }, NUM_REPETITIONS );

#endif

#if defined(USE_CUDA)

#define CUDA_SIZE ( (2 << 24) - 87 )

FIVE_BENCHMARK_TEMPLATES_TWO_TYPES( fortranRaja, subscriptRaja, tensorAbstractionRaja, rajaViewRaja,
                                    pointerRaja, RAJA::PERM_IJK, RAJA::cuda_exec< THREADS_PER_BLOCK >,
                                    { CUDA_SIZE }, NUM_REPETITIONS );

FIVE_BENCHMARK_TEMPLATES_TWO_TYPES( fortranRaja, subscriptRaja, tensorAbstractionRaja, rajaViewRaja,
                                    pointerRaja, RAJA::PERM_KJI, RAJA::cuda_exec< THREADS_PER_BLOCK >,
                                    { CUDA_SIZE }, NUM_REPETITIONS );

#endif

} // namespace benchmarking
} // namespace LvArray

int main( int argc, char * * argv )
{
  ::benchmark::Initialize( &argc, argv );
  if( ::benchmark::ReportUnrecognizedArguments( argc, argv ) )
    return 1;

  LVARRAY_LOG( "VALUE_TYPE = " << cxx_utilities::demangleType< LvArray::benchmarking::VALUE_TYPE >() );
  LVARRAY_LOG( "Serial problems of size ( " << SERIAL_SIZE << ", 3, 3 )." );

#if defined(USE_OPENMP)
  LVARRAY_LOG( "OMP problems of size ( " << OMP_SIZE << ", 3, 3 )." );
#endif

#if defined(USE_CUDA)
  LVARRAY_LOG( "CUDA problems of size ( " << CUDA_SIZE << ", 3, 3 )." );
#endif

  ::benchmark::RunSpecifiedBenchmarks();

  return LvArray::benchmarking::verifyResults( LvArray::benchmarking::resultsMap );
}
