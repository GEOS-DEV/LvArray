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
#include "SparsityPattern.hpp"
#include "ArrayOfArrays.hpp"
#include "StringUtilities.hpp"
#include "benchmarkSparsityGenerationKernels.hpp"

// TPL includes
#include <benchmark/benchmark.h>

// System includes
#include <utility>

namespace LvArray
{
namespace benchmarking
{

#define TIMING_LOOP( KERNEL ) \
  for( auto _ : state ) \
  { \
    KERNEL; \
    benchmark::ClobberMemory(); \
  } \

void elemLoopNoPreallocationNative( ::benchmark::State & state )
{
  SparsityGenerationNative kernels( state );
  TIMING_LOOP( kernels.resize( 0 ); kernels.generateElemLoop() );
}

void elemLoopExactAllocationNative( benchmark::State & state )
{
  SparsityGenerationNative kernels( state );
  TIMING_LOOP( kernels.resizeExact(); kernels.generateElemLoopView() );
}

void elemLoopPreallocatedNative( benchmark::State & state )
{
  SparsityGenerationNative kernels( state );
  TIMING_LOOP( kernels.resize( MAX_COLUMNS_PER_ROW ); kernels.generateElemLoopView() );
}

void nodeLoopNoPreallocationNative( benchmark::State & state )
{
  SparsityGenerationNative kernels( state );
  TIMING_LOOP( kernels.resize( 0 ); kernels.generateNodeLoop() );
}

void nodeLoopExactAllocationNative( benchmark::State & state )
{
  SparsityGenerationNative kernels( state );
  TIMING_LOOP( kernels.resizeExact(); kernels.generateNodeLoop() );
}

void nodeLoopPreallocatedNative( benchmark::State & state )
{
  SparsityGenerationNative kernels( state );
  TIMING_LOOP( kernels.resize( MAX_COLUMNS_PER_ROW ); kernels.generateNodeLoopView() );
}

template< typename POLICY >
void nodeLoopExactAllocationRAJA( benchmark::State & state )
{
  LVARRAY_MARK_FUNCTION_TAG_STRING( std::string( "nodeLoopExactAllocationRAJA_" ) + LvArray::demangleType< POLICY >() );

  SparsityGenerationRAJA< POLICY > kernels( state );
  TIMING_LOOP( kernels.resizeExact(); kernels.generateNodeLoopView() );
}

template< typename POLICY >
void nodeLoopPreallocatedRAJA( benchmark::State & state )
{
  LVARRAY_MARK_FUNCTION_TAG_STRING( std::string( "nodeLoopPreallocatedRAJA_" ) + LvArray::demangleType< POLICY >() );
  SparsityGenerationRAJA< POLICY > kernels( state );
  TIMING_LOOP( kernels.resize( MAX_COLUMNS_PER_ROW ); kernels.generateNodeLoopView() );
}

template< typename POLICY >
void addToRow( benchmark::State & state )
{
  LVARRAY_MARK_FUNCTION_TAG_STRING( std::string( "addToRow_" ) + LvArray::demangleType< POLICY >() );
  CRSMatrixAddToRow< POLICY > kernels( state );
  TIMING_LOOP( kernels.add() );
}

int const NO_ALLOCATION_SIZE = 10;
int const SERIAL_SIZE = 100;

#if defined(USE_OPENMP)
int const OMP_SIZE = 100;
#endif

#if defined(USE_CUDA)
int const CUDA_SIZE = 100;
#endif

void registerBenchmarks()
{
  // Register the native benchmarks.
  REGISTER_BENCHMARK( WRAP( { NO_ALLOCATION_SIZE, NO_ALLOCATION_SIZE, NO_ALLOCATION_SIZE } ), elemLoopNoPreallocationNative );
  REGISTER_BENCHMARK( WRAP( { SERIAL_SIZE, SERIAL_SIZE, SERIAL_SIZE } ), elemLoopExactAllocationNative );
  REGISTER_BENCHMARK( WRAP( { SERIAL_SIZE, SERIAL_SIZE, SERIAL_SIZE } ), elemLoopPreallocatedNative );
  REGISTER_BENCHMARK( WRAP( { NO_ALLOCATION_SIZE, NO_ALLOCATION_SIZE, NO_ALLOCATION_SIZE } ), nodeLoopNoPreallocationNative );
  REGISTER_BENCHMARK( WRAP( { SERIAL_SIZE, SERIAL_SIZE, SERIAL_SIZE } ), nodeLoopExactAllocationNative );
  REGISTER_BENCHMARK( WRAP( { SERIAL_SIZE, SERIAL_SIZE, SERIAL_SIZE } ), nodeLoopPreallocatedNative );

  // Register the RAJA benchmarks.
  forEachArg( []( auto tuple )
  {
    INDEX_TYPE const size = std::get< 0 >( tuple );
    using POLICY = std::tuple_element_t< 1, decltype( tuple ) >;
    REGISTER_BENCHMARK_TEMPLATE( WRAP( { size, size, size } ), nodeLoopExactAllocationRAJA, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( WRAP( { size, size, size } ), nodeLoopPreallocatedRAJA, POLICY );
  },
              std::make_tuple( SERIAL_SIZE, serialPolicy {} )
  #if defined(USE_OPENMP)
              , std::make_tuple( OMP_SIZE, parallelHostPolicy {} )
  #endif
  #if defined(USE_CUDA)
              , std::make_tuple( CUDA_SIZE, parallelDevicePolicy< THREADS_PER_BLOCK > {} )
  #endif
              );

  // Register the RAJA benchmarks.
  forEachArg( []( auto tuple )
  {
    INDEX_TYPE const size = std::get< 0 >( tuple );
    using POLICY = std::tuple_element_t< 1, decltype( tuple ) >;
    REGISTER_BENCHMARK_TEMPLATE( WRAP( { size, size, size } ), addToRow, POLICY );
  },
              std::make_tuple( SERIAL_SIZE, serialPolicy {} )
  #if defined(USE_OPENMP)
              , std::make_tuple( OMP_SIZE, parallelHostPolicy {} )
  #endif
  #if defined(USE_CUDA)
              , std::make_tuple( CUDA_SIZE, parallelDevicePolicy< THREADS_PER_BLOCK > {} )
  #endif
              );
}

} // namespace benchmarking
} // namespace LvArray

int main( int argc, char * * argv )
{
  LvArray::benchmarking::registerBenchmarks();
  ::benchmark::Initialize( &argc, argv );
  if( ::benchmark::ReportUnrecognizedArguments( argc, argv ) )
    return 1;

  LVARRAY_LOG( "COLUMN_TYPE = " << LvArray::demangleType< LvArray::benchmarking::COLUMN_TYPE >() );
  LVARRAY_LOG( "INDEX_TYPE = " << LvArray::demangleType< LvArray::benchmarking::INDEX_TYPE >() );

  LvArray::benchmarking::INDEX_TYPE size = std::pow( LvArray::benchmarking::SERIAL_SIZE, 3 );
  LVARRAY_LOG( "Serial problems of size ( " << size << " )." );

#if defined(USE_OPENMP)
  size = std::pow( LvArray::benchmarking::OMP_SIZE, 3 );
  LVARRAY_LOG( "OMP problems of size ( " << size << " )." );
#endif

#if defined(USE_CUDA)
  size = std::pow( LvArray::benchmarking::CUDA_SIZE, 3 );
  LVARRAY_LOG( "CUDA problems of size ( " << size << " )." );
#endif

  ::benchmark::RunSpecifiedBenchmarks();
}
