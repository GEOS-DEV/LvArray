/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "benchmarkHelpers.hpp"
#include "SparsityPattern.hpp"
#include "ArrayOfArrays.hpp"
#include "system.hpp"
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

void elemLoopExactAllocationNative( benchmark::State & state )
{
  SparsityGenerationNative kernels( state );
  TIMING_LOOP( kernels.resizeExact(); kernels.generateElemLoop() );
}

void elemLoopPreallocatedNative( benchmark::State & state )
{
  SparsityGenerationNative kernels( state );
  TIMING_LOOP( kernels.resize( MAX_COLUMNS_PER_ROW ); kernels.generateElemLoop() );
}

void nodeLoopExactAllocationNative( benchmark::State & state )
{
  SparsityGenerationNative kernels( state );
  TIMING_LOOP( kernels.resizeExact(); kernels.generateNodeLoop() );
}

void nodeLoopPreallocatedNative( benchmark::State & state )
{
  SparsityGenerationNative kernels( state );
  TIMING_LOOP( kernels.resize( MAX_COLUMNS_PER_ROW ); kernels.generateNodeLoop() );
}

template< typename POLICY >
void nodeLoopExactAllocationRAJA( benchmark::State & state )
{
  CALI_CXX_MARK_PRETTY_FUNCTION;
  SparsityGenerationRAJA< POLICY > kernels( state );
  TIMING_LOOP( kernels.resizeExact(); kernels.generateNodeLoop() );
}

template< typename POLICY >
void nodeLoopPreallocatedRAJA( benchmark::State & state )
{
  CALI_CXX_MARK_PRETTY_FUNCTION;
  SparsityGenerationRAJA< POLICY > kernels( state );
  TIMING_LOOP( kernels.resize( MAX_COLUMNS_PER_ROW ); kernels.generateNodeLoop() );
}

template< typename POLICY >
void addToRow( benchmark::State & state )
{
  CALI_CXX_MARK_PRETTY_FUNCTION;
  CRSMatrixAddToRow< POLICY > kernels( state );
  TIMING_LOOP( kernels.add() );
}

int const SERIAL_SIZE = 100;

#if defined(RAJA_ENABLE_OPENMP)
int const OMP_SIZE = 100;
#endif

#if defined(LVARRAY_USE_CUDA) && defined(LVARRAY_USE_CHAI)
int const CUDA_SIZE = 100;
#endif

void registerBenchmarks()
{
  // Register the native benchmarks.
  REGISTER_BENCHMARK( WRAP( { SERIAL_SIZE, SERIAL_SIZE, SERIAL_SIZE } ), elemLoopExactAllocationNative );
  REGISTER_BENCHMARK( WRAP( { SERIAL_SIZE, SERIAL_SIZE, SERIAL_SIZE } ), elemLoopPreallocatedNative );
  REGISTER_BENCHMARK( WRAP( { SERIAL_SIZE, SERIAL_SIZE, SERIAL_SIZE } ), nodeLoopExactAllocationNative );
  REGISTER_BENCHMARK( WRAP( { SERIAL_SIZE, SERIAL_SIZE, SERIAL_SIZE } ), nodeLoopPreallocatedNative );

  // Register the RAJA benchmarks.
  typeManipulation::forEachArg( []( auto tuple )
  {
    INDEX_TYPE const size = std::get< 0 >( tuple );
    using POLICY = std::tuple_element_t< 1, decltype( tuple ) >;
    REGISTER_BENCHMARK_TEMPLATE( WRAP( { size, size, size } ), nodeLoopExactAllocationRAJA, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( WRAP( { size, size, size } ), nodeLoopPreallocatedRAJA, POLICY );
  },
                                std::make_tuple( SERIAL_SIZE, serialPolicy {} )
  #if defined(RAJA_ENABLE_OPENMP)
                                , std::make_tuple( OMP_SIZE, parallelHostPolicy {} )
  #endif
  #if defined(LVARRAY_USE_CUDA) && defined(LVARRAY_USE_CHAI)
                                , std::make_tuple( CUDA_SIZE, parallelDevicePolicy< THREADS_PER_BLOCK > {} )
  #endif
                                );

  // Register the RAJA benchmarks.
  typeManipulation::forEachArg( []( auto tuple )
  {
    INDEX_TYPE const size = std::get< 0 >( tuple );
    using POLICY = std::tuple_element_t< 1, decltype( tuple ) >;
    REGISTER_BENCHMARK_TEMPLATE( WRAP( { size, size, size } ), addToRow, POLICY );
  },
                                std::make_tuple( SERIAL_SIZE, serialPolicy {} )
  #if defined(RAJA_ENABLE_OPENMP)
                                , std::make_tuple( OMP_SIZE, parallelHostPolicy {} )
  #endif
  #if defined(LVARRAY_USE_CUDA) && defined(LVARRAY_USE_CHAI)
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

  LVARRAY_LOG( "COLUMN_TYPE = " << LvArray::system::demangleType< LvArray::benchmarking::COLUMN_TYPE >() );
  LVARRAY_LOG( "INDEX_TYPE = " << LvArray::system::demangleType< LvArray::benchmarking::INDEX_TYPE >() );

  LvArray::benchmarking::INDEX_TYPE size = std::pow( LvArray::benchmarking::SERIAL_SIZE, 3 );
  LVARRAY_LOG( "Serial problems of size ( " << size << " )." );

#if defined(RAJA_ENABLE_OPENMP)
  size = std::pow( LvArray::benchmarking::OMP_SIZE, 3 );
  LVARRAY_LOG( "OMP problems of size ( " << size << " )." );
#endif

#if defined(LVARRAY_USE_CUDA) && defined(LVARRAY_USE_CHAI)
  size = std::pow( LvArray::benchmarking::CUDA_SIZE, 3 );
  LVARRAY_LOG( "CUDA problems of size ( " << size << " )." );
#endif

  ::benchmark::RunSpecifiedBenchmarks();
}
