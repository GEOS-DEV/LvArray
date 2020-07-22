/*
 * Copyright (c) 2020, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "benchmarkReduceKernels.hpp"

// TPL includes
#include <benchmark/benchmark.h>


namespace LvArray
{
namespace benchmarking
{

ResultsMap< VALUE_TYPE, 1 > resultsMap;

void fortranArrayNative( benchmark::State & state )
{
  ReduceNative kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.fortranArray();
}

void fortranViewNative( benchmark::State & state )
{
  ReduceNative kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.fortranView();
}

void fortranSliceNative( benchmark::State & state )
{
  ReduceNative kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.fortranSlice();
}

void subscriptArrayNative( benchmark::State & state )
{
  ReduceNative kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.subscriptArray();
}

void subscriptViewNative( benchmark::State & state )
{
  ReduceNative kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.subscriptView();
}

void subscriptSliceNative( benchmark::State & state )
{
  ReduceNative kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.subscriptSlice();
}

void rajaViewNative( benchmark::State & state )
{
  ReduceNative kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.RAJAView();
}

void pointerNative( benchmark::State & state )
{
  ReduceNative kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.pointer();
}

template< typename POLICY >
void fortranArrayRAJA( benchmark::State & state )
{
  ReduceRAJA< POLICY > kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.fortranArray();
}

template< typename POLICY >
void fortranViewRAJA( benchmark::State & state )
{
  ReduceRAJA< POLICY > kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.fortranView();
}

template< typename POLICY >
void fortranSliceRAJA( benchmark::State & state )
{
  ReduceRAJA< POLICY > kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.fortranSlice();
}

template< typename POLICY >
void subscriptViewRAJA( benchmark::State & state )
{
  ReduceRAJA< POLICY > kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.subscriptView();
}

template< typename POLICY >
void subscriptSliceRAJA( benchmark::State & state )
{
  ReduceRAJA< POLICY > kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.subscriptSlice();
}

template< typename POLICY >
void rajaViewRAJA( benchmark::State & state )
{
  ReduceRAJA< POLICY > kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.RAJAView();
}

template< typename POLICY >
void pointerRAJA( benchmark::State & state )
{
  ReduceRAJA< POLICY > kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.pointer();
}

INDEX_TYPE const SERIAL_SIZE = (2 << 20) + 573;
#if defined(USE_OPENMP)
INDEX_TYPE const OMP_SIZE = SERIAL_SIZE;
#endif
#if defined(USE_CUDA) && defined(USE_CHAI)
INDEX_TYPE const CUDA_SIZE = SERIAL_SIZE;
#endif

void registerBenchmarks()
{
  // Register the native benchmarks.
  REGISTER_BENCHMARK( { SERIAL_SIZE }, fortranArrayNative );
  REGISTER_BENCHMARK( { SERIAL_SIZE }, fortranViewNative );
  REGISTER_BENCHMARK( { SERIAL_SIZE }, fortranSliceNative );
  REGISTER_BENCHMARK( { SERIAL_SIZE }, subscriptArrayNative );
  REGISTER_BENCHMARK( { SERIAL_SIZE }, subscriptViewNative );
  REGISTER_BENCHMARK( { SERIAL_SIZE }, subscriptSliceNative );
  REGISTER_BENCHMARK( { SERIAL_SIZE }, rajaViewNative );
  REGISTER_BENCHMARK( { SERIAL_SIZE }, pointerNative );

  // Register the RAJA benchmarks.
  typeManipulation::forEachArg( []( auto tuple )
  {
    INDEX_TYPE const size = std::get< 0 >( tuple );
    using POLICY = std::tuple_element_t< 1, decltype( tuple ) >;
    REGISTER_BENCHMARK_TEMPLATE( { size }, fortranViewRAJA, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( { size }, fortranSliceRAJA, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( { size }, subscriptViewRAJA, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( { size }, subscriptSliceRAJA, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( { size }, rajaViewRAJA, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( { size }, pointerRAJA, POLICY );
  },
                                std::make_tuple( SERIAL_SIZE, serialPolicy {} )
  #if defined(USE_OPENMP)
                                , std::make_tuple( OMP_SIZE, parallelHostPolicy {} )
  #endif
  #if defined(USE_CUDA) && defined(USE_CHAI)
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
  {
    return 1;
  }

  LVARRAY_LOG( "VALUE_TYPE = " << LvArray::system::demangleType< LvArray::benchmarking::VALUE_TYPE >() );
  LVARRAY_LOG( "INDEX_TYPE = " << LvArray::system::demangleType< LvArray::benchmarking::INDEX_TYPE >() );

  LVARRAY_LOG( "Serial problems of size ( " << LvArray::benchmarking::SERIAL_SIZE << " )." );

#if defined(USE_OPENMP)
  LVARRAY_LOG( "OMP problems of size ( " << LvArray::benchmarking::OMP_SIZE << " )." );
#endif

#if defined(USE_CUDA) && defined(USE_CHAI)
  LVARRAY_LOG( "CUDA problems of size ( " << LvArray::benchmarking::CUDA_SIZE << " )." );
#endif

  ::benchmark::RunSpecifiedBenchmarks();

  return LvArray::benchmarking::verifyResults( LvArray::benchmarking::resultsMap );
}
