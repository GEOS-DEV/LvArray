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

ResultsMap< VALUE_TYPE, 1 > resultsMap;

void fortranArrayNative( benchmark::State & state )
{
  InnerProductNative kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.fortranArray();
}

void fortranViewNative( benchmark::State & state )
{
  InnerProductNative kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.fortranView();
}

void fortranSliceNative( benchmark::State & state )
{
  InnerProductNative kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.fortranSlice();
}

void subscriptArrayNative( benchmark::State & state )
{
  InnerProductNative kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.subscriptArray();
}

void subscriptViewNative( benchmark::State & state )
{
  InnerProductNative kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.subscriptView();
}

void subscriptSliceNative( benchmark::State & state )
{
  InnerProductNative kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.subscriptSlice();
}

void rajaViewNative( benchmark::State & state )
{
  InnerProductNative kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.RAJAView();
}

void pointerNative( benchmark::State & state )
{
  InnerProductNative kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.pointer();
}


template< typename POLICY >
void fortranViewRAJA( benchmark::State & state )
{
  InnerProductRAJA< POLICY > kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.fortranView();
}

template< typename POLICY >
void fortranSliceRAJA( benchmark::State & state )
{
  InnerProductRAJA< POLICY > kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.fortranSlice();
}

template< typename POLICY >
void subscriptArrayRAJA( benchmark::State & state )
{
  InnerProductRAJA< POLICY > kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.subscriptArray();
}

template< typename POLICY >
void subscriptViewRAJA( benchmark::State & state )
{
  InnerProductRAJA< POLICY > kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.subscriptView();
}

template< typename POLICY >
void subscriptSliceRAJA( benchmark::State & state )
{
  InnerProductRAJA< POLICY > kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.subscriptSlice();
}

template< typename POLICY >
void rajaViewRAJA( benchmark::State & state )
{
  InnerProductRAJA< POLICY > kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.RAJAView();
}

template< typename POLICY >
void pointerRAJA( benchmark::State & state )
{
  InnerProductRAJA< POLICY > kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.pointer();
}


INDEX_TYPE const SERIAL_SIZE = (2 << 20) + 573;
#if defined(USE_OPENMP)
INDEX_TYPE const OMP_SIZE = SERIAL_SIZE;
#endif
#if defined(USE_CUDA)
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
  forEachArg( []( auto tuple )
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

  LVARRAY_LOG( "VALUE_TYPE = " << LvArray::demangleType< LvArray::benchmarking::VALUE_TYPE >() );
  LVARRAY_LOG( "INDEX_TYPE = " << LvArray::demangleType< LvArray::benchmarking::INDEX_TYPE >() );

  LVARRAY_LOG( "Serial problems of size ( " << LvArray::benchmarking::SERIAL_SIZE << " )." );

#if defined(USE_OPENMP)
  LVARRAY_LOG( "OMP problems of size ( " << LvArray::benchmarking::OMP_SIZE << " )." );
#endif

#if defined(USE_CUDA)
  LVARRAY_LOG( "CUDA problems of size ( " << LvArray::benchmarking::CUDA_SIZE << " )." );
#endif

  ::benchmark::RunSpecifiedBenchmarks();

  return LvArray::benchmarking::verifyResults( LvArray::benchmarking::resultsMap );
}
