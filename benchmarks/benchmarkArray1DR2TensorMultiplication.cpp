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

ResultsMap< VALUE_TYPE, 1 > resultsMap;

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION >
void fortranArrayNative( benchmark::State & state )
{
  ArrayOfR2TensorsNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.fortranArray();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION >
void fortranViewNative( benchmark::State & state )
{
  ArrayOfR2TensorsNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.fortranView();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION >
void fortranSliceNative( benchmark::State & state )
{
  ArrayOfR2TensorsNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.fortranSlice();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION >
void subscriptArrayNative( benchmark::State & state )
{
  ArrayOfR2TensorsNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.subscriptArray();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION >
void subscriptViewNative( benchmark::State & state )
{
  ArrayOfR2TensorsNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.subscriptView();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION >
void subscriptSliceNative( benchmark::State & state )
{
  ArrayOfR2TensorsNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.subscriptSlice();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION >
void tensorAbstractionArrayNative( benchmark::State & state )
{
  ArrayOfR2TensorsNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.tensorAbstractionArray();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION >
void tensorAbstractionViewNative( benchmark::State & state )
{
  ArrayOfR2TensorsNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.tensorAbstractionView();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION >
void tensorAbstractionSliceNative( benchmark::State & state )
{
  ArrayOfR2TensorsNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.tensorAbstractionSlice();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION >
void RAJAViewNative( benchmark::State & state )
{
  ArrayOfR2TensorsNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.RAJAView();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION >
void pointerNative( benchmark::State & state )
{
  ArrayOfR2TensorsNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.pointer();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION, typename POLICY >
void fortranViewRAJA( benchmark::State & state )
{
  ArrayOfR2TensorsRAJA< PERMUTATION, POLICY > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.fortranView();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION, typename POLICY >
void fortranSliceRAJA( benchmark::State & state )
{
  ArrayOfR2TensorsRAJA< PERMUTATION, POLICY > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.fortranSlice();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION, typename POLICY >
void subscriptViewRAJA( benchmark::State & state )
{
  ArrayOfR2TensorsRAJA< PERMUTATION, POLICY > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.subscriptView();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION, typename POLICY >
void subscriptSliceRAJA( benchmark::State & state )
{
  ArrayOfR2TensorsRAJA< PERMUTATION, POLICY > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.subscriptSlice();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION, typename POLICY >
void tensorAbstractionViewRAJA( benchmark::State & state )
{
  ArrayOfR2TensorsRAJA< PERMUTATION, POLICY > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.tensorAbstractionView();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION, typename POLICY >
void tensorAbstractionSliceRAJA( benchmark::State & state )
{
  ArrayOfR2TensorsRAJA< PERMUTATION, POLICY > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.tensorAbstractionSlice();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION, typename POLICY >
void RAJAViewRAJA( benchmark::State & state )
{
  ArrayOfR2TensorsRAJA< PERMUTATION, POLICY > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.RAJAView();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION, typename POLICY >
void pointerRAJA( benchmark::State & state )
{
  ArrayOfR2TensorsRAJA< PERMUTATION, POLICY > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.pointer();
}



INDEX_TYPE const SERIAL_SIZE = (2 << 18) - 87;

#if defined(USE_OPENMP)
INDEX_TYPE const OMP_SIZE = (2 << 22) - 87;
#endif

#if defined(USE_CUDA)
constexpr INDEX_TYPE CUDA_SIZE = (2 << 24) - 87;
#endif

void registerBenchmarks()
{
  // Register the native benchmarks.
  forEachArg( []( auto permutation )
  {
    using PERMUTATION = decltype( permutation );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, fortranArrayNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, fortranViewNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, fortranSliceNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, subscriptArrayNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, subscriptViewNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, subscriptSliceNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, tensorAbstractionArrayNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, tensorAbstractionViewNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, tensorAbstractionSliceNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, RAJAViewNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, pointerNative, PERMUTATION );
  },
              RAJA::PERM_IJK {}
              , RAJA::PERM_KJI {}
              );

  // Register the RAJA benchmarks.
  forEachArg( []( auto tuple )
  {
    INDEX_TYPE const size = std::get< 0 >( tuple );
    using PERMUTATION = std::tuple_element_t< 1, decltype( tuple ) >;
    using POLICY = std::tuple_element_t< 2, decltype( tuple ) >;
    REGISTER_BENCHMARK_TEMPLATE( { size }, fortranViewRAJA, PERMUTATION, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( { size }, fortranSliceRAJA, PERMUTATION, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( { size }, subscriptViewRAJA, PERMUTATION, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( { size }, subscriptSliceRAJA, PERMUTATION, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( { size }, tensorAbstractionViewRAJA, PERMUTATION, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( { size }, tensorAbstractionSliceRAJA, PERMUTATION, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( { size }, RAJAViewRAJA, PERMUTATION, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( { size }, pointerRAJA, PERMUTATION, POLICY );
  },
              std::make_tuple( SERIAL_SIZE, RAJA::PERM_IJK {}, serialPolicy {} )
              , std::make_tuple( SERIAL_SIZE, RAJA::PERM_KJI {}, serialPolicy {} )
  #if defined(USE_OPENMP)
              , std::make_tuple( OMP_SIZE, RAJA::PERM_IJK {}, parallelHostPolicy {} )
              , std::make_tuple( OMP_SIZE, RAJA::PERM_KJI {}, parallelHostPolicy {} )
  #endif
  #if defined(USE_CUDA)
              , std::make_tuple( CUDA_SIZE, RAJA::PERM_IJK {}, parallelDevicePolicy< THREADS_PER_BLOCK > {} )
              , std::make_tuple( CUDA_SIZE, RAJA::PERM_KJI {}, parallelDevicePolicy< THREADS_PER_BLOCK > {} )
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
  LVARRAY_LOG( "Serial problems of size ( " << LvArray::benchmarking::SERIAL_SIZE << ", 3, 3 )." );

#if defined(USE_OPENMP)
  LVARRAY_LOG( "OMP problems of size ( " << LvArray::benchmarking::OMP_SIZE << ", 3, 3 )." );
#endif

#if defined(USE_CUDA)
  LVARRAY_LOG( "CUDA problems of size ( " << LvArray::benchmarking::CUDA_SIZE << ", 3, 3 )." );
#endif

  ::benchmark::RunSpecifiedBenchmarks();

  return LvArray::benchmarking::verifyResults( LvArray::benchmarking::resultsMap );
}
