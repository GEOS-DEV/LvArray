/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
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
void tensorAbstractionFortranArrayNative( benchmark::State & state )
{
  ArrayOfR2TensorsNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.tensorAbstractionFortranArray();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION >
void tensorAbstractionFortranViewNative( benchmark::State & state )
{
  ArrayOfR2TensorsNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.tensorAbstractionFortranView();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION >
void tensorAbstractionFortranSliceNative( benchmark::State & state )
{
  ArrayOfR2TensorsNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.tensorAbstractionFortranSlice();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION >
void tensorAbstractionSubscriptArrayNative( benchmark::State & state )
{
  ArrayOfR2TensorsNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.tensorAbstractionSubscriptArray();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION >
void tensorAbstractionSubscriptViewNative( benchmark::State & state )
{
  ArrayOfR2TensorsNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.tensorAbstractionSubscriptView();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION >
void tensorAbstractionSubscriptSliceNative( benchmark::State & state )
{
  ArrayOfR2TensorsNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.tensorAbstractionSubscriptSlice();
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
void tensorAbstractionFortranViewRAJA( benchmark::State & state )
{
  ArrayOfR2TensorsRAJA< PERMUTATION, POLICY > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.tensorAbstractionFortranView();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION, typename POLICY >
void tensorAbstractionFortranSliceRAJA( benchmark::State & state )
{
  ArrayOfR2TensorsRAJA< PERMUTATION, POLICY > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.tensorAbstractionFortranSlice();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION, typename POLICY >
void tensorAbstractionSubscriptViewRAJA( benchmark::State & state )
{
  ArrayOfR2TensorsRAJA< PERMUTATION, POLICY > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.tensorAbstractionSubscriptView();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename PERMUTATION, typename POLICY >
void tensorAbstractionSubscriptSliceRAJA( benchmark::State & state )
{
  ArrayOfR2TensorsRAJA< PERMUTATION, POLICY > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.tensorAbstractionSubscriptSlice();
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

#if defined(RAJA_ENABLE_OPENMP)
INDEX_TYPE const OMP_SIZE = (2 << 22) - 87;
#endif

// The non Array benchmarks could be run without chai, but then what's the point.
#if defined(LVARRAY_USE_CUDA) && defined(LVARRAY_USE_CHAI)
constexpr INDEX_TYPE CUDA_SIZE = (2 << 24) - 87;
#endif

void registerBenchmarks()
{
  // Register the native benchmarks.
  typeManipulation::forEachArg( []( auto permutation )
  {
    using PERMUTATION = decltype( permutation );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, fortranArrayNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, fortranViewNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, fortranSliceNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, subscriptArrayNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, subscriptViewNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, subscriptSliceNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, tensorAbstractionFortranArrayNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, tensorAbstractionFortranViewNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, tensorAbstractionFortranSliceNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, tensorAbstractionSubscriptArrayNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, tensorAbstractionSubscriptViewNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, tensorAbstractionSubscriptSliceNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, RAJAViewNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( { SERIAL_SIZE }, pointerNative, PERMUTATION );
  },
                                RAJA::PERM_IJK {}
                                , RAJA::PERM_KJI {}
                                );

  // Register the RAJA benchmarks.
  typeManipulation::forEachArg( []( auto tuple )
  {
    INDEX_TYPE const size = std::get< 0 >( tuple );
    using PERMUTATION = std::tuple_element_t< 1, decltype( tuple ) >;
    using POLICY = std::tuple_element_t< 2, decltype( tuple ) >;
    REGISTER_BENCHMARK_TEMPLATE( { size }, fortranViewRAJA, PERMUTATION, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( { size }, fortranSliceRAJA, PERMUTATION, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( { size }, subscriptViewRAJA, PERMUTATION, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( { size }, subscriptSliceRAJA, PERMUTATION, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( { size }, tensorAbstractionFortranViewRAJA, PERMUTATION, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( { size }, tensorAbstractionFortranSliceRAJA, PERMUTATION, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( { size }, tensorAbstractionSubscriptViewRAJA, PERMUTATION, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( { size }, tensorAbstractionSubscriptSliceRAJA, PERMUTATION, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( { size }, RAJAViewRAJA, PERMUTATION, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( { size }, pointerRAJA, PERMUTATION, POLICY );
  },
                                std::make_tuple( SERIAL_SIZE, RAJA::PERM_IJK {}, serialPolicy {} )
                                , std::make_tuple( SERIAL_SIZE, RAJA::PERM_KJI {}, serialPolicy {} )
  #if defined(RAJA_ENABLE_OPENMP)
                                , std::make_tuple( OMP_SIZE, RAJA::PERM_IJK {}, parallelHostPolicy {} )
                                , std::make_tuple( OMP_SIZE, RAJA::PERM_KJI {}, parallelHostPolicy {} )
  #endif
  #if defined(LVARRAY_USE_CUDA) && defined(LVARRAY_USE_CHAI)
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

  LVARRAY_LOG( "VALUE_TYPE = " << LvArray::system::demangleType< LvArray::benchmarking::VALUE_TYPE >() );
  LVARRAY_LOG( "Serial problems of size ( " << LvArray::benchmarking::SERIAL_SIZE << ", 3, 3 )." );

#if defined(RAJA_ENABLE_OPENMP)
  LVARRAY_LOG( "OMP problems of size ( " << LvArray::benchmarking::OMP_SIZE << ", 3, 3 )." );
#endif

#if defined(LVARRAY_USE_CUDA) && defined(LVARRAY_USE_CHAI)
  LVARRAY_LOG( "CUDA problems of size ( " << LvArray::benchmarking::CUDA_SIZE << ", 3, 3 )." );
#endif

  ::benchmark::RunSpecifiedBenchmarks();

  return LvArray::benchmarking::verifyResults( LvArray::benchmarking::resultsMap );
}
