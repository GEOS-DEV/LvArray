/*
 * Copyright (c) 2020, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "benchmarkMatrixMatrixKernels.hpp"

// TPL includes
#include <benchmark/benchmark.h>

// System includes
#include <utility>

namespace LvArray
{
namespace benchmarking
{

ResultsMap< VALUE_TYPE, 3 > resultsMap;

template< typename PERMUTATION >
void fortranArrayNative( benchmark::State & state )
{
  MatrixMatrixNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.fortranArray();
}

template< typename PERMUTATION >
void fortranViewNative( benchmark::State & state )
{
  MatrixMatrixNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.fortranView();
}

template< typename PERMUTATION >
void fortranSliceNative( benchmark::State & state )
{
  MatrixMatrixNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.fortranSlice();
}

template< typename PERMUTATION >
void subscriptArrayNative( benchmark::State & state )
{
  MatrixMatrixNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.subscriptArray();
}

template< typename PERMUTATION >
void subscriptViewNative( benchmark::State & state )
{
  MatrixMatrixNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.subscriptView();
}

template< typename PERMUTATION >
void subscriptSliceNative( benchmark::State & state )
{
  MatrixMatrixNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.subscriptSlice();
}
template< typename PERMUTATION >
void RAJAViewNative( benchmark::State & state )
{
  MatrixMatrixNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.RAJAView();
}

template< typename PERMUTATION >
void pointerNative( benchmark::State & state )
{
  MatrixMatrixNative< PERMUTATION > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.pointer();
}

template< typename PERMUTATION, typename POLICY >
void fortranViewRAJA( benchmark::State & state )
{
  MatrixMatrixRAJA< PERMUTATION, POLICY > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.fortranView();
}

template< typename PERMUTATION, typename POLICY >
void fortranSliceRAJA( benchmark::State & state )
{
  MatrixMatrixRAJA< PERMUTATION, POLICY > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.fortranSlice();
}

template< typename PERMUTATION, typename POLICY >
void subscriptViewRAJA( benchmark::State & state )
{
  MatrixMatrixRAJA< PERMUTATION, POLICY > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.subscriptView();
}

template< typename PERMUTATION, typename POLICY >
void subscriptSliceRAJA( benchmark::State & state )
{
  MatrixMatrixRAJA< PERMUTATION, POLICY > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.subscriptSlice();
}
template< typename PERMUTATION, typename POLICY >
void RAJAViewRAJA( benchmark::State & state )
{
  MatrixMatrixRAJA< PERMUTATION, POLICY > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.RAJAView();
}

template< typename PERMUTATION, typename POLICY >
void pointerRAJA( benchmark::State & state )
{
  MatrixMatrixRAJA< PERMUTATION, POLICY > const kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.pointer();
}

INDEX_TYPE const SERIAL_N = (2 << 7) + 73;
INDEX_TYPE const SERIAL_L = (2 << 7) - 71;
INDEX_TYPE const SERIAL_M = (2 << 7) - 3;

#if defined(USE_OPENMP)
INDEX_TYPE const OMP_N = SERIAL_N;
INDEX_TYPE const OMP_L = SERIAL_L;
INDEX_TYPE const OMP_M = SERIAL_M;
#endif

#if defined(USE_CUDA) && defined(USE_CHAI)
INDEX_TYPE const CUDA_N = SERIAL_N;
INDEX_TYPE const CUDA_L = SERIAL_L;
INDEX_TYPE const CUDA_M = SERIAL_M;
#endif

void registerBenchmarks()
{
  // Register the native benchmarks.
  typeManipulation::forEachArg( []( auto permutation )
  {
    using PERMUTATION = decltype( permutation );
    REGISTER_BENCHMARK_TEMPLATE( WRAP( { SERIAL_N, SERIAL_L, SERIAL_M } ), fortranArrayNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( WRAP( { SERIAL_N, SERIAL_L, SERIAL_M } ), fortranViewNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( WRAP( { SERIAL_N, SERIAL_L, SERIAL_M } ), fortranSliceNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( WRAP( { SERIAL_N, SERIAL_L, SERIAL_M } ), subscriptArrayNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( WRAP( { SERIAL_N, SERIAL_L, SERIAL_M } ), subscriptViewNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( WRAP( { SERIAL_N, SERIAL_L, SERIAL_M } ), subscriptSliceNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( WRAP( { SERIAL_N, SERIAL_L, SERIAL_M } ), RAJAViewNative, PERMUTATION );
    REGISTER_BENCHMARK_TEMPLATE( WRAP( { SERIAL_N, SERIAL_L, SERIAL_M } ), pointerNative, PERMUTATION );
  },
                                RAJA::PERM_IJ {}
                                , RAJA::PERM_JI {}
                                );

  // Register the RAJA benchmarks.
  typeManipulation::forEachArg( []( auto tuple )
  {
    INDEX_TYPE const N = std::get< 0 >( tuple );
    INDEX_TYPE const L = std::get< 1 >( tuple );
    INDEX_TYPE const M = std::get< 2 >( tuple );
    using PERMUTATION = std::tuple_element_t< 3, decltype( tuple ) >;
    using POLICY = std::tuple_element_t< 4, decltype( tuple ) >;
    REGISTER_BENCHMARK_TEMPLATE( WRAP( { N, L, M } ), fortranViewRAJA, PERMUTATION, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( WRAP( { N, L, M } ), fortranSliceRAJA, PERMUTATION, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( WRAP( { N, L, M } ), subscriptViewRAJA, PERMUTATION, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( WRAP( { N, L, M } ), subscriptSliceRAJA, PERMUTATION, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( WRAP( { N, L, M } ), RAJAViewRAJA, PERMUTATION, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( WRAP( { N, L, M } ), pointerRAJA, PERMUTATION, POLICY );
  },
                                std::make_tuple( SERIAL_N, SERIAL_L, SERIAL_M, RAJA::PERM_IJ {}, serialPolicy {} )
                                , std::make_tuple( SERIAL_N, SERIAL_L, SERIAL_M, RAJA::PERM_JI {}, serialPolicy {} )
  #if defined(USE_OPENMP)
                                , std::make_tuple( OMP_N, SERIAL_L, OMP_M, RAJA::PERM_IJ {}, parallelHostPolicy {} )
                                , std::make_tuple( OMP_N, SERIAL_L, OMP_M, RAJA::PERM_JI {}, parallelHostPolicy {} )
  #endif
  #if defined(USE_CUDA) && defined(USE_CHAI)
                                , std::make_tuple( CUDA_N, SERIAL_L, CUDA_M, RAJA::PERM_IJ {}, parallelDevicePolicy< THREADS_PER_BLOCK > {} )
                                , std::make_tuple( CUDA_N, SERIAL_L, CUDA_M, RAJA::PERM_JI {}, parallelDevicePolicy< THREADS_PER_BLOCK > {} )
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
  LVARRAY_LOG( "INDEX_TYPE = " << LvArray::system::demangleType< LvArray::benchmarking::INDEX_TYPE >() );

  LVARRAY_LOG( "Serial problems of size ( " << LvArray::benchmarking::SERIAL_N << ", " <<
               LvArray::benchmarking::SERIAL_L << ", " << LvArray::benchmarking::SERIAL_M << " )." );

#if defined(USE_OPENMP)
  LVARRAY_LOG( "OMP problems of size ( " << LvArray::benchmarking::OMP_N << ", " <<
               LvArray::benchmarking::OMP_L << ", " << LvArray::benchmarking::OMP_M << " )." );
#endif

#if defined(USE_CUDA) && defined(USE_CHAI)
  LVARRAY_LOG( "CUDA problems of size ( " << LvArray::benchmarking::CUDA_N << ", " <<
               LvArray::benchmarking::CUDA_L << ", " << LvArray::benchmarking::CUDA_M << " )." );
#endif

  ::benchmark::RunSpecifiedBenchmarks();

  return LvArray::benchmarking::verifyResults( LvArray::benchmarking::resultsMap );
}
