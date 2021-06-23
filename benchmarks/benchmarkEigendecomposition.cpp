/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "benchmarkHelpers.hpp"
#include "benchmarkEigendecompositionKernels.hpp"

// TPL includes
#include <benchmark/benchmark.h>

// System includes
#include <utility>

namespace LvArray
{
namespace benchmarking
{

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename M, typename PERM_2D, typename PERM_3D, typename POLICY >
void eigenvalues( benchmark::State & state )
{
  Eigendecomposition< M {}, PERM_2D, PERM_3D, POLICY > const kernels( state );
  kernels.eigenvalues();
}
///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename M, typename PERM_2D, typename PERM_3D, typename POLICY >
void eigenvectors( benchmark::State & state )
{
  Eigendecomposition< M {}, PERM_2D, PERM_3D, POLICY > const kernels( state );
  kernels.eigenvectors();
}

INDEX_TYPE const SERIAL_SIZE_2x2 = (2 << 22) - 87;
INDEX_TYPE const SERIAL_SIZE_3x3 = (2 << 19) - 87;

#if defined(RAJA_ENABLE_OPENMP)
INDEX_TYPE const OMP_SIZE_2x2 = (2 << 24) - 87;
INDEX_TYPE const OMP_SIZE_3x3 = (2 << 23) - 87;
#endif

// The non Array benchmarks could be run without chai, but then what's the point.
#if defined(LVARRAY_USE_CUDA) && defined(LVARRAY_USE_CHAI)
constexpr INDEX_TYPE CUDA_SIZE_2x2 = (2 << 24) - 87;
constexpr INDEX_TYPE CUDA_SIZE_3x3 = (2 << 24) - 87;
#endif

void registerBenchmarks()
{
  // Register the RAJA benchmarks.
  typeManipulation::forEachArg( []( auto tuple )
  {
    INDEX_TYPE const size = std::get< 0 >( tuple );
    using M = std::tuple_element_t< 1, decltype( tuple ) >;
    using PERM_2D = std::tuple_element_t< 2, decltype( tuple ) >;
    using PERM_3D = std::tuple_element_t< 3, decltype( tuple ) >;
    using POLICY = std::tuple_element_t< 4, decltype( tuple ) >;

    REGISTER_BENCHMARK_TEMPLATE( { size }, eigenvalues, M, PERM_2D, PERM_3D, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( { size }, eigenvectors, M, PERM_2D, PERM_3D, POLICY );

  }, std::make_tuple( SERIAL_SIZE_2x2, std::integral_constant< int, 2 > {}, RAJA::PERM_IJ {}, RAJA::PERM_IJK {}, serialPolicy {} )
                                , std::make_tuple( SERIAL_SIZE_2x2, std::integral_constant< int, 2 > {}, RAJA::PERM_JI {}, RAJA::PERM_KJI {}, serialPolicy {} )
                                , std::make_tuple( SERIAL_SIZE_3x3, std::integral_constant< int, 3 > {}, RAJA::PERM_IJ {}, RAJA::PERM_IJK {}, serialPolicy {} )
                                , std::make_tuple( SERIAL_SIZE_3x3, std::integral_constant< int, 3 > {}, RAJA::PERM_JI {}, RAJA::PERM_KJI {}, serialPolicy {} )
  #if defined(RAJA_ENABLE_OPENMP)
                                , std::make_tuple( OMP_SIZE_2x2, std::integral_constant< int, 2 > {}, RAJA::PERM_IJ {}, RAJA::PERM_IJK {}, parallelHostPolicy {} )
                                , std::make_tuple( OMP_SIZE_2x2, std::integral_constant< int, 2 > {}, RAJA::PERM_JI {}, RAJA::PERM_KJI {}, parallelHostPolicy {} )
                                , std::make_tuple( OMP_SIZE_3x3, std::integral_constant< int, 3 > {}, RAJA::PERM_IJ {}, RAJA::PERM_IJK {}, parallelHostPolicy {} )
                                , std::make_tuple( OMP_SIZE_3x3, std::integral_constant< int, 3 > {}, RAJA::PERM_JI {}, RAJA::PERM_KJI {}, parallelHostPolicy {} )
  #endif
  #if defined(LVARRAY_USE_CUDA) && defined(LVARRAY_USE_CHAI)
                                , std::make_tuple( CUDA_SIZE_2x2, std::integral_constant< int, 2 > {}, RAJA::PERM_IJ {}, RAJA::PERM_IJK {}, parallelDevicePolicy< THREADS_PER_BLOCK > {} )
                                , std::make_tuple( CUDA_SIZE_2x2, std::integral_constant< int, 2 > {}, RAJA::PERM_JI {}, RAJA::PERM_KJI {}, parallelDevicePolicy< THREADS_PER_BLOCK > {} )
                                , std::make_tuple( CUDA_SIZE_3x3, std::integral_constant< int, 3 > {}, RAJA::PERM_IJ {}, RAJA::PERM_IJK {}, parallelDevicePolicy< THREADS_PER_BLOCK > {} )
                                , std::make_tuple( CUDA_SIZE_3x3, std::integral_constant< int, 3 > {}, RAJA::PERM_JI {}, RAJA::PERM_KJI {}, parallelDevicePolicy< THREADS_PER_BLOCK > {} )
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

  LVARRAY_LOG( "Matrices of type = " << LvArray::system::demangleType< LvArray::benchmarking::VALUE_TYPE >() );
  LVARRAY_LOG( "Results of type = " << LvArray::system::demangleType< LvArray::benchmarking::FLOAT >() );
  LVARRAY_LOG( "Serial number of 2x2 matrices = " << LvArray::benchmarking::SERIAL_SIZE_2x2 );
  LVARRAY_LOG( "Serial number of 3x3 matrices = " << LvArray::benchmarking::SERIAL_SIZE_3x3 );

#if defined(RAJA_ENABLE_OPENMP)
  LVARRAY_LOG( "OMP number of 2x2 matrices = " << LvArray::benchmarking::OMP_SIZE_2x2 );
  LVARRAY_LOG( "OMP number of 3x3 matrices = " << LvArray::benchmarking::OMP_SIZE_3x3 );
#endif

#if defined(LVARRAY_USE_CUDA) && defined(LVARRAY_USE_CHAI)
  LVARRAY_LOG( "CUDA number of 2x2 matrices = " << LvArray::benchmarking::CUDA_SIZE_2x2 );
  LVARRAY_LOG( "CUDA number of 3x3 matrices = " << LvArray::benchmarking::CUDA_SIZE_3x3 );
#endif

  ::benchmark::RunSpecifiedBenchmarks();

  return 0;
}
