/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "benchmarkArrayOfArraysNodeToElementMapConstructionKernels.hpp"

// TPL includes
#include <benchmark/benchmark.h>


namespace LvArray
{
namespace benchmarking
{

void vector( benchmark::State & state )
{
  CALI_CXX_MARK_PRETTY_FUNCTION;
  NaiveNodeToElemMapConstruction kernels( state, __PRETTY_FUNCTION__ );
  kernels.vector();
}

void naive( benchmark::State & state )
{
  CALI_CXX_MARK_PRETTY_FUNCTION;
  NaiveNodeToElemMapConstruction kernels( state, __PRETTY_FUNCTION__ );
  kernels.naive();
}

template< typename POLICY >
void overAllocation( benchmark::State & state )
{
  CALI_CXX_MARK_PRETTY_FUNCTION;
  NodeToElemMapConstruction< POLICY > kernels( state, __PRETTY_FUNCTION__ );
  kernels.overAllocation();
}

template< typename POLICY >
void resizeFromCapacities( benchmark::State & state )
{
  CALI_CXX_MARK_PRETTY_FUNCTION;
  NodeToElemMapConstruction< POLICY > kernels( state, __PRETTY_FUNCTION__ );
  kernels.resizeFromCapacities();
}

INDEX_TYPE const NX = 200;
INDEX_TYPE const NY = 200;
INDEX_TYPE const NZ = 200;

void registerBenchmarks()
{
  REGISTER_BENCHMARK( WRAP( { NX, NY, NZ } ), vector );
  REGISTER_BENCHMARK( WRAP( { 30, 30, 30 } ), naive );

  // Register the RAJA benchmarks.
  typeManipulation::forEachArg( []( auto tuple )
  {
    INDEX_TYPE const nx = std::get< 0 >( tuple );
    INDEX_TYPE const ny = std::get< 1 >( tuple );
    INDEX_TYPE const nz = std::get< 2 >( tuple );
    using POLICY = std::tuple_element_t< 3, decltype( tuple ) >;

    REGISTER_BENCHMARK_TEMPLATE( WRAP( { nx, ny, nz } ), overAllocation, POLICY );
    REGISTER_BENCHMARK_TEMPLATE( WRAP( { nx, ny, nz } ), resizeFromCapacities, POLICY );
  }, std::make_tuple( NX, NY, NZ, serialPolicy {} )
  #if defined(RAJA_ENABLE_OPENMP)
                                , std::make_tuple( NX, NY, NZ, parallelHostPolicy {} )
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

  LVARRAY_LOG( "INDEX_TYPE = " << LvArray::system::demangleType< LvArray::benchmarking::INDEX_TYPE >() );

  LVARRAY_LOG( "Mesh of size ( " << LvArray::benchmarking::NX << ", " << LvArray::benchmarking::NY <<
               ", " << LvArray::benchmarking::NZ << " )." );

  ::benchmark::RunSpecifiedBenchmarks();
}
