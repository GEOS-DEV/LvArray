/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "benchmarkArrayOfArraysReduceKernels.hpp"

// TPL includes
#include <benchmark/benchmark.h>


namespace LvArray
{
namespace benchmarking
{

ResultsMap< VALUE_TYPE, 1 > resultsMap;

void reduceFortranArray( benchmark::State & state )
{
  ArrayOfArraysReduce kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.reduceFortranArray();
}

void reduceSubscriptArray( benchmark::State & state )
{
  ArrayOfArraysReduce kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.reduceSubscriptArray();
}

void reduceFortranView( benchmark::State & state )
{
  ArrayOfArraysReduce kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.reduceFortranView();
}

void reduceSubscriptView( benchmark::State & state )
{
  ArrayOfArraysReduce kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.reduceSubscriptView();
}

void reduceVector( benchmark::State & state )
{
  ArrayOfArraysReduce kernels( state, __PRETTY_FUNCTION__, resultsMap );
  kernels.reduceVector();
}

INDEX_TYPE const SERIAL_SIZE = (2 << 10) + 573;

void registerBenchmarks()
{
  // Register the native benchmarks.
  REGISTER_BENCHMARK( { SERIAL_SIZE }, reduceFortranArray );
  REGISTER_BENCHMARK( { SERIAL_SIZE }, reduceSubscriptArray );
  REGISTER_BENCHMARK( { SERIAL_SIZE }, reduceFortranView );
  REGISTER_BENCHMARK( { SERIAL_SIZE }, reduceSubscriptView );
  REGISTER_BENCHMARK( { SERIAL_SIZE }, reduceVector );
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

  ::benchmark::RunSpecifiedBenchmarks();

  return LvArray::benchmarking::verifyResults( LvArray::benchmarking::resultsMap );
}
