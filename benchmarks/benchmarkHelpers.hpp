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
/* *UNCRUSTIFY-OFF* */
#pragma once

// Source includes
#include "benchmarkCommon.hpp"
#include "../unitTests/testUtils.hpp"

// System includes
#include <random>
#include <chrono>
#include <unordered_map>

namespace LvArray
{

using namespace testing;

namespace benchmarking
{

#define DECLARE_BENCHMARK( func, args, iter ) \
  BENCHMARK( func ) \
    ->Args( args ) \
    ->Repetitions( iter ) \
    ->UseRealTime() \
    ->ComputeStatistics( "min", \
                         []( std::vector< double > const & times ) \
  { \
    return *std::min_element( times.begin(), times.end() ); \
  } \
                         ) \
    ->ComputeStatistics( "max", \
                         []( std::vector< double > const & times ) \
  { \
    return *std::max_element( times.begin(), times.end() ); \
  } \
                         )


#define DECLARE_BENCHMARK_TEMPLATE( func, T, args, iter ) \
  BENCHMARK_TEMPLATE( func, T ) \
    ->Args( args ) \
    ->Repetitions( iter ) \
    ->UseRealTime() \
    ->ComputeStatistics( "min", \
                         []( std::vector< double > const & times ) \
  { \
    return *std::min_element( times.begin(), times.end() ); \
  } \
                         ) \
    ->ComputeStatistics( "max", \
                         []( std::vector< double > const & times ) \
  { \
    return *std::max_element( times.begin(), times.end() ); \
  } \
                         )


#define WRAP( ... ) __VA_ARGS__


#define FOUR_BENCHMARK_TEMPLATES_ONE_TYPE( B0, B1, B2, B3, T, ARGS, REPS ) \
  DECLARE_BENCHMARK_TEMPLATE( B0, WRAP( T ), WRAP( ARGS ), REPS ); \
  DECLARE_BENCHMARK_TEMPLATE( B1, WRAP( T ), WRAP( ARGS ), REPS ); \
  DECLARE_BENCHMARK_TEMPLATE( B2, WRAP( T ), WRAP( ARGS ), REPS ); \
  DECLARE_BENCHMARK_TEMPLATE( B3, WRAP( T ), WRAP( ARGS ), REPS )


#define FIVE_BENCHMARK_TEMPLATES_ONE_TYPE( B0, B1, B2, B3, B4, T, ARGS, REPS ) \
  FOUR_BENCHMARK_TEMPLATES_ONE_TYPE( B0, B1, B2, B3, WRAP( T ), WRAP( ARGS ), REPS ); \
  DECLARE_BENCHMARK_TEMPLATE( B4, WRAP( T ), WRAP( ARGS ), REPS )


#define FOUR_BENCHMARK_TEMPLATES_TWO_TYPES( B0, B1, B2, B3, T0, T1, ARGS, REPS ) \
  FOUR_BENCHMARK_TEMPLATES_ONE_TYPE( B0, B1, B2, B3, WRAP( std::pair< T0, T1 > ), WRAP( ARGS ), REPS )


#define FIVE_BENCHMARK_TEMPLATES_TWO_TYPES( B0, B1, B2, B3, B4, T0, T1, ARGS, REPS ) \
  FIVE_BENCHMARK_TEMPLATES_ONE_TYPE( B0, B1, B2, B3, B4, WRAP( std::pair< T0, T1 > ), WRAP( ARGS ), REPS )


inline std::uint_fast64_t getSeed()
{
  auto const curTime = std::chrono::high_resolution_clock::now().time_since_epoch();
  static std::uint_fast64_t const seed = std::chrono::duration_cast< std::chrono::nanoseconds >( curTime ).count();
  return seed;
}


template< typename PERMUTATION >
void initialize( Array< VALUE_TYPE, PERMUTATION > const & array, int & iter )
{
  ++iter;
  std::mt19937_64 gen( iter * getSeed() );

  std::uniform_int_distribution< VALUE_TYPE > dis( -100, 100 );

  forValuesInSlice( array.toSlice(),
                    [&dis, &gen]( VALUE_TYPE & value )
      {
        value = dis( gen );
      }
                    );
}


template< typename PERMUTATION >
RajaView< VALUE_TYPE, PERMUTATION > makeRajaView( Array< VALUE_TYPE, PERMUTATION > const & array )
{
  constexpr int NDIM = getDimension( PERMUTATION {} );
  std::array< INDEX_TYPE, NDIM > sizes;

  for( int i = 0 ; i < NDIM ; ++i )
  {
    sizes[ i ] = array.dims()[ i ];
  }

  constexpr std::array< camp::idx_t, NDIM > const permutation = RAJA::as_array< PERMUTATION >::get();
  return RajaView< VALUE_TYPE, PERMUTATION >( array.data(), RAJA::make_permuted_layout( sizes, permutation ) );
}


template< typename PERMUTATION >
INDEX_TYPE reduce( Array< VALUE_TYPE, PERMUTATION > const & array )
{
  VALUE_TYPE result = 0;

  forValuesInSlice( array.toSlice(),
                    [&result]( VALUE_TYPE const & value )
      {
        result += value;
      }
                    );

  return result;
}


template< unsigned long N >
using ResultsMap = std::map< std::array< INDEX_TYPE, N >, std::map< std::string, VALUE_TYPE > >;


template< unsigned long N >
void registerResult( ResultsMap< N > & resultsMap,
                     std::array< INDEX_TYPE, N > const & args,
                     VALUE_TYPE const result,
                     char const * const functionName )
{
  auto const ret = resultsMap[ args ].insert( { std::string( functionName ), result } );
  if( !ret.second && result != ret.first->second )
  {
    LVARRAY_LOG( "### " << functionName );
    LVARRAY_LOG( "###\t Produced different results than the previous run! "
                 << result << " != " << ret.first->second );
  }
}


template< unsigned long N >
inline int verifyResults( ResultsMap< N > const & benchmarkResults )
{
  bool success = true;
  for( auto const & run : benchmarkResults )
  {
    std::array< INDEX_TYPE, N > const & args = run.first;
    std::map< std::string, VALUE_TYPE > const & results = run.second;
    VALUE_TYPE const firstResult = results.begin()->second;

    bool allResultsEqual = true;
    for( auto const & funcAndResult : results )
    {
      allResultsEqual = allResultsEqual && funcAndResult.second == firstResult;
    }

    if( !allResultsEqual )
    {
      success = false;

      std::cout << "### The benchmarks produced different results with arguments ";
      std::cout << args[ 0 ];
      for( unsigned long i = 1 ; i < N ; ++i )
      {
        std::cout << ", " << args[ i ];
      }
      std::cout << std::endl;

      std::map< VALUE_TYPE, std::vector< std::string > > functionsByResult;

      for( auto const & funcAndResult : results )
      {
        functionsByResult[ funcAndResult.second ].push_back( funcAndResult.first );
      }

      for( auto const & resultAndNames : functionsByResult )
      {
        for( std::string const & functionName : resultAndNames.second )
        {
          LVARRAY_LOG( "### " << functionName );
        }

        LVARRAY_LOG( "###\t resulted in " << resultAndNames.first );
      }
    }
  }

  if( !success )
  {
    return 1;
  }

  if( benchmarkResults.size() != 1 )
  {
    LVARRAY_LOG( "### Cannot compare the results of every benchmark because they were given different arguments." );
    LVARRAY_LOG( "### However all benchmarks with the same arguments gave the same answers." );
  }
  else
  {
    LVARRAY_LOG( "### All benchmarks produced the same answers." );
  }

  return 0;
}


} // namespace benchmarking
} // namespace LvArray
/* *UNCRUSITIFY-ON* */
