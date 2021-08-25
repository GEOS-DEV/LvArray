/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

#pragma once

// Source includes
#include "../unitTests/testUtils.hpp"
#include "Array.hpp"
#include "ArrayOfArrays.hpp"
#include "system.hpp"

// System includes
#include <random>
#include <chrono>
#include <unordered_map>

#if defined(LVARRAY_USE_CALIPER)
  #include <caliper/cali.h>
#define CALI_CXX_MARK_PRETTY_FUNCTION cali::Function __cali_ann ## __func__( __PRETTY_FUNCTION__ )
#else
#define CALI_CXX_MARK_PRETTY_FUNCTION
#define CALI_CXX_MARK_SCOPE( string )
#endif

namespace LvArray
{

using namespace testing;


#if defined(LVARRAY_USE_CHAI)
static_assert( std::is_same< DEFAULT_BUFFER< int >, ChaiBuffer< int > >::value,
               "The default buffer should be ChaiBuffer when chai is enabled." );
#endif

namespace benchmarking
{

namespace internal
{

template< typename T >
std::string typeToString( T const & )
{ return LvArray::system::demangleType< T >(); }

inline std::string typeToString( RAJA::PERM_I const & ) { return "RAJA::PERM_I"; }

inline std::string typeToString( RAJA::PERM_IJ const & ) { return "RAJA::PERM_IJ"; }
inline std::string typeToString( RAJA::PERM_JI const & ) { return "RAJA::PERM_JI"; }

inline std::string typeToString( RAJA::PERM_IJK const & ) { return "RAJA::PERM_IJK"; }
inline std::string typeToString( RAJA::PERM_KJI const & ) { return "RAJA::PERM_KJI"; }

} // namespace internal

#define ACCESS_IJ( N, M, i, j ) M * i + j
#define ACCESS_JI( N, M, i, j ) N * j + i

#define ACCESS_IJK( N, M, P, i, j, k ) M * P * i + P * j + k
#define ACCESS_KJI( N, M, P, i, j, k ) M * N * k + N * j + i

using INDEX_TYPE = std::ptrdiff_t;

template< typename T, typename PERMUTATION >
using ArrayT = LvArray::Array< T, typeManipulation::getDimension< PERMUTATION >, PERMUTATION, INDEX_TYPE, DEFAULT_BUFFER >;

template< typename T, typename PERMUTATION >
using ArrayViewT = LvArray::ArrayView< T,
typeManipulation::getDimension< PERMUTATION >,
typeManipulation::getStrideOneDimension( PERMUTATION {} ),
INDEX_TYPE,
DEFAULT_BUFFER >;

template< typename T, typename PERMUTATION >
using ArraySliceT = LvArray::ArraySlice< T,
typeManipulation::getDimension< PERMUTATION >,
typeManipulation::getStrideOneDimension( PERMUTATION {} ),
INDEX_TYPE >;

template< typename T, typename PERMUTATION >
using RajaView = RAJA::View< T,
RAJA::Layout< typeManipulation::getDimension< PERMUTATION >,
INDEX_TYPE,
typeManipulation::getStrideOneDimension( PERMUTATION {} ) >>;

template< typename T >
using ArrayOfArraysT = ArrayOfArrays< T, INDEX_TYPE, DEFAULT_BUFFER >;

template< typename T, bool CONST_SIZES >
using ArrayOfArraysViewT = ArrayOfArraysView< T, INDEX_TYPE const, CONST_SIZES, DEFAULT_BUFFER >;

template< typename ARG, typename ... ARGS >
std::string typeListToString()
{
  std::string string = internal::typeToString( ARG {} );

  typeManipulation::forEachArg( [&string] ( auto const & arg )
  {
    string += ", " + internal::typeToString( arg );
  }, ARGS {} ... );

  return string;
}



#define REGISTER_BENCHMARK( args, func ) \
  { \
    ::benchmark::RegisterBenchmark( STRINGIZE( func ), func ) \
      ->Args( args ) \
      ->UseRealTime() \
      ->ComputeStatistics( "min", []( std::vector< double > const & times ) \
{ return *std::min_element( times.begin(), times.end() ); } ) \
      ->ComputeStatistics( "max", []( std::vector< double > const & times ) \
{ return *std::max_element( times.begin(), times.end() ); } ); \
  }


#define REGISTER_BENCHMARK_TEMPLATE( args, func, ... ) \
  { \
    std::string functionName = STRINGIZE( func ) "< "; \
    functionName += typeListToString< __VA_ARGS__ >() + " >"; \
    ::benchmark::RegisterBenchmark( functionName.c_str(), func< __VA_ARGS__ > ) \
      ->Args( args ) \
      ->UseRealTime() \
      ->ComputeStatistics( "min", []( std::vector< double > const & times ) \
{ return *std::min_element( times.begin(), times.end() ); } ) \
      ->ComputeStatistics( "max", []( std::vector< double > const & times ) \
{ return *std::max_element( times.begin(), times.end() ); } ); \
  }


#define WRAP( ... ) __VA_ARGS__


inline std::uint_fast64_t getSeed()
{
  auto const curTime = std::chrono::high_resolution_clock::now().time_since_epoch();
  static std::uint_fast64_t const seed = std::chrono::duration_cast< std::chrono::nanoseconds >( curTime ).count();
  return seed;
}


template< typename T, int NDIM, int USD >
void initialize( ArraySlice< T, NDIM, USD, INDEX_TYPE > const slice, int & iter )
{
  ++iter;
  std::mt19937_64 gen( iter * getSeed() );

  // This is used to generate a random real number between -100, 100.
  // std::uniform_real_distribution is not used because it is really slow on Lassen.
  std::uniform_int_distribution< std::int64_t > dis( -1e18, 1e18 );

  forValuesInSlice( slice, [&dis, &gen]( T & value )
  {
    double const valueFP = dis( gen ) / 1e16;
    value = valueFP;
  } );
}


template< typename T, typename PERMUTATION >
RajaView< T, PERMUTATION > makeRajaView( ArrayT< T, PERMUTATION > const & array )
{
  constexpr int NDIM = typeManipulation::getDimension< PERMUTATION >;
  std::array< INDEX_TYPE, NDIM > sizes;

  for( int i = 0; i < NDIM; ++i )
  {
    sizes[ i ] = array.dims()[ i ];
  }

  constexpr std::array< camp::idx_t, NDIM > const permutation = RAJA::as_array< PERMUTATION >::get();
  return RajaView< T, PERMUTATION >( array.data(), RAJA::make_permuted_layout( sizes, permutation ) );
}


template< typename T, typename PERMUTATION >
INDEX_TYPE reduce( ArrayT< T, PERMUTATION > const & array )
{
  T result = 0;

  forValuesInSlice( array.toSlice(),
                    [&result]( T const & value )
  {
    result += value;
  } );

  return result;
}

template< typename PERM >
void constructStructuredElementToNodeMap( ArrayT< INDEX_TYPE, PERM > & elementToNodeMap,
                                          INDEX_TYPE const nx,
                                          INDEX_TYPE const ny,
                                          INDEX_TYPE const nz )
{
  elementToNodeMap.resize( nx * ny * nz, 8 );

  INDEX_TYPE const elemJp = nx;
  INDEX_TYPE const elemKp = elemJp * ny;

  INDEX_TYPE const nodeJp = nx + 1;
  INDEX_TYPE const nodeKp = nodeJp * ( ny + 1 );

  // Populate the element to node map.
  for( INDEX_TYPE i = 0; i < nx; ++i )
  {
    for( INDEX_TYPE j = 0; j < ny; ++j )
    {
      for( INDEX_TYPE k = 0; k < nz; ++k )
      {
        INDEX_TYPE const elemIndex = i + elemJp * j + elemKp * k;
        INDEX_TYPE const firstNodeIndex = i + nodeJp * j + nodeKp * k;
        elementToNodeMap( elemIndex, 0 ) = firstNodeIndex;
        elementToNodeMap( elemIndex, 1 ) = firstNodeIndex + 1;
        elementToNodeMap( elemIndex, 2 ) = firstNodeIndex + 1 + nodeJp;
        elementToNodeMap( elemIndex, 3 ) = firstNodeIndex + nodeJp;
        elementToNodeMap( elemIndex, 4 ) = firstNodeIndex + nodeKp;
        elementToNodeMap( elemIndex, 5 ) = firstNodeIndex + nodeKp + 1;
        elementToNodeMap( elemIndex, 6 ) = firstNodeIndex + nodeKp + 1 + nodeJp;
        elementToNodeMap( elemIndex, 7 ) = firstNodeIndex + nodeKp + nodeJp;
      }
    }
  }
}



template< typename T, unsigned long N >
using ResultsMap = std::map< std::array< INDEX_TYPE, N >, std::map< std::string, T > >;

template< typename T >
std::enable_if_t< std::is_integral< T >::value, bool >
equal( T const a, T const b )
{ return a == b; }

template< typename T >
std::enable_if_t< std::is_floating_point< T >::value, bool >
equal( T const a, T const b )
{
  T const diff = std::abs( a - b );
  return diff < 1e-15 || diff / a < 1e-10;
}


template< typename T, unsigned long N >
void registerResult( ResultsMap< T, N > & resultsMap,
                     std::array< INDEX_TYPE, N > const & args,
                     T const result,
                     std::string const & functionName )
{
  auto const ret = resultsMap[ args ].insert( { std::string( functionName ), result } );
  if( !ret.second && !equal( result, ret.first->second ) )
  {
    LVARRAY_LOG( "### " << functionName );
    LVARRAY_LOG( "###\t Produced different results than the previous run! "
                 << result << " != " << ret.first->second );
  }
}


template< typename T, unsigned long N >
inline int verifyResults( ResultsMap< T, N > const & benchmarkResults )
{
  bool success = true;
  for( auto const & run : benchmarkResults )
  {
    std::array< INDEX_TYPE, N > const & args = run.first;
    std::map< std::string, T > const & results = run.second;
    T const firstResult = results.begin()->second;

    bool allResultsEqual = true;
    for( auto const & funcAndResult : results )
    {
      allResultsEqual = allResultsEqual && equal( funcAndResult.second, firstResult );
    }

    if( !allResultsEqual )
    {
      success = false;

      std::cout << "### The benchmarks produced different results with arguments ";
      std::cout << args[ 0 ];
      for( unsigned long i = 1; i < N; ++i )
      {
        std::cout << ", " << args[ i ];
      }
      std::cout << std::endl;

      std::map< T, std::vector< std::string > > functionsByResult;

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
