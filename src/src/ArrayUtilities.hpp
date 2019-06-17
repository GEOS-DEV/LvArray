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

/*
 * StringUtilities.hpp
 *
 *  Created on: Nov 12, 2014
 *      Author: rrsettgast
 */

#ifndef ARRAYUTILITIES_HPP_
#define ARRAYUTILITIES_HPP_


#include <string>
#include <vector>
#include <iostream>
#include "Array.hpp"

namespace cxx_utilities
{
template< typename T, int NDIM, typename INDEX_TYPE >
void equateStlVector( LvArray::Array<T, NDIM, INDEX_TYPE> & lhs, std::vector<T> const & rhs )
{
  static_assert( NDIM == 1, "Dimension must be 1" );
  GEOS_ERROR_IF( lhs.size() != static_cast<INDEX_TYPE>(rhs.size()), "Size mismatch" );
  for( unsigned int a=0 ; a<rhs.size() ; ++a )
  {
    lhs[a] = rhs[a];
  }
}

template<typename T>
void equateStlVector( T & lhs, std::vector<T> const & rhs )
{
  GEOS_ERROR_IF( rhs.size() > 1, "Size should not be greater than 1 but is: " << rhs.size());
  if( rhs.size() == 1 )
  {
    lhs = rhs[0];
  }
}



template< typename T, int NDIM, typename INDEX_TYPE, typename DATA_VECTOR_TYPE >
static void stringToArray(  LvArray::Array<T, NDIM, INDEX_TYPE, DATA_VECTOR_TYPE > & array,
                            std::string valueString )
{
  // erase all spaces from input string
  valueString.erase(std::remove(valueString.begin(), valueString.end(), ' '), valueString.end());

  size_t openPos = 0;
  size_t closePos = 0;

  GEOS_ERROR_IF( valueString[0]!='{',
                 "First non-space character of input string for an array must be {" );

  size_t const numOpen = std::count( valueString.begin(), valueString.end(), '{' );
  size_t const numClose = std::count( valueString.begin(), valueString.end(), '}' );

  GEOS_ERROR_IF( numOpen != numClose,
                 "Number of opening { not equal to number of } in processing of string for filling"
                 " an Array. Given string is: \n"<<valueString);

  // allow for a null input
  if( valueString=="{}" )
  {
    array.clear();
    return;
  }

  // aftre allowing for the null input, disallow a sub-array null input
  GEOS_ERROR_IF( valueString.find("{}")!=std::string::npos,
                 "Cannot have an empty sub-dimension of an array, i.e. { { 0, 1}, {} }. "
                 "The input is"<<valueString );

  // get the number of dimensions from the number of { characters that begin the input string
  int const ndims = integer_conversion<int>(valueString.find_first_not_of('{'));
  GEOS_ERROR_IF( ndims!=NDIM,
                 "number of dimensions in string ("<<ndims<<
                 ") does not match dimensions of array("<<NDIM<<
                 "). String is:/n"<<valueString );

  int dimLevel = -1;
  INDEX_TYPE dims[NDIM] = {0};
  INDEX_TYPE currentDims[NDIM] = {0};
  for( int i=0 ; i<NDIM ; ++i )
  {
    dims[i]=1;
    currentDims[i] = 1;
  }
  bool dimSet[NDIM] = {false};

  char lastChar = 0;

  for( size_t charCount = 0; charCount<valueString.size() ; ++charCount )
  {
    char const c = valueString[charCount];
    if( c=='{')
    {
      ++dimLevel;
    }
    else if( c=='}')
    {
      dimSet[dimLevel] = true;
      GEOS_ERROR_IF( dims[dimLevel]!=currentDims[dimLevel],
                     "Dimension "<<dimLevel<<" is inconsistent across the expression. "
                     "The first set value of the dimension is "<<dims[dimLevel]<<
                     " while the current value of the dimension is"<<currentDims[dimLevel]<<
                     ". The values that have been parsed prior to the error are:\n"<<
                     valueString.substr(0,charCount+1) );
      currentDims[dimLevel] = 1;
      --dimLevel;
      GEOS_ERROR_IF( dimLevel<0 && charCount<(valueString.size()-1),
                     "In parsing the input string, the current dimension of the array has dropped "
                     "below 0. This means that there are more '}' than '{' at some point in the"
                     " parsing. The values that have been parsed prior to the error are:\n"<<
                     valueString.substr(0,charCount+1) );

    }
    else if( c==',' )
    {
      GEOS_ERROR_IF( lastChar=='{' || lastChar==',',
                     "character of ',' follows '"<<lastChar<<"'. Comma must follow an array value.");
      if( dimSet[dimLevel]==false )
      {
        ++(dims[dimLevel]);
      }
      ++(currentDims[dimLevel]);

    }

    lastChar = c;
  }
  GEOS_ERROR_IF( dimLevel!=-1,
                 "Expression fails to close all '{' with a corresponding '}'. Check your input:"<<
                 valueString );


  array.resize( NDIM, dims );

  T * arrayData = array.data();

  // In order to use the stringstream facility to read in values of a Array<string>,
  // we need to replace all {}, with spaces.
  std::replace( valueString.begin(), valueString.end(), '{', ' ' );
  std::replace( valueString.begin(), valueString.end(), '}', ' ' );
  std::replace( valueString.begin(), valueString.end(), ',', ' ' );
  std::istringstream strstream(valueString);

  // iterate through the stream and insert values into array in a linear fashion. This will be
  // incorrect if we ever have Array with a permuted index capability.
  while( strstream )
  {
    int c = strstream.peek();

    if( c== ' ' )
    {
      strstream.ignore();
    }
    else
    {
      strstream>>*(arrayData++);
    }
  }
}

template< typename T, typename INDEX_TYPE >
struct arrayToStringHelper
{
  template< int NDIM >
  static
  typename std::enable_if< NDIM==0, void >::type
  dimExpansion( T const * const data,
                INDEX_TYPE const * const dims,
                INDEX_TYPE const * const strides,
                std::string & output )
  {
    for( INDEX_TYPE i=0 ; i<dims[0] ; ++i )
    {
      output += std::to_string( data[i] );
      if( i < dims[0] - 1 )
      {
        output += ", ";
      }
    }
  }

  template< int NDIM >
  static
  typename std::enable_if< NDIM!=0, void >::type
  dimExpansion( T const * const data,
                INDEX_TYPE const * const dims,
                INDEX_TYPE const * const strides,
                std::string & output )
  {
    for( INDEX_TYPE i=0 ; i<dims[0] ; ++i )
    {
      output += "{ ";
      dimExpansion<NDIM-1>( &(data[ i*strides[0] ] ), dims+1, strides+1, output );
      output += " }";
      if( i < dims[0] - 1 )
      {
        output += ", ";
      }

    }
  }

};



template< typename T, int NDIM, typename INDEX_TYPE, typename DATA_VECTOR_TYPE >
static std::string arrayToString(  LvArray::ArrayView<T, NDIM, INDEX_TYPE, DATA_VECTOR_TYPE > const & array )
{
  std::string output;
  output.clear();

  output += "{ ";
  arrayToStringHelper<T, INDEX_TYPE>::template dimExpansion<NDIM-1>( array.data(),
                                                                   array.dims(),
                                                                   array.strides(),
                                                                   output );

  output += " }";

  return output;
}


}

#endif /* STRINGUTILITIES_HPP_ */
