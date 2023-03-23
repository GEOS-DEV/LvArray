/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file input.hpp
 * @brief Contains functions for creating objects from strings.
 */

#pragma once

// Source includes
#include "Array.hpp"

// System includes
#include <string>
#include <iostream>

namespace LvArray
{

/**
 * @brief Contains functions for filling array objects from strings.
 */
namespace input
{
namespace internal
{

/**
 * @struct StringToArrayHelper
 * @brief A helper struct to recursively read an istringstream into an array.
 */
template< typename T, typename INDEX_TYPE >
struct StringToArrayHelper
{

  /**
   * @brief function to skip ',' delimiters in a istringstream
   * @param inputStream The istringstream to operate on.
   */
  static void skipDelimiters( std::istringstream & inputStream )
  {
    while( inputStream.peek() == ' ' )
    {
      inputStream.ignore();
    }
  }

  /**
   * @brief Reads a value of the array from the stream
   * @param arrayValue A reference to the array value to read in
   * @param inputStream the stream to read a value from
   */
  template< int NDIM >
  static void Read( T & arrayValue,
                    INDEX_TYPE const *,
                    std::istringstream & inputStream )
  {
    inputStream >> arrayValue;

    LVARRAY_THROW_IF( inputStream.fail(),
                      "Invalid value of type " << typeid(T).name() << " in: " <<
                      ( inputStream.eof()  ?  "" : inputStream.str().substr( inputStream.tellg() ) ),
                      std::invalid_argument );
  }

  /**
   * @brief Recursively read values from an istringstream into an array.
   * @param arraySlice The arraySlice that provides the interface to write data into the array.
   * @param dims The dimensions of the array.
   * @param inputStream The stream to read from.
   */
  template< int NDIM, int USD >
  static void
  Read( ArraySlice< T, NDIM, USD, INDEX_TYPE > const arraySlice,
        INDEX_TYPE const * const dims,
        std::istringstream & inputStream )
  {
    LVARRAY_THROW_IF( inputStream.peek() != '{', "Opening '{' not found for input array: "<<inputStream.str(),
                      std::invalid_argument );
    inputStream.ignore();

    for( int i=0; i<(*dims); ++i )
    {
      skipDelimiters( inputStream );
      Read< NDIM-1 >( arraySlice[i], dims+1, inputStream );
    }

    skipDelimiters( inputStream );
    LVARRAY_THROW_IF( inputStream.peek() != '}', "Closing '}' not found for input array: "<<inputStream.str(),
                      std::invalid_argument );
    inputStream.ignore();
  }
};

} // namespace internal

/**
 * @brief This function reads the contents of a string into an Array.
 * @param array Reference to the array that will receive the contents of the input.
 * @param valueString The string that contains the data to read into @p array.
 *
 * @details The contents of @p valueString are parsed and placed into @p array. @p array is resized to
 *   allow space for receiving the contents of @p valueString. The required notation for array values in
 *   @p valueString are similar to the requirements of initialization of a standard c-array:
 * @code
 *     Array<T,1> --> "{ val[0], val[1], val[2], ... }"
 *     Array<T,2> --> "{ { val[0][0], val[0][1], val[0][2], ... },
 *                       { val[1][0], val[1][1], val[1][2], ... }, ... } "
 * @endcode
 *
 * @note * A null initializer is allowed via "{}". All values must be delimited with a ','.
 *   All spaces are stripped prior to processing, please don't use tabs for anything.
 */
template< typename T,
          int NDIM,
          typename PERMUTATION,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
static void stringToArray( Array< T, NDIM, PERMUTATION, INDEX_TYPE, BUFFER_TYPE > & array,
                           std::string valueString )
{
  // Check to make sure there are no space delimited values. The assumption is anything that is not
  // a '{' or '}' or ',' or ' ' is part of a value. Loope over the string and keep track of whether
  // or not there is a value on the left of the char. If there is a value to the left, with a space
  // on the left, and we run into another value, then this means that there is a space delimited
  // entry.
  {
    bool valueOnLeft = false;
    bool spaceOnLeft = false;

    for( char const c : valueString )
    {
      if( c != '{' && c != ',' && c != '}' && c!=' ' )
      {
        if( valueOnLeft && spaceOnLeft )
        {
          LVARRAY_THROW( "Array value sequence specified without ',' delimiter: "<<valueString,
                         std::invalid_argument );
        }
      }

      // If a open/close brace or ',' delimiter, then there is neither a value or space
      // to the left of subsequent characters.
      if( c == '{' || c == ',' || c == '}' )
      {
        valueOnLeft = false;
        spaceOnLeft = false;
      }
      else // if the first check fails, then either c is a space or value
      {
        // if it is a value, then set the valueOnLeft flag to true for subsequent c
        if( c != ' ' )
        {
          valueOnLeft = true;
          spaceOnLeft = false;
        }
        else // if it is a space, then set spaceOnLeft to true for subsequent c
        {
          spaceOnLeft = true;
        }
      }
    }
  }

  // erase all spaces from input string to simplify parsing
  valueString.erase( std::remove( valueString.begin(), valueString.end(), ' ' ), valueString.end());

  // allow for a null input
  if( valueString=="{}" )
  {
    array.clear();
    return;
  }

  // checking for various formatting errors
  LVARRAY_THROW_IF( valueString.find( "}{" ) != std::string::npos,
                    "Sub arrays not separated by ',' delimiter: "<<valueString,
                    std::invalid_argument );

  LVARRAY_THROW_IF( valueString[0]!='{',
                    "First non-space character of input string for an array must be '{'. Given string is: \n"<<valueString,
                    std::invalid_argument );

  size_t const numOpen = std::count( valueString.begin(), valueString.end(), '{' );
  size_t const numClose = std::count( valueString.begin(), valueString.end(), '}' );

  LVARRAY_THROW_IF( numOpen != numClose,
                    "Number of opening '{' not equal to number of '}' in processing of string for filling"
                    " an Array. Given string is: \n"<<valueString,
                    std::invalid_argument );


  // after allowing for the null input, disallow a sub-array null input
  LVARRAY_THROW_IF( valueString.find( "{}" )!=std::string::npos,
                    "Cannot have an empty sub-dimension of an array, i.e. { { 0, 1}, {} }. "
                    "The input is"<<valueString,
                    std::invalid_argument );

  // get the number of dimensions from the number of { characters that begin the input string
  int const ndims = LvArray::integerConversion< int >( valueString.find_first_not_of( '{' ));
  LVARRAY_THROW_IF( ndims!=NDIM,
                    "number of dimensions in string ("<<ndims<<
                    ") does not match dimensions of array("<<NDIM<<
                    "). String is:/n"<<valueString,
                    std::invalid_argument );


  // now get the number of dimensions, and the size of each dimension.

  // use dimLevel to track the current dimension we are parsing
  int dimLevel = -1;

  // dims is the dimensions that get set the first diving down.
  INDEX_TYPE dims[NDIM] = {0};

  // currentDims is used to track the dimensions for subsequent dives down the dimensions.
  INDEX_TYPE currentDims[NDIM] = {0};

  // flag to see if the dims value has been set for a given dimension
  bool dimSet[NDIM] = {false};

  for( int i=0; i<NDIM; ++i )
  {
    dims[i]=1;
    currentDims[i] = 1;
  }

  char lastChar = 0;
  for( size_t charCount = 0; charCount<valueString.size(); ++charCount )
  {
    char const c = valueString[charCount];
    // this had better be true for the first char...we had a check for this. This is why we can
    // set dimLevel = -1 to start.
    if( c=='{' )
    {
      ++dimLevel;
    }
    else if( c=='}' )
    {
      LVARRAY_THROW_IF( lastChar==',',
                        "character '}' follows '"<<lastChar<<"'. Closing brace must follow an array value.",
                        std::invalid_argument );

      // } means that we are closing a dimension. That means we know the size of this dimLevel
      dimSet[dimLevel] = true;
      LVARRAY_THROW_IF( dims[dimLevel]!=currentDims[dimLevel],
                        "Dimension "<<dimLevel<<" is inconsistent across the expression. "
                                                "The first set value of the dimension is "<<dims[dimLevel]<<
                        " while the current value of the dimension is"<<currentDims[dimLevel]<<
                        ". The values that have been parsed prior to the error are:\n"<<
                        valueString.substr( 0, charCount+1 ),
                        std::invalid_argument );

      // reset currentDims and drop dimLevel for post-closure parsing
      currentDims[dimLevel] = 1;
      --dimLevel;
      LVARRAY_THROW_IF( dimLevel<0 && charCount<(valueString.size()-1),
                        "In parsing the input string, the current dimension of the array has dropped "
                        "below 0. This means that there are more '}' than '{' at some point in the"
                        " parsing. The values that have been parsed prior to the error are:\n"<<
                        valueString.substr( 0, charCount+1 ),
                        std::invalid_argument );

    }
    else if( c==',' ) // we are counting the dimension sizes because there is a delimiter.
    {
      LVARRAY_THROW_IF( lastChar=='{' || lastChar==',',
                        "character of ',' follows '"<<lastChar<<"'. Comma must follow an array value.",
                        std::invalid_argument );
      if( dimSet[dimLevel]==false )
      {
        ++(dims[dimLevel]);
      }
      ++(currentDims[dimLevel]);
    }
    lastChar = c;
  }
  LVARRAY_THROW_IF( dimLevel!=-1,
                    "Expression fails to close all '{' with a corresponding '}'. Check your input:"<<
                    valueString,
                    std::invalid_argument );

  array.resize( NDIM, dims );


  // we need to replace our ',' with ' ' for reading in an array of strings, otherwise the
  // stringstream::operator>> will grab the ','
  std::replace( valueString.begin(), valueString.end(), ',', ' ' );

  // we also need to add a ' ' in front of any '}' otherwise the
  // stringstream::operator>> will grab the }
  for( std::string::size_type a=0; a<valueString.size(); ++a )
  {
    if( valueString[a] == '}' )
    {
      valueString.insert( a, " " );
      ++a;
    }
  }
  std::istringstream strstream( valueString );
  // this recursively reads the values from the stringstream
  internal::StringToArrayHelper< T, INDEX_TYPE >::Read( array.toSlice(), array.dims(), strstream );
}

} // namespace input
} // namespace LvArray
