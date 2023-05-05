/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "input.hpp"
#include "output.hpp"
#include "MallocBuffer.hpp"

// TPL includes
#include <gtest/gtest.h>

// System includes
#include <string>

namespace LvArray
{

template< typename T, typename PERMUTATION >
using ArrayT = Array< T, typeManipulation::getDimension< PERMUTATION >, PERMUTATION, std::ptrdiff_t, MallocBuffer >;

TEST( input, stringToArrayErrors )
{
  std::string input;

  // This should work
  {
    input = " { { {0,1,2},{3,4,5} }, { {6,7,8},{9,10,11} }, { {12,13,14},{15,16,17} } , { {18,19,20},{21,22,23} } }";
    ArrayT< int, RAJA::PERM_IJK > array;
    input::stringToArray( array, input );
  }

  {
    input = " { 10 1 } ";
    ArrayT< int, RAJA::PERM_I > array;
    ASSERT_THROW( input::stringToArray( array, input ), std::invalid_argument );
  }

  {
    input = " { { 1, 2 }{ 3, 4 } } ";
    ArrayT< int, RAJA::PERM_IJ > array;
    ASSERT_THROW( input::stringToArray( array, input ), std::invalid_argument );
  }

  // This should fail the num('{')==num('}') test
  {
    input = " { { {0,1,2},{3,4,5} }, { {6,7,8},{9,10,11} }, { {12,13,14},{15,16,17} } , { {18,19,20},{21,22,23} } ";
    ArrayT< int, RAJA::PERM_IJK > array;
    ASSERT_THROW( input::stringToArray( array, input ), std::invalid_argument );
  }
  {
    input = " { { {0,1,2},{3,4,5} }, { {6,7,8},{9,10,11} }, { {12,13,14},{15,16,17}  , { {18,19,20},{21,22,23} } }";
    ArrayT< int, RAJA::PERM_IKJ > array;
    ASSERT_THROW( input::stringToArray( array, input ), std::invalid_argument );
  }
  {
    input = " { { {0,1,2},{3,4,5} }, { {6,7,8},{9,10,11} }, { {12,13,14,{15,16,17} } , { {18,19,20},{21,22,23} } }";
    ArrayT< int, RAJA::PERM_JIK > array;
    ASSERT_THROW( input::stringToArray( array, input ), std::invalid_argument );
  }
  {
    input = " { { {0,1,2},{3,4,5} }, { {6,7,8,{9,10,11} }, { {12,13,14},{15,16,17} } , { {18,19,20},{21,22,23} } }";
    ArrayT< int, RAJA::PERM_JKI > array;
    ASSERT_THROW( input::stringToArray( array, input ), std::invalid_argument );
  }
  {
    input = " { { 0,1,2},{3,4,5} }, { {6,7,8},{9,10,11} }, { {12,13,14},{15,16,17} } , { {18,19,20},{21,22,23} } }";
    ArrayT< int, RAJA::PERM_KIJ > array;
    ASSERT_THROW( input::stringToArray( array, input ), std::invalid_argument );
  }
  {
    input = "  { {0,1,2},{3,4,5} }, { {6,7,8},{9,10,11} }, { {12,13,14},{15,16,17} } , { {18,19,20},{21,22,23} } ";
    ArrayT< int, RAJA::PERM_KJI > array;
    ASSERT_THROW( input::stringToArray( array, input ), std::invalid_argument );
  }

  {
    input = " { { {,1,2},{3,4,5} }, { {6,7,8},{9,10,11} }, { {12,13,14},{15,16,17} } , { {18,19,20},{21,22,23} } }";
    ArrayT< int, RAJA::PERM_IJK > array;
    ASSERT_THROW( input::stringToArray( array, input ), std::invalid_argument );
  }

  {
    input = " { { {},{3,4,5} }, { {6,7,8},{9,10,11} }, { {12,13,14},{15,16,17} } , { {18,19,20},{21,22,23} } }";
    ArrayT< int, RAJA::PERM_IJK > array;
    ASSERT_THROW( input::stringToArray( array, input ), std::invalid_argument );
  }
  {
    input = " { { {0,1,2}}{ } }";
    ArrayT< int, RAJA::PERM_IJK > array;
    ASSERT_THROW( input::stringToArray( array, input ), std::invalid_argument );
  }


}

TEST( input, stringToArray3d )
{
  std::string input;
  input += "{ ";
  int numI = 4;
  int numJ = 5;
  int numK = 3;
  for( int i=0; i<4; ++i )
  {
    input += "{ ";
    for( int j=0; j<5; ++j )
    {
      input += "{ ";
      for( int k=0; k<3; ++k )
      {
        input += std::to_string( i*2+j*3+k*4 );
        if( k<(numK-1) )
        {
          input += " , ";
        }
      }
      input += " }";
      if( j<(numJ-1) )
      {
        input += " , ";
      }
    }
    input += " }";
    if( i<(numI-1) )
    {
      input += " , ";
    }
  }
  input += " }";

  ArrayT< int, RAJA::PERM_JIK > array;
  input::stringToArray( array, input );

  ASSERT_EQ( array.size( 0 ), numI );
  ASSERT_EQ( array.size( 1 ), numJ );
  ASSERT_EQ( array.size( 2 ), numK );

  for( int i=0; i<array.size( 0 ); ++i )
  {
    for( int j=0; j<array.size( 1 ); ++j )
    {
      for( int k=0; k<array.size( 2 ); ++k )
      {
        ASSERT_EQ( array[i][j][k], i*2+j*3+k*4 );
      }
    }
  }
}

TEST( input, arrayToString )
{
  ArrayT< int, RAJA::PERM_IKJ > array( 2, 4, 3 );

  for( int i=0; i<2; ++i )
  {
    for( int j=0; j<4; ++j )
    {
      for( int k=0; k<3; ++k )
      {
        array[i][j][k] = i*100 + j*10 + k;
      }
    }
  }

  std::stringstream ss;
  ss << array;
  ASSERT_EQ( ss.str(), "{ { { 0, 1, 2 }, { 10, 11, 12 }, { 20, 21, 22 }, { 30, 31, 32 } },"
                       " { { 100, 101, 102 }, { 110, 111, 112 }, { 120, 121, 122 }, { 130, 131, 132 } } }" );

}

} // namespace LvArray
