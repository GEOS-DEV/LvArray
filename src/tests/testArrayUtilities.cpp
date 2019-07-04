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
 * Copyright (c) 2015, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */

#include <gtest/gtest.h>
#include<string>
#include "ArrayUtilities.hpp"

const char IGNORE_OUTPUT[] = ".*";

using namespace cxx_utilities;
using namespace std;

TEST( testArrayUtilities, stringToArrayErrors )
{
  string input ;

  // This should work
  {
    input = " { { {0,1,2},{3,4,5} }, { {6,7,8},{9,10,11} }, { {12,13,14},{15,16,17} } , { {18,19,20},{21,22,23} } }";
    LvArray::Array<int,3,int> array;
    stringToArray( array, input );
  }

  {
    input = " { 10 1 } ";
    LvArray::Array<int,1,int> array;
    EXPECT_DEATH_IF_SUPPORTED( stringToArray( array, input ), IGNORE_OUTPUT );
  }

  {
    input = " { { 1, 2 }{ 3, 4 } } ";
    LvArray::Array<int,2,int> array;
    EXPECT_DEATH_IF_SUPPORTED( stringToArray( array, input ), IGNORE_OUTPUT );
  }

  // This should fail the num('{')==num('}') test
  {
    input = " { { {0,1,2},{3,4,5} }, { {6,7,8},{9,10,11} }, { {12,13,14},{15,16,17} } , { {18,19,20},{21,22,23} } ";
    LvArray::Array<int,3,int> array;
    EXPECT_DEATH_IF_SUPPORTED( stringToArray( array, input ), IGNORE_OUTPUT );
  }
  {
    input = " { { {0,1,2},{3,4,5} }, { {6,7,8},{9,10,11} }, { {12,13,14},{15,16,17}  , { {18,19,20},{21,22,23} } }";
    LvArray::Array<int,3,int> array;
    EXPECT_DEATH_IF_SUPPORTED( stringToArray( array, input ), IGNORE_OUTPUT );
  }
  {
    input = " { { {0,1,2},{3,4,5} }, { {6,7,8},{9,10,11} }, { {12,13,14,{15,16,17} } , { {18,19,20},{21,22,23} } }";
    LvArray::Array<int,3,int> array;
    EXPECT_DEATH_IF_SUPPORTED( stringToArray( array, input ), IGNORE_OUTPUT );
  }
  {
    input = " { { {0,1,2},{3,4,5} }, { {6,7,8,{9,10,11} }, { {12,13,14},{15,16,17} } , { {18,19,20},{21,22,23} } }";
    LvArray::Array<int,3,int> array;
    EXPECT_DEATH_IF_SUPPORTED( stringToArray( array, input ), IGNORE_OUTPUT );
  }
  {
    input = " { { 0,1,2},{3,4,5} }, { {6,7,8},{9,10,11} }, { {12,13,14},{15,16,17} } , { {18,19,20},{21,22,23} } }";
    LvArray::Array<int,3,int> array;
    EXPECT_DEATH_IF_SUPPORTED( stringToArray( array, input ), IGNORE_OUTPUT );
  }
  {
    input = "  { {0,1,2},{3,4,5} }, { {6,7,8},{9,10,11} }, { {12,13,14},{15,16,17} } , { {18,19,20},{21,22,23} } ";
    LvArray::Array<int,3,int> array;
    EXPECT_DEATH_IF_SUPPORTED( stringToArray( array, input ), IGNORE_OUTPUT );
  }

  {
    input = " { { {,1,2},{3,4,5} }, { {6,7,8},{9,10,11} }, { {12,13,14},{15,16,17} } , { {18,19,20},{21,22,23} } }";
    LvArray::Array<int,3,int> array;
    EXPECT_DEATH_IF_SUPPORTED( stringToArray( array, input ), IGNORE_OUTPUT );
  }

  {
    input = " { { {},{3,4,5} }, { {6,7,8},{9,10,11} }, { {12,13,14},{15,16,17} } , { {18,19,20},{21,22,23} } }";
    LvArray::Array<int,3,int> array;
    EXPECT_DEATH_IF_SUPPORTED( stringToArray( array, input ), IGNORE_OUTPUT );
  }
  {
    input = " { { {0,1,2}}{ } }";
    LvArray::Array<int,3,int> array;
    EXPECT_DEATH_IF_SUPPORTED( stringToArray( array, input ), IGNORE_OUTPUT );
  }


}

TEST( testArrayUtilities, stringToArray3d )
{
//  string input = " { { {0,1,2},{3,4,5} }, { {6,7,8},{9,10,11} }, { {12,13,14},{15,16,17} } , { {18,19,20},{21,22,23} } }";
  string input;
  input += "{ ";
  int numI = 4;
  int numJ = 5;
  int numK = 3;
  for( int i=0 ; i<4 ; ++i )
  {
    input += "{ ";
    for( int j=0 ; j<5 ; ++j )
    {
      input += "{ ";
      for( int k=0 ; k<3 ; ++k )
      {
        input += std::to_string(i*2+j*3+k*4);
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

  LvArray::Array<int,3,int> array;
  stringToArray( array, input );

  ASSERT_EQ( array.size(0), numI );
  ASSERT_EQ( array.size(1), numJ );
  ASSERT_EQ( array.size(2), numK );

  for( int i=0 ; i<array.size(0) ; ++i )
  {
    for( int j=0 ; j<array.size(1) ; ++j )
    {
      for( int k=0 ; k<array.size(2) ; ++k )
      {
        ASSERT_EQ( array[i][j][k], i*2+j*3+k*4 );
      }
    }
  }
}



TEST( testArrayUtilities, arrayToString )
{
  LvArray::Array<int,3,int> array;
  array.resize( 2, 4, 3 );

  for( int i=0 ; i<2; ++i )
  {
    for( int j=0 ; j<4; ++j )
    {
      for( int k=0 ; k<3; ++k )
      {
        array[i][j][k] = i*100 + j*10 + k ;
      }
    }
  }
  string stringy = arrayToString( array );
  ASSERT_EQ( stringy, "{ { { 0, 1, 2 }, { 10, 11, 12 }, { 20, 21, 22 }, { 30, 31, 32 } },"
                       " { { 100, 101, 102 }, { 110, 111, 112 }, { 120, 121, 122 }, { 130, 131, 132 } } }" );

}

int main( int argc, char* argv[] )
{
  logger::InitializeLogger();

  int result = 0;
  testing::InitGoogleTest( &argc, argv );
  result = RUN_ALL_TESTS();

  logger::FinalizeLogger();

#ifdef USE_CHAI
  chai::ArrayManager::finalize();
#endif

  return result;
}

