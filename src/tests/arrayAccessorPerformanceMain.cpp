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
 * main.cpp
 *
 *  Created on: Jul 30, 2016
 *      Author: rrsettgast
 */


#include "arrayAccessorPerformance.hpp"
#include <chrono>
#include <thread>

using namespace LvArray;


int main( int /*argc*/, char* argv[] )
{

  integer_t seed = time( nullptr );

  const integer_t num_i = std::stoi( argv[1] );
  const integer_t num_k = std::stoi( argv[2] );
  const integer_t num_j = std::stoi( argv[3] );
  const integer_t ITERATIONS = std::stoi( argv[4] );
  const integer_t seedmod = std::stoi( argv[5] );

  const integer_t output = std::stoi( argv[6] );

  //***************************************************************************
  //***** Setup Arrays ********************************************************
  //***************************************************************************

  double * const restrict A  = new double[ static_cast< std::size_t >(num_i*num_k)];
  double * const restrict B  = new double[ static_cast< std::size_t >(num_k*num_j)];
  double * const restrict C1D          = new double[ static_cast< std::size_t >(num_i*num_j)];
  double * const restrict C1D_restrict = new double[ static_cast< std::size_t >(num_i*num_j)];

//  integer_t lengthsA[] = { num_i, num_k };
//  integer_t lengthsB[] = { num_k, num_j };
//  integer_t lengthsC[] = { num_i, num_j };
//
//  integer_t stridesA[] = { num_k, 1};
//  integer_t stridesB[] = { num_j, 1};
//  integer_t stridesC[] = { num_j, 1};


  Array< double, 2 > ArrayA( num_i, num_k );
  Array< double, 2 > ArrayB( num_k, num_j );
  Array< double, 2 > ArrayC_SquareNFC( num_i, num_j );
  Array< double, 2 > ArrayC_SquarePBV( num_i, num_j );
  Array< double, 2 > ArrayC_SquarePBR( num_i, num_j );
  Array< double, 2 > ArrayC_ParenNFC( num_i, num_j );
  Array< double, 2 > ArrayC_ParenPBV( num_i, num_j );
  Array< double, 2 > ArrayC_ParenPBR( num_i, num_j );



  srand( static_cast< unsigned int >(seed * seedmod) );

  for( integer_t i = 0 ; i < num_i ; ++i )
    for( integer_t k = 0 ; k < num_k ; ++k )
    {
      A[i*num_k+k] = rand();
      ArrayA[i][k] = A[i*num_k+k];
    }

  for( integer_t k = 0 ; k < num_k ; ++k )
    for( integer_t j = 0 ; j < num_j ; ++j )
    {
      B[k*num_j+j] = rand();
      ArrayB[k][j] = B[k*num_j+j];
    }

  for( integer_t i = 0 ; i < num_i ; ++i )
  {
    for( integer_t j = 0 ; j < num_j ; ++j )
    {
      C1D[i*num_j+j] = 0.0;
      C1D_restrict[i*num_j+j] = 0.0;

      ArrayC_SquareNFC[i][j] = 0;
      ArrayC_SquarePBV[i][j] = 0;
      ArrayC_SquarePBR[i][j] = 0;

      ArrayC_ParenNFC[i][j] = 0;
      ArrayC_ParenPBV[i][j] = 0;
      ArrayC_ParenPBR[i][j] = 0;
    }
  }

  ArrayView< double, 2 > & arrayViewA = ArrayA;
  ArrayView< double, 2 > & arrayViewB = ArrayB;
  ArrayView< double, 2 > & arrayViewC_SquareNFC = ArrayC_SquareNFC;
  ArrayView< double, 2 > & arrayViewC_SquarePBV = ArrayC_SquarePBV;
  ArrayView< double, 2 > & arrayViewC_SquarePBR = ArrayC_SquarePBR;
  ArrayView< double, 2 > & arrayViewC_ParenNFC = ArrayC_ParenNFC;
  ArrayView< double, 2 > & arrayViewC_ParenPBV = ArrayC_ParenPBV;
  ArrayView< double, 2 > & arrayViewC_ParenPBR = ArrayC_ParenPBR;



  std::this_thread::sleep_for( std::chrono::milliseconds( 100 ));
  double runTime_Direct1dAccess  = MatrixMultiply_1D ( num_i, num_j, num_k, ITERATIONS, A, B, C1D );

  std::this_thread::sleep_for( std::chrono::milliseconds( 100 ));
  double runTime_Direct1dRestrict = MatrixMultiply_1Dr( num_i, num_j, num_k, ITERATIONS, A, B, C1D_restrict );



  std::this_thread::sleep_for( std::chrono::milliseconds( 100 ));
  double startTime = GetTimeMs64();
  for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter )
  {
    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {
        for( integer_t k = 0 ; k < num_k ; ++k )
        {
          arrayViewC_SquareNFC[i][j] += arrayViewA[i][k] * arrayViewB[k][j] + 3.1415 * arrayViewA[i][k] + 1.61803 * arrayViewB[k][j];
        }
      }
    }
  }
  double endTime = GetTimeMs64();
  double runTime_arrayViewSquareNFC = ( endTime - startTime ) / 1000.0;


  std::this_thread::sleep_for( std::chrono::milliseconds( 100 ));
  double runTime_arrayViewSquarePBV = MatrixMultiply_2D_accessor( num_i, num_j, num_k, ITERATIONS, arrayViewA, arrayViewB, arrayViewC_SquarePBV );

  std::this_thread::sleep_for( std::chrono::milliseconds( 100 ));
  double runTime_arrayViewSquarePBR = MatrixMultiply_2D_accessorRef( num_i, num_j, num_k, ITERATIONS, arrayViewA, arrayViewB, arrayViewC_SquarePBR );



  std::this_thread::sleep_for( std::chrono::milliseconds( 100 ));
  startTime = GetTimeMs64();
  for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter )
  {
    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {
        for( integer_t k = 0 ; k < num_k ; ++k )
        {
          arrayViewC_ParenNFC( i, j ) += ArrayA( i, k ) * ArrayB( k, j ) + 3.1415 * ArrayA( i, k ) + 1.61803 * ArrayB( k, j );
        }
      }
    }
  }
  endTime = GetTimeMs64();
  double runTime_arrayViewParenNFC =( endTime - startTime ) / 1000.0;

  std::this_thread::sleep_for( std::chrono::milliseconds( 100 ));
  double runTime_arrayViewParenPBV = MatrixMultiply_2D_accessorPBV2( num_i, num_j, num_k, ITERATIONS, arrayViewA, arrayViewB, arrayViewC_ParenPBV );


  std::this_thread::sleep_for( std::chrono::milliseconds( 100 ));
  double runTime_arrayViewParenPBR = MatrixMultiply_2D_accessorRef2( num_i, num_j, num_k, ITERATIONS, arrayViewA, arrayViewB, arrayViewC_ParenPBR );



  double minRunTime = 1.0e99;
  minRunTime = std::min( runTime_Direct1dAccess, runTime_Direct1dRestrict );
  minRunTime = std::min( minRunTime, runTime_arrayViewSquareNFC );
  minRunTime = std::min( minRunTime, runTime_arrayViewSquarePBV );
  minRunTime = std::min( minRunTime, runTime_arrayViewSquarePBR );
  minRunTime = std::min( minRunTime, runTime_arrayViewParenNFC );
  minRunTime = std::min( minRunTime, runTime_arrayViewParenPBV );
  minRunTime = std::min( minRunTime, runTime_arrayViewParenPBR );

  if( output > 2 )
  {

    double error_ArraySquareNFC = 0.0;
    double error_ArraySquarePBV = 0.0;
    double error_ArraySquarePBR = 0.0;

    double error_ArrayParenNFC = 0.0;
    double error_ArrayParenPBV = 0.0;
    double error_ArrayParenPBR = 0.0;

    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {

        error_ArraySquareNFC += pow( C1D[i*num_j+j] - ArrayC_SquareNFC[i][j], 2 );
        error_ArraySquarePBV += pow( C1D[i*num_j+j] - ArrayC_SquarePBV[i][j], 2 );
        error_ArraySquarePBR += pow( C1D[i*num_j+j] - ArrayC_SquarePBR[i][j], 2 );

        error_ArrayParenNFC += pow( C1D[i*num_j+j] - ArrayC_ParenNFC[i][j], 2 );
        error_ArrayParenPBV += pow( C1D[i*num_j+j] - ArrayC_ParenPBV[i][j], 2 );
        error_ArrayParenPBR += pow( C1D[i*num_j+j] - ArrayC_ParenPBR[i][j], 2 );

      }
    }

    GEOS_LOG( "error_ArraySquareNFC = "<<error_ArraySquareNFC );
    GEOS_LOG( "error_ArraySquarePBV = "<<error_ArraySquarePBV );
    GEOS_LOG( "error_ArraySquarePBR = "<<error_ArraySquarePBR );

    GEOS_LOG( "error_ArrayParenNFC = "<<error_ArrayParenNFC );
    GEOS_LOG( "error_ArrayParenPBV = "<<error_ArrayParenPBV );
    GEOS_LOG( "error_ArrayParenPBR = "<<error_ArrayParenPBR );


  }

  if( output > 1 )
  {
    printf( "1d array                             : %8.3f, %8.2f\n", runTime_Direct1dAccess, runTime_Direct1dAccess/minRunTime );
    printf( "1d array restrict                    : %8.3f, %8.2f\n", runTime_Direct1dRestrict, runTime_Direct1dRestrict / minRunTime );

    printf( "Array[] nfc                   : %8.3f, %8.2f\n", runTime_arrayViewSquareNFC, runTime_arrayViewSquareNFC / minRunTime );
    printf( "Array[] pbv                   : %8.3f, %8.2f\n", runTime_arrayViewSquarePBV, runTime_arrayViewSquarePBV / minRunTime );
    printf( "Array[] pbr                   : %8.3f, %8.2f\n", runTime_arrayViewSquarePBR, runTime_arrayViewSquarePBR / minRunTime );

    printf( "Array() nfc                   : %8.3f, %8.2f\n", runTime_arrayViewParenNFC, runTime_arrayViewParenNFC / minRunTime );
    printf( "Array() pbv                   : %8.3f, %8.2f\n", runTime_arrayViewParenPBV, runTime_arrayViewParenPBV / minRunTime );
    printf( "Array() pbr                   : %8.3f, %8.2f\n", runTime_arrayViewParenPBR, runTime_arrayViewParenPBR / minRunTime );

  }


  if( output == 1 )
  {
    printf( "%8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n",
            runTime_Direct1dAccess,
            runTime_Direct1dRestrict,

            runTime_arrayViewSquareNFC,
            runTime_arrayViewSquarePBV,
            runTime_arrayViewSquarePBR,

            runTime_arrayViewParenNFC,
            runTime_arrayViewParenPBV,
            runTime_arrayViewParenPBR
            );
  }
  return 0;
}
