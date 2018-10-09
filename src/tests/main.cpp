/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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


int main( int /*argc*/, char* argv[] )
{
//  int testLengths[] = {2,4,3};
//  GEOS_LOG("stride<3>(testLengths) = "<<stride<3>(testLengths));
//  GEOS_LOG("stride<2>(testLengths) = "<<stride<2>(testLengths+1));
//  GEOS_LOG("stride<1>(testLengths) = "<<stride<1>(testLengths+2));
//
//
//  dimensions<2,4,3> dims;
//  GEOS_LOG("dims.ndim() = "<<dimensions<1,2,3> ::NDIMS());
////  GEOS_LOG("dims.ndim() = "<<dims.ndims());
//  GEOS_LOG("dims.dim0   = "<<dims.DIMENSION<0>());
//  GEOS_LOG("dims.dim1   = "<<dims.DIMENSION<1>());
//  GEOS_LOG("dims.dim2   = "<<dims.DIMENSION<2>());
//
//  GEOS_LOG("dims.STRIDE<0>   = "<<dims.STRIDE<0>());
//  GEOS_LOG("dims.STRIDE<1>   = "<<dims.STRIDE<1>());
//  GEOS_LOG("dims.STRIDE<2>   = "<<dims.STRIDE<2>());

  integer_t seed = time( NULL );

  const integer_t num_i = std::stoi( argv[1] );
  const integer_t num_k = std::stoi( argv[2] );
  const integer_t num_j = std::stoi( argv[3] );
  const integer_t ITERATIONS = std::stoi( argv[4] );
  const integer_t seedmod = std::stoi( argv[5] );

  const integer_t output = std::stoi( argv[6] );

  //***************************************************************************
  //***** Setup Arrays ********************************************************
  //***************************************************************************

  double * const restrict A  = new double[num_i*num_k];
  double * const restrict B  = new double[num_k*num_j];
  double * const restrict C1D          = new double[num_i*num_j];
  double * const restrict C1D_restrict = new double[num_i*num_j];

  double * const restrict ptrC_SquareNFC = new double[num_i*num_j];
  double * const restrict ptrC_SquarePBV = new double[num_i*num_j];
  double * const restrict ptrC_SquarePBR = new double[num_i*num_j];
  double * const restrict ptrC_SquareLAC = new double[num_i*num_j];

  double * const restrict ptrC_ParenNFC = new double[num_i*num_j];
  double * const restrict ptrC_ParenPBV = new double[num_i*num_j];
  double * const restrict ptrC_ParenPBR = new double[num_i*num_j];


  double * const restrict tempC = new double[num_i*num_j];

  integer_t lengthsA[] = { num_i, num_k };
  integer_t lengthsB[] = { num_k, num_j };
  integer_t lengthsC[] = { num_i, num_j };

  integer_t stridesA[] = { num_k, 1};
  integer_t stridesB[] = { num_j, 1};
  integer_t stridesC[] = { num_j, 1};


  ManagedArray<double,2> ManagedArrayA( num_i, num_k );
  ManagedArray<double,2> ManagedArrayB( num_k, num_j );
  ManagedArray<double,2> ManagedArrayC_SquareNFC( num_i, num_j );
  ManagedArray<double,2> ManagedArrayC_SquarePBV( num_i, num_j );
  ManagedArray<double,2> ManagedArrayC_SquarePBR( num_i, num_j );
  ManagedArray<double,2> ManagedArrayC_ParenNFC( num_i, num_j );
  ManagedArray<double,2> ManagedArrayC_ParenPBV( num_i, num_j );
  ManagedArray<double,2> ManagedArrayC_ParenPBR( num_i, num_j );


//
//  double * const restrict ptrA_SelfDescribing  = new double[num_i*num_k +
// maxDim()];
//  double * const restrict ptrB_SelfDescribing  = new double[num_k*num_j +
// maxDim()];
//  double * const restrict ptrC_SelfDescribing  = new double[num_i*num_j +
// maxDim()];
//
//  integer_t * A2_dims = reinterpret_cast<integer_t*>(
// &(ptrA_SelfDescribing[0]) );
//  A2_dims[0] = 2;
//  A2_dims[1] = num_i;
//  A2_dims[2] = num_k;
//  double * ptrAdata_SelfDescribing = &(ptrA_SelfDescribing[maxDim()]);
//
//  integer_t * B2_dims = reinterpret_cast<integer_t*>(
// &(ptrB_SelfDescribing[0]) );
//  B2_dims[0] = 2;
//  B2_dims[1] = num_k;
//  B2_dims[2] = num_j;
//  double * ptrBdata_SelfDescribing = &(ptrB_SelfDescribing[maxDim()]);
//
//
//  integer_t * C2_dims = reinterpret_cast<integer_t*>(
// &(ptrC_SelfDescribing[0]) );
//  C2_dims[0] = 2;
//  C2_dims[1] = num_i;
//  C2_dims[2] = num_j;
//  double * ptrCdata_SelfDescribing = &(ptrC_SelfDescribing[maxDim()]);
//
//
//
  srand( seed * seedmod );

  for( integer_t i = 0 ; i < num_i ; ++i )
    for( integer_t k = 0 ; k < num_k ; ++k )
    {
      A[i*num_k+k] = rand();
      ManagedArrayA[i][k] = A[i*num_k+k];
//      ptrAdata_SelfDescribing[i*num_k+k] = A[i*num_k+k];
    }

  for( integer_t k = 0 ; k < num_k ; ++k )
    for( integer_t j = 0 ; j < num_j ; ++j )
    {
      B[k*num_j+j] = rand();
      ManagedArrayB[k][j] = B[k*num_j+j];
//      ptrBdata_SelfDescribing[k*num_j+j] = B[k*num_j+j];
    }

  for( integer_t i = 0 ; i < num_i ; ++i )
  {
    for( integer_t j = 0 ; j < num_j ; ++j )
    {
      C1D[i*num_j+j] = 0.0;
      C1D_restrict[i*num_j+j] = 0.0;

      ptrC_SquareNFC[i*num_j+j] = 0.0;
      ptrC_SquarePBV[i*num_j+j] = 0.0;
      ptrC_SquarePBR[i*num_j+j] = 0.0;
      ptrC_SquareLAC[i*num_j+j] = 0.0;

      ptrC_ParenNFC[i*num_j+j] = 0.0;
      ptrC_ParenPBV[i*num_j+j] = 0.0;
      ptrC_ParenPBR[i*num_j+j] = 0.0;

      tempC[i*num_j+j] = 0.0;

      ManagedArrayC_SquareNFC[i][j] = 0;
      ManagedArrayC_SquarePBV[i][j] = 0;
      ManagedArrayC_SquarePBR[i][j] = 0;

      ManagedArrayC_ParenNFC[i][j] = 0;
      ManagedArrayC_ParenPBV[i][j] = 0;
      ManagedArrayC_ParenPBR[i][j] = 0;


//      ptrCdata_SelfDescribing[i*num_j+j] = 0.0;
    }
  }



  ArrayView<double,2> accessorA( A, lengthsA, stridesA );
  ArrayView<double,2> accessorB( B, lengthsB, stridesB );
  ArrayView<double,2> accessorC_SquareNFC( ptrC_SquareNFC, lengthsC, stridesC );
  ArrayView<double,2> accessorC_SquarePBV( ptrC_SquarePBV, lengthsC, stridesC );
  ArrayView<double,2> accessorC_SquarePBR( ptrC_SquarePBR, lengthsC, stridesC );

  ArrayView<double,2> accessorC_ParenNFC( ptrC_ParenNFC, lengthsC, stridesC );
  ArrayView<double,2> accessorC_ParenPBV( ptrC_ParenPBV, lengthsC, stridesC );
  ArrayView<double,2> accessorC_ParenPBR( ptrC_ParenPBR, lengthsC, stridesC );


  ArrayView<double,2> arrayViewA(ManagedArrayA);
  ArrayView<double,2> arrayViewB(ManagedArrayB);
  ArrayView<double,2> arrayViewC_SquareNFC( ManagedArrayC_SquareNFC );
  ArrayView<double,2> arrayViewC_SquarePBV( ManagedArrayC_SquarePBV );
  ArrayView<double,2> arrayViewC_SquarePBR( ManagedArrayC_SquarePBR );
  ArrayView<double,2> arrayViewC_ParenNFC( ManagedArrayC_ParenNFC );
  ArrayView<double,2> arrayViewC_ParenPBV( ManagedArrayC_ParenPBV );
  ArrayView<double,2> arrayViewC_ParenPBR( ManagedArrayC_ParenPBR );



  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  double runTime_Direct1dAccess  = MatrixMultiply_1D ( num_i, num_j, num_k, ITERATIONS, A, B, C1D );

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  double runTime_Direct1dRestrict = MatrixMultiply_1Dr( num_i, num_j, num_k, ITERATIONS, A, B, C1D_restrict );



  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  double startTime = GetTimeMs64();
  for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter )
  {
    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {
        for( integer_t k = 0 ; k < num_k ; ++k )
        {
          accessorC_SquareNFC[i][j] += accessorA[i][k] * accessorB[k][j] + 3.1415 * accessorA[i][k] + 1.61803 * accessorB[k][j];;
        }
      }
    }
  }
  double endTime = GetTimeMs64();
  double runTime_SquareNFC = ( endTime - startTime ) / 1000.0;


  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  double runTime_SquarePBV = MatrixMultiply_2D_accessor(       num_i, num_j, num_k, ITERATIONS, accessorA, accessorB, accessorC_SquarePBV );

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  double runTime_SquarePBR = MatrixMultiply_2D_accessorRef(       num_i, num_j, num_k, ITERATIONS, accessorA, accessorB, accessorC_SquarePBR );


//  double runTime_SquareLAC = MatrixMultiply_2D_constructAccessorR( num_i,
// num_j, num_k, ITERATIONS,
//                                                                                   A,
// lengthsA,
//                                                                                   B,
// lengthsB,
//                                                                                   ptrC_SquareLAC,
// lengthsC );



  double runTime_PBVSelfDescribing = 100;//MatrixMultiply_2D_accessorRef( num_i,
                                         // num_j, num_k, ITERATIONS,
                                         // accessorA_SelfDescribing,
                                         // accessorB_SelfDescribing,
                                         // accessorC_SelfDescribing );

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  startTime = GetTimeMs64();
  for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter )
  {
    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {
        for( integer_t k = 0 ; k < num_k ; ++k )
        {
          accessorC_ParenNFC(i,j) += accessorA(i,k) * accessorB(k,j) + 3.1415 * accessorA(i,k) + 1.61803 * accessorB(k,j);
        }
      }
    }
  }
  endTime = GetTimeMs64();
  double runTime_ParenNFC =( endTime - startTime ) / 1000.0;

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  double runTime_ParenPBV = MatrixMultiply_2D_accessorPBV2( num_i, num_j, num_k, ITERATIONS, accessorA, accessorB, accessorC_ParenPBV );

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  double runTime_ParenPBR = MatrixMultiply_2D_accessorRef2( num_i, num_j, num_k, ITERATIONS, accessorA, accessorB, accessorC_ParenPBR );



  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  startTime = GetTimeMs64();
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
  endTime = GetTimeMs64();
  double runTime_arrayViewSquareNFC = ( endTime - startTime ) / 1000.0;


  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  double runTime_arrayViewSquarePBV = MatrixMultiply_2D_accessor(       num_i, num_j, num_k, ITERATIONS, arrayViewA, arrayViewB, arrayViewC_SquarePBV );

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  double runTime_arrayViewSquarePBR = MatrixMultiply_2D_accessorRef(       num_i, num_j, num_k, ITERATIONS, arrayViewA, arrayViewB, arrayViewC_SquarePBR );



  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  startTime = GetTimeMs64();
  for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter )
  {
    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {
        for( integer_t k = 0 ; k < num_k ; ++k )
        {
          ManagedArrayC_ParenNFC(i,j) += ManagedArrayA(i,k) * ManagedArrayB(k,j) + 3.1415 * ManagedArrayA(i,k) + 1.61803 * ManagedArrayB(k,j);
        }
      }
    }
  }
  endTime = GetTimeMs64();
  double runTime_arrayViewParenNFC =( endTime - startTime ) / 1000.0;

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  double runTime_arrayViewParenPBV = MatrixMultiply_2D_accessorPBV2( num_i, num_j, num_k, ITERATIONS, arrayViewA, arrayViewB, arrayViewC_ParenPBV );


  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  double runTime_arrayViewParenPBR = MatrixMultiply_2D_accessorRef2( num_i, num_j, num_k, ITERATIONS, arrayViewA, arrayViewB, arrayViewC_ParenPBR );



  double minRunTime = 1.0e99;
  minRunTime = std::min(runTime_Direct1dAccess,runTime_Direct1dRestrict);
  minRunTime = std::min( minRunTime, runTime_SquareNFC );
  minRunTime = std::min( minRunTime, runTime_SquarePBV );
  minRunTime = std::min( minRunTime, runTime_SquarePBR );
  minRunTime = std::min( minRunTime, runTime_ParenNFC );
  minRunTime = std::min( minRunTime, runTime_ParenPBV );
  minRunTime = std::min( minRunTime, runTime_ParenPBR );
  minRunTime = std::min( minRunTime, runTime_arrayViewSquareNFC );
  minRunTime = std::min( minRunTime, runTime_arrayViewSquarePBV );
  minRunTime = std::min( minRunTime, runTime_arrayViewSquarePBR );
  minRunTime = std::min( minRunTime, runTime_arrayViewParenNFC );
  minRunTime = std::min( minRunTime, runTime_arrayViewParenPBV );
  minRunTime = std::min( minRunTime, runTime_arrayViewParenPBR );

  if( output > 2 )
  {
    double error_SquareNFC   = 0.0;
    double error_SquarePBV = 0.0;
    double error_SquarePBR = 0.0;

    double error_ParenNFC = 0.0;
    double error_ParenPBV = 0.0;
    double error_ParenPBR = 0.0;

    double error_ManagedArraySquareNFC = 0.0;
    double error_ManagedArraySquarePBV = 0.0;
    double error_ManagedArraySquarePBR = 0.0;

    double error_ManagedArrayParenNFC = 0.0;
    double error_ManagedArrayParenPBV = 0.0;
    double error_ManagedArrayParenPBR = 0.0;

    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {
        error_SquareNFC += pow( C1D[i*num_j+j] - ptrC_SquareNFC[i*num_j+j], 2 );
        error_SquarePBV += pow( C1D[i*num_j+j] - ptrC_SquarePBV[i*num_j+j], 2 );
        error_SquarePBR += pow( C1D[i*num_j+j] - ptrC_SquarePBR[i*num_j+j], 2 );

        error_ParenNFC += pow( C1D[i*num_j+j] - ptrC_ParenNFC[i*num_j+j], 2 );
        error_ParenPBV += pow( C1D[i*num_j+j] - ptrC_ParenPBV[i*num_j+j], 2 );
        error_ParenPBR += pow( C1D[i*num_j+j] - ptrC_ParenPBR[i*num_j+j], 2 );

        error_ManagedArraySquareNFC += pow( C1D[i*num_j+j] - ManagedArrayC_SquareNFC[i][j], 2 );
        error_ManagedArraySquarePBV += pow( C1D[i*num_j+j] - ManagedArrayC_SquarePBV[i][j], 2 );
        error_ManagedArraySquarePBR += pow( C1D[i*num_j+j] - ManagedArrayC_SquarePBR[i][j], 2 );

        error_ManagedArrayParenNFC += pow( C1D[i*num_j+j] - ManagedArrayC_ParenNFC[i][j], 2 );
        error_ManagedArrayParenPBV += pow( C1D[i*num_j+j] - ManagedArrayC_ParenPBV[i][j], 2 );
        error_ManagedArrayParenPBR += pow( C1D[i*num_j+j] - ManagedArrayC_ParenPBR[i][j], 2 );

      }
    }
    GEOS_LOG("error_SquareNFC = "<<error_SquareNFC);
    GEOS_LOG("error_SquarePBV = "<<error_SquarePBV);
    GEOS_LOG("error_SquarePBR = "<<error_SquarePBR);
    GEOS_LOG("error_SquarePBR = "<<error_SquarePBR);

    GEOS_LOG("error_ParenNFC = "<<error_ParenNFC);
    GEOS_LOG("error_ParenPBV = "<<error_ParenPBV);
    GEOS_LOG("error_ParenPBR = "<<error_ParenPBR);

    GEOS_LOG("error_ManagedArraySquareNFC = "<<error_ManagedArraySquareNFC);
    GEOS_LOG("error_ManagedArraySquarePBV = "<<error_ManagedArraySquarePBV);
    GEOS_LOG("error_ManagedArraySquarePBR = "<<error_ManagedArraySquarePBR);

    GEOS_LOG("error_ManagedArrayParenNFC = "<<error_ManagedArrayParenNFC);
    GEOS_LOG("error_ManagedArrayParenPBV = "<<error_ManagedArrayParenPBV);
    GEOS_LOG("error_ManagedArrayParenPBR = "<<error_ManagedArrayParenPBR);


  }

  if( output > 1 )
  {
    printf( "1d array                             : %8.3f, %8.2f\n", runTime_Direct1dAccess, runTime_Direct1dAccess/minRunTime);
    printf( "1d array restrict                    : %8.3f, %8.2f\n", runTime_Direct1dRestrict, runTime_Direct1dRestrict / minRunTime);

    printf( "accessor[] nfc                       : %8.3f, %8.2f\n", runTime_SquareNFC, runTime_SquareNFC / minRunTime);
    printf( "accessor[] pbv                       : %8.3f, %8.2f\n", runTime_SquarePBV, runTime_SquarePBV / minRunTime);
    printf( "accessor[] pbr                       : %8.3f, %8.2f\n", runTime_SquarePBR, runTime_SquarePBR / minRunTime);

    printf( "accessor() nfc                       : %8.3f, %8.2f\n", runTime_ParenNFC, runTime_ParenNFC / minRunTime);
    printf( "accessor() pbv                       : %8.3f, %8.2f\n", runTime_ParenPBV, runTime_ParenPBV / minRunTime);
    printf( "accessor() pbr                       : %8.3f, %8.2f\n", runTime_ParenPBR, runTime_ParenPBR / minRunTime);

    printf( "ManagedArray[] nfc                   : %8.3f, %8.2f\n", runTime_arrayViewSquareNFC, runTime_arrayViewSquareNFC / minRunTime);
    printf( "ManagedArray[] pbv                   : %8.3f, %8.2f\n", runTime_arrayViewSquarePBV, runTime_arrayViewSquarePBV / minRunTime);
    printf( "ManagedArray[] pbr                   : %8.3f, %8.2f\n", runTime_arrayViewSquarePBR, runTime_arrayViewSquarePBR / minRunTime);

    printf( "ManagedArray() nfc                   : %8.3f, %8.2f\n", runTime_arrayViewParenNFC, runTime_arrayViewParenNFC / minRunTime);
    printf( "ManagedArray() pbv                   : %8.3f, %8.2f\n", runTime_arrayViewParenPBV, runTime_arrayViewParenPBV / minRunTime);
    printf( "ManagedArray() pbr                   : %8.3f, %8.2f\n", runTime_arrayViewParenPBR, runTime_arrayViewParenPBR / minRunTime);

  }


  if( output == 1 )
  {
    printf( "%8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n",
            runTime_Direct1dAccess,
            runTime_Direct1dRestrict,

            runTime_SquareNFC,
            runTime_SquarePBV,
            runTime_SquarePBR,

            runTime_ParenNFC,
            runTime_ParenPBV,
            runTime_ParenPBR,

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
