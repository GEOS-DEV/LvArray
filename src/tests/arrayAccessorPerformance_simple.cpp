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


#include <sys/time.h>
#include <stdint.h>
#include <string>
#include <math.h>

#include "MultidimensionalArray2.hpp"

using namespace multidimensionalArray;
inline uint64_t GetTimeMs64()
{
  struct timeval tv;

  gettimeofday( &tv, NULL );

  uint64_t ret = tv.tv_usec;
  /* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
  ret /= 1000;

  /* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
  ret += ( tv.tv_sec * 1000 );

  return ret;

}



double MatrixMultiply_1Dr( integer_t const num_i,
                           integer_t const num_j,
                           integer_t const num_k,
                           integer_t const ITERATIONS,
                           double const * const restrict  A,
                           double const * const restrict B,
                           double * const restrict C )
{
  uint64_t startTime = GetTimeMs64();
  for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter )
  {
    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {
        for( integer_t k = 0 ; k < num_k ; ++k )
        {
          C[ i*num_j+j ] += A[ i*num_k+k ] * B[ k*num_j+j ] + 3.1415 * A[ i*num_k+k ] + 1.61803 * B[ k*num_j+j ];
        }
      }
    }
  }
  uint64_t endTime = GetTimeMs64();
  return ( endTime - startTime ) / 1000.0;
}

double PBV( integer_t const num_i,
            integer_t const num_j,
            integer_t const num_k,
            integer_t const ITERATIONS,
#if 0
            ArrayAccessor<double const,2> const A0,
            ArrayAccessor<double const,2> const B0,
            ArrayAccessor<double,2> C0 )
{
  integer_t const stridesA[] = { num_k, 1 };
  integer_t const stridesB[] = { num_j, 1 };
  integer_t const stridesC[] = { num_j, 1 };

  double const * const restrict ptrA = A0.data();
  integer_t const * const lengthA = A0.lengths();
  double const * const restrict ptrB = B0.data();
  integer_t const * const lengthB = B0.lengths();
  double * const restrict ptrC = C0.data();
  integer_t const * const lengthC = C0.lengths();

  ArrayAccessor<double const,2> const A( ptrA, lengthA, stridesA );
  ArrayAccessor<double const,2> const B( ptrB, lengthB, stridesB );
  ArrayAccessor<double,2> C( ptrC, lengthC, stridesC );
#else
            ArrayAccessor<double const,2> const A,
            ArrayAccessor<double const,2> const B,
            ArrayAccessor<double,2> C )
{
#endif

  uint64_t startTime = GetTimeMs64();
  for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter )
  {
    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {
        for( integer_t k = 0 ; k < num_k ; ++k )
        {
          C[i][j] += A[i][k] * B[k][j] + 3.1415 * A[i][k] + 1.61803 * B[k][j];
        }
      }
    }
  }
  uint64_t endTime = GetTimeMs64();
  return ( endTime - startTime ) / 1000.0;
}


double PBVP( integer_t const num_i,
             integer_t const num_j,
             integer_t const num_k,
             integer_t const ITERATIONS,
             ArrayAccessor<double const,2> const A,
             ArrayAccessor<double const,2> const B,
             ArrayAccessor<double,2> C )
{

  uint64_t startTime = GetTimeMs64();
  for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter )
  {
    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {
        for( integer_t k = 0 ; k < num_k ; ++k )
        {
          C(i,j) += A(i,k) * B(k,j) + 3.1415 * A(i,k) + 1.61803 * B(k,j);
        }
      }
    }
  }
  uint64_t endTime = GetTimeMs64();
  return ( endTime - startTime ) / 1000.0;
}


double PBV_PointerConstruct_Restrict( integer_t const num_i,
                                      integer_t const num_j,
                                      integer_t const num_k,
                                      integer_t const ITERATIONS,
                                      double const * const restrict ptrA,
                                      integer_t const * const lengthA,
                                      double const * const restrict ptrB,
                                      integer_t const * const lengthB,
                                      double * const restrict ptrC,
                                      integer_t const * const lengthC )
{
  integer_t const stridesA[] = { num_k, 1 };
  integer_t const stridesB[] = { num_j, 1 };
  integer_t const stridesC[] = { num_j, 1 };

  ArrayAccessor<double const,2> const A( ptrA, lengthA, stridesA );
  ArrayAccessor<double const,2> const B( ptrB, lengthB, stridesB );
  ArrayAccessor<double,2> C( ptrC, lengthC, stridesC );

  uint64_t startTime = GetTimeMs64();
  for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter )
  {
    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {
        for( integer_t k = 0 ; k < num_k ; ++k )
        {
          C[i][j] += A[i][k] * B[k][j] + 3.1415 * A[i][k] + 1.61803 * B[k][j];
        }
      }
    }
  }
  uint64_t endTime = GetTimeMs64();
  return ( endTime - startTime ) / 1000.0;
}

int main( int argc, char* argv[] )
{
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
  double A[num_i][num_k];
  double B[num_k][num_j];

  double * const restrict C1 = new double[num_i*num_j];
  double C2_native[num_i][num_j];
  double * const restrict C2_1 =  new double[num_i*num_j];
  double * const restrict C2_2 =  new double[num_i*num_j];
  double * const restrict C2_3 =  new double[num_i*num_j];

  srand( seed * seedmod );

  for( integer_t i = 0 ; i < num_i ; ++i )
    for( integer_t k = 0 ; k < num_k ; ++k )
      A[i][k] = rand();

  for( integer_t k = 0 ; k < num_k ; ++k )
    for( integer_t j = 0 ; j < num_j ; ++j )
    {
      B[k][j] = rand();
    }

  for( integer_t i = 0 ; i < num_i ; ++i )
  {
    for( integer_t j = 0 ; j < num_j ; ++j )
    {
      C1[i*num_j+j] = 0.0;
      C2_1[i*num_j+j] = 0.0;
      C2_2[i*num_j+j] = 0.0;
      C2_3[i*num_j+j] = 0.0;
    }
  }

  double runTime1r = MatrixMultiply_1Dr( num_i, num_j, num_k, ITERATIONS, &(A[0][0]), &(B[0][0]), C1 );


  integer_t const restrict lengthsA[] = { num_i, num_k };
  integer_t const restrict lengthsB[] = { num_k, num_j };
  integer_t const restrict lengthsC[] = { num_i, num_j };

  integer_t const restrict stridesA[] = { num_k, 1 };
  integer_t const restrict stridesB[] = { num_j, 1 };
  integer_t const restrict stridesC[] = { num_j, 1 };


  ArrayAccessor<double const,2> const accessorA( &(A[0][0]), lengthsA, stridesA );
  ArrayAccessor<double const,2> const accessorB( &(B[0][0]), lengthsB, stridesB );
  ArrayAccessor<double,2> accessorC_1( C2_1, lengthsC, stridesC );
  ArrayAccessor<double,2> accessorC_2( C2_2, lengthsC, stridesC );
  ArrayAccessor<double,2> accessorC_3( C2_3, lengthsC, stridesC );



  double runTime2_1 = PBV(       num_i, num_j, num_k, ITERATIONS, accessorA, accessorB, accessorC_1 );
  double runTime2_2 = PBV_PointerConstruct_Restrict( num_i, num_j, num_k, ITERATIONS,
                                                     &(A[0][0]), lengthsA,
                                                     &(B[0][0]), lengthsB,
                                                     C2_2, lengthsC );
  double runTime2_3 = PBVP(       num_i, num_j, num_k, ITERATIONS, accessorA, accessorB, accessorC_3 );

  if( output > 2 )
  {
    double error2_1 = 0.0;
    double error2_2 = 0.0;
    double error2_3 = 0.0;

    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {
        error2_1 += pow( C1[i*num_j+j] - C2_1[i*num_j+j], 2 );
        error2_2 += pow( C1[i*num_j+j] - C2_2[i*num_j+j], 2 );
        error2_3 += pow( C1[i*num_j+j] - C2_3[i*num_j+j], 2 );
      }
    }
    std::cout<<"error2_1 = "<<error2_1<<std::endl;
    std::cout<<"error2_2 = "<<error2_2<<std::endl;
    std::cout<<"error2_3 = "<<error2_3<<std::endl;
  }

  if( output > 1 )
  {
    printf( "1d array restrict                    : %8.3f, %8.3f\n", runTime1r, runTime1r / runTime1r);
    printf( "accessor pbv                         : %8.3f, %8.3f\n", runTime2_1, runTime2_1 / runTime1r);
    printf( "accessor construct from ptr restrict : %8.3f, %8.3f\n", runTime2_2, runTime2_2 / runTime1r);
    printf( "accessor pbv()                       : %8.3f, %8.3f\n", runTime2_3, runTime2_3 / runTime1r);
  }

  if( output == 1 )
  {
    printf( "%8.3f %8.3f %8.3f \n", runTime1r, runTime2_1, runTime2_2 );
  }

  return 0;
}
