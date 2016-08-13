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
uint64_t GetTimeMs64()
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

double MatrixMultiply_1D( integer_t const num_i,
                          integer_t const num_j,
                          integer_t const num_k,
                          integer_t const ITERATIONS,
                          double const * const  A,
                          double const * const  B,
                          double * const  C )
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

double MatrixMultiply_1Dr( integer_t const num_i,
                          integer_t const num_j,
                          integer_t const num_k,
                          integer_t const ITERATIONS,
                          double const * const __restrict__  A,
                          double const * const __restrict__ B,
                          double * const __restrict__ C )
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

#define MATMULT \
uint64_t startTime = GetTimeMs64(); \
for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter ) \
{ \
  for( integer_t i = 0 ; i < num_i ; ++i ) \
  { \
    for( integer_t j = 0 ; j < num_j ; ++j ) \
    { \
      for( integer_t k = 0 ; k < num_k ; ++k ) \
      { \
        C[i][j] += A[i][k] * B[k][j] + 3.1415 * A[i][k] + 1.61803 * B[k][j]; \
      } \
    } \
  } \
} \
uint64_t endTime = GetTimeMs64(); \
return ( endTime - startTime ) / 1000.0;

inline double MatrixMultiply_2D_accessorInline( integer_t const num_i,
                                                integer_t const num_j,
                                                integer_t const num_k,
                                                integer_t const ITERATIONS,
                                                ArrayAccessor<double const,2> A,
                                                ArrayAccessor<double const,2> B,
                                                ArrayAccessor<double,2> C )
{
  MATMULT
}

double MatrixMultiply_2D_accessor( integer_t const num_i,
                                   integer_t const num_j,
                                   integer_t const num_k,
                                   integer_t const ITERATIONS,
                                   ArrayAccessor<double const,2> A,
                                   ArrayAccessor<double const,2> B,
                                   ArrayAccessor<double,2> C )
{
  MATMULT
}

inline double MatrixMultiply_2D_accessorInlineRef( integer_t const num_i,
                                                integer_t const num_j,
                                                integer_t const num_k,
                                                integer_t const ITERATIONS,
                                                ArrayAccessor<double const,2> & A,
                                                ArrayAccessor<double const,2> & B,
                                                ArrayAccessor<double,2> & C )
{
  MATMULT
}

double MatrixMultiply_2D_accessorRef( integer_t const num_i,
                                   integer_t const num_j,
                                   integer_t const num_k,
                                   integer_t const ITERATIONS,
                                   ArrayAccessor<double const,2> & A,
                                   ArrayAccessor<double const,2> & B,
                                   ArrayAccessor<double,2> & C )
{
  MATMULT
}



double MatrixMultiply_2D_copyConstruct( integer_t const num_i,
                                        integer_t const num_j,
                                        integer_t const num_k,
                                        integer_t const ITERATIONS,
                                        ArrayAccessor<double const,2> A0,
                                        ArrayAccessor<double const,2> B0,
                                        ArrayAccessor<double,2> C0 )
{
  ArrayAccessor<double const,2> A(A0);
  ArrayAccessor<double const,2> B(B0);
  ArrayAccessor<double,2> C(C0);
  MATMULT
}


double MatrixMultiply_2D_copyConstruct2( integer_t const num_i,
                                        integer_t const num_j,
                                        integer_t const num_k,
                                        integer_t const ITERATIONS,
                                        ArrayAccessor<double const,2> A0,
                                        ArrayAccessor<double const,2> B0,
                                        ArrayAccessor<double,2> C0 )
{
  integer_t stridesA[] = { num_k, 1 };
  integer_t stridesB[] = { num_j, 1 };
  integer_t stridesC[] = { num_j, 1 };

  ArrayAccessor<double const,2> A( A0.data(), A0.lengths(), stridesA );
  ArrayAccessor<double const,2> B( B0.data(), B0.lengths(), stridesB );
  ArrayAccessor<double,2> C( C0.data(), C0.lengths(), stridesC );
  MATMULT
}


double MatrixMultiply_2D_constructAccessorR( integer_t const num_i,
                                            integer_t const num_j,
                                            integer_t const num_k,
                                            integer_t const ITERATIONS,
                                            double const * const __restrict__ ptrA,
                                            integer_t const * const lengthA,
                                            double const * const __restrict__ ptrB,
                                            integer_t const * const lengthB,
                                            double * const __restrict__ ptrC,
                                            integer_t const * const lengthC )
{
  integer_t stridesA[] = { num_k, 1 };
  integer_t stridesB[] = { num_j, 1 };
  integer_t stridesC[] = { num_j, 1 };

  ArrayAccessor<double const,2> A( ptrA, lengthA, stridesA );
  ArrayAccessor<double const,2> B( ptrB, lengthB, stridesB );
  ArrayAccessor<double,2> C( ptrC, lengthC, stridesC );

  MATMULT
}

double MatrixMultiply_2D_constructAccessor( integer_t const num_i,
                                            integer_t const num_j,
                                            integer_t const num_k,
                                            integer_t const ITERATIONS,
                                            double const * const ptrA,
                                            integer_t const * const lengthA,
                                            double const * const ptrB,
                                            integer_t const * const lengthB,
                                            double * const ptrC,
                                            integer_t const * const lengthC )
{

  integer_t stridesA[] = { num_k, 1 };
  integer_t stridesB[] = { num_j, 1 };
  integer_t stridesC[] = { num_j, 1 };

  ArrayAccessor<double const,2> A( ptrA, lengthA, stridesA );
  ArrayAccessor<double const,2> B( ptrB, lengthB, stridesB );
  ArrayAccessor<double,2> C( ptrC, lengthC, stridesC );

  MATMULT
}


//
//double MatrixMultiply_2D_accessorSliced( integer const num_i,
//                                         integer const num_j,
//                                         integer const num_k,
//                                         integer const ITERATIONS,
//                                         ArrayAccessor<double,2> A,
//                                         ArrayAccessor<double,2> B,
//                                         ArrayAccessor<double,2> C )
//{
//  uint64_t startTime = GetTimeMs64();
//  for( integer iter = 0 ; iter < ITERATIONS ; ++iter )
//  {
//    for( integer i = 0 ; i < num_i ; ++i )
//    {
//      ArrayAccessor<double,1> arrayAi = A[i];
//      ArrayAccessor<double,1> arrayCi = C[i];
//      for( integer j = 0 ; j < num_j ; ++j )
//      {
//        for( integer k = 0 ; k < num_k ; ++k )
//        {
//          arrayCi[j] += arrayAi[k] * B[k][j];
//        }
//      }
//    }
//  }
//  uint64_t endTime = GetTimeMs64();
//  return ( endTime - startTime ) / 1000.0;
//}

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

  double * const restrict C1a = new double[num_i*num_j];
  double * const restrict C1b = new double[num_i*num_j];
  double C2_native[num_i][num_j];
  double * const restrict C2_1 = new double[num_i*num_j];
  double * const restrict C2_2 = new double[num_i*num_j];
  double * const restrict C2_3 = new double[num_i*num_j];
  double * const restrict C2_4 = new double[num_i*num_j];
  double * const restrict C2_5 = new double[num_i*num_j];
  double * const restrict C2_6 = new double[num_i*num_j];
  double * const restrict C2_7 = new double[num_i*num_j];
  double * const restrict C2_8 = new double[num_i*num_j];
  double * const restrict C2_9 = new double[num_i*num_j];
  double * const restrict C2_10 = new double[num_i*num_j];

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
      C1a[i*num_j+j] = 0.0;
      C1b[i*num_j+j] = 0.0;
      C2_native[i][j] = 0.0;
      C2_1[i*num_j+j] = 0.0;
      C2_2[i*num_j+j] = 0.0;
      C2_3[i*num_j+j] = 0.0;
      C2_4[i*num_j+j] = 0.0;
      C2_5[i*num_j+j] = 0.0;
      C2_6[i*num_j+j] = 0.0;
      C2_7[i*num_j+j] = 0.0;
      C2_8[i*num_j+j] = 0.0;
      C2_9[i*num_j+j] = 0.0;
      C2_10[i*num_j+j] = 0.0;
    }
  }

  double runTime1  = MatrixMultiply_1D( num_i, num_j, num_k, ITERATIONS, &(A[0][0]), &(B[0][0]), C1a );
  double runTime1r = MatrixMultiply_1Dr( num_i, num_j, num_k, ITERATIONS, &(A[0][0]), &(B[0][0]), C1b );




  uint64_t startTime = GetTimeMs64();
  for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter )
  {
    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {
        for( integer_t k = 0 ; k < num_k ; ++k )
        {
          C2_native[i][j] += A[i][k] * B[k][j] + 3.1415 * A[i][k] + 1.61803 * B[k][j];
        }
      }
    }
  }
  uint64_t endTime = GetTimeMs64();
  double runTime2_native = ( endTime - startTime ) / 1000.0;






  integer_t lengthsA[] = { num_i , num_k };
  integer_t lengthsB[] = { num_k , num_j };
  integer_t lengthsC[] = { num_i , num_j };

  integer_t stridesA[] = { num_k, 1 };
  integer_t stridesB[] = { num_j, 1 };
  integer_t stridesC[] = { num_j, 1 };

  ArrayAccessor<double const,2> accessorA( &(A[0][0]), lengthsA, stridesA );
  ArrayAccessor<double const,2> accessorB( &(B[0][0]), lengthsB, stridesB );
  ArrayAccessor<double,2> accessorC_1( C2_1, lengthsC, stridesC );
  ArrayAccessor<double,2> accessorC_2( C2_2, lengthsC, stridesC );
  ArrayAccessor<double,2> accessorC_3( C2_3, lengthsC, stridesC );
  ArrayAccessor<double,2> accessorC_4( C2_4, lengthsC, stridesC );
  ArrayAccessor<double,2> accessorC_5( C2_5, lengthsC, stridesC );
  ArrayAccessor<double,2> accessorC_8( C2_8, lengthsC, stridesC );
  ArrayAccessor<double,2> accessorC_9( C2_9, lengthsC, stridesC );



  startTime = GetTimeMs64();
  for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter )
  {
    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {
        for( integer_t k = 0 ; k < num_k ; ++k )
        {
          accessorC_1[i][j] += accessorA[i][k] * accessorB[k][j] + 3.1415 * accessorA[i][k] + 1.61803 * accessorB[k][j];
        }
      }
    }
  }
  endTime = GetTimeMs64();
  double runTime2_1 = ( endTime - startTime ) / 1000.0;



  double runTime2_2 = MatrixMultiply_2D_accessor(       num_i, num_j, num_k, ITERATIONS, accessorA, accessorB, accessorC_2 );
  double runTime2_3 = MatrixMultiply_2D_accessorInline( num_i, num_j, num_k, ITERATIONS, accessorA, accessorB, accessorC_3 );

  double runTime2_4 = MatrixMultiply_2D_accessorRef(       num_i, num_j, num_k, ITERATIONS, accessorA, accessorB, accessorC_4 );
  double runTime2_5 = MatrixMultiply_2D_accessorInlineRef( num_i, num_j, num_k, ITERATIONS, accessorA, accessorB, accessorC_5 );

  double runTime2_6 = MatrixMultiply_2D_constructAccessor( num_i, num_j, num_k, ITERATIONS,
                                                          &(A[0][0]), lengthsA,
                                                          &(B[0][0]), lengthsB,
                                                          C2_6, lengthsC );

  double runTime2_7 = MatrixMultiply_2D_constructAccessorR( num_i, num_j, num_k, ITERATIONS,
                                                          &(A[0][0]), lengthsA,
                                                          &(B[0][0]), lengthsB,
                                                          C2_7, lengthsC );

  double runTime2_8 = MatrixMultiply_2D_copyConstruct( num_i, num_j, num_k, ITERATIONS,  accessorA, accessorB, accessorC_8 );
  double runTime2_9 = MatrixMultiply_2D_copyConstruct2( num_i, num_j, num_k, ITERATIONS, accessorA, accessorB, accessorC_9 );


  if( output >= 3 )
  {
    double error1a = 0.0;
    double error1b = 0.0;
    double error2_1 = 0.0;
    double error2_2 = 0.0;
    double error2_3 = 0.0;
    double error2_4 = 0.0;
    double error2_5 = 0.0;
    double error2_6 = 0.0;
    double error2_7 = 0.0;
    double error2_8 = 0.0;
    double error2_9 = 0.0;

    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {
        error1a  += pow( C1a[i*num_j+j] - C2_native[i][j] , 2 ) ;
        error1b  += pow( C1b[i*num_j+j] - C2_native[i][j] , 2 ) ;
        error2_1 += pow( C2_native[i][j] - C2_1[i*num_j+j] , 2 ) ;
        error2_2 += pow( C2_native[i][j] - C2_2[i*num_j+j] , 2 ) ;
        error2_3 += pow( C2_native[i][j] - C2_3[i*num_j+j] , 2 ) ;
        error2_4 += pow( C2_native[i][j] - C2_4[i*num_j+j] , 2 ) ;
        error2_5 += pow( C2_native[i][j] - C2_5[i*num_j+j] , 2 ) ;
        error2_6 += pow( C2_native[i][j] - C2_6[i*num_j+j] , 2 ) ;
        error2_7 += pow( C2_native[i][j] - C2_7[i*num_j+j] , 2 ) ;
        error2_8 += pow( C2_native[i][j] - C2_8[i*num_j+j] , 2 ) ;
        error2_9 += pow( C2_native[i][j] - C2_9[i*num_j+j] , 2 ) ;
      }
    }
    std::cout<<"error1a = "<<error1a<<std::endl;
    std::cout<<"error1b = "<<error1b<<std::endl;
    std::cout<<"error2_1 = "<<error2_1<<std::endl;
    std::cout<<"error2_2 = "<<error2_2<<std::endl;
    std::cout<<"error2_3 = "<<error2_3<<std::endl;
    std::cout<<"error2_4 = "<<error2_4<<std::endl;
    std::cout<<"error2_5 = "<<error2_5<<std::endl;
    std::cout<<"error2_6 = "<<error2_6<<std::endl;
    std::cout<<"error2_7 = "<<error2_7<<std::endl;
    std::cout<<"error2_8 = "<<error2_8<<std::endl;
    std::cout<<"error2_9 = "<<error2_9<<std::endl;
  }

  if( output == 1 )
  {
    printf( "1d array                             : %8.3f, %8.3f\n", runTime1, 1.0);
    printf( "1d array restrict                    : %8.3f, %8.3f\n", runTime1r, runTime1r / runTime1);
    printf( "2d native                            : %8.3f, %8.3f\n", runTime2_native, runTime2_native / runTime1);
    printf( "accessor no func                     : %8.3f, %8.3f\n", runTime2_1, runTime2_1 / runTime1);
    printf( "accessor pbv                         : %8.3f, %8.3f\n", runTime2_2, runTime2_2 / runTime1);
    printf( "accessor pbv inline                  : %8.3f, %8.3f\n", runTime2_3, runTime2_3 / runTime1);
    printf( "accessor pbr                         : %8.3f, %8.3f\n", runTime2_4, runTime2_4 / runTime1);
    printf( "accessor pbr inline                  : %8.3f, %8.3f\n", runTime2_5, runTime2_5 / runTime1);
    printf( "accessor construct from ptr          : %8.3f, %8.3f\n", runTime2_6, runTime2_6 / runTime1);
    printf( "accessor construct from ptr restrict : %8.3f, %8.3f\n", runTime2_7, runTime2_7 / runTime1);
    printf( "accessor copy construct              : %8.3f, %8.3f\n", runTime2_8, runTime2_8 / runTime1);
    printf( "accessor copy construct ptr          : %8.3f, %8.3f\n", runTime2_9, runTime2_9 / runTime1);
  }

  if( output == 2 )
  {
    printf( "%8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f \n", runTime1,
                                                                                          runTime1r,
                                                                                          runTime2_native,
                                                                                          runTime2_1,
                                                                                          runTime2_2,
                                                                                          runTime2_3,
                                                                                          runTime2_4,
                                                                                          runTime2_5,
                                                                                          runTime2_6,
                                                                                          runTime2_7,
                                                                                          runTime2_8,
                                                                                          runTime2_9 );
  }

  return 0;
}
