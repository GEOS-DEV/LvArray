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


#include "gtest/gtest.h"
#include <string>

#include "Array.hpp"
#include "SetSignalHandling.hpp"
#include "stackTrace.hpp"
#include <type_traits>

using namespace LvArray;

template < typename T >
using array = Array< T, 1, int >;

template < typename T >
using arrayView = ArrayView< T, 1, int >;

template < typename T >
using arraySlice = ArraySlice< T, 1, int >;


template < typename T >
using array2D = Array< T, 2, int >;

template < typename T >
using arrayView2D = ArrayView< T, 2, int >;

//class junk
//{
//public:
//  junk( junk const & source ):
//    m{ source.m[0], source.m[1] }
//  {}
//
//  double m[2];
//  chai::ManagedArray<double> m2;
//};

//TEST( ArrayView, triviallyCopyable )
//{
//  //  ASSERT_TRUE( std::is_trivially_copyable< junk >::value );
//    static_assert( std::is_trivially_copyable< junk >::value, "blah");
//}


static_assert( std::is_trivially_copyable< ArraySlice<int,1,int> >::value,
               "ArraySlice does not satisfy is_trivially_copyable");
//static_assert( std::is_trivially_copyable< ArrayView<int,1,int> >::value,
//               "ArrayView does not satisfy is_trivially_copyable");

TEST( ArrayView, test_upcast )
{
  constexpr int N = 10;     /* Number of values to push_back */

  array< int > arr(N);
  array< int > const & arrConst = arr;

  for( int a=0 ; a<N ; ++a )
  {
    arr[a] = a;
  }

  {
    arrayView< int > & arrView = arr;
    arrayView< int > const & arrViewConst = arr;
    arrayView< int const > & arrConstView = arrView;
    arrayView< int const > const & arrConstViewConst = arr;

    for( int a=0 ; a<N ; ++a )
    {
      ASSERT_EQ( arr[a], arrView[a] );
      ASSERT_EQ( arr[a], arrViewConst[a] );
      ASSERT_EQ( arr[a], arrConstView[a] );
      ASSERT_EQ( arr[a], arrConstViewConst[a] );
    }

    arraySlice<int> arrSlice1 = arrView;
    arraySlice<int> const & arrSlice2 = arrView;
    arraySlice<int const> arrSlice3 = arrView;
    arraySlice<int const> const & arrSlice4 = arrView;

    for( int a=0 ; a<N ; ++a )
    {
      ASSERT_EQ( arr[a], arrSlice1[a] );
      ASSERT_EQ( arr[a], arrSlice2[a] );
      ASSERT_EQ( arr[a], arrSlice3[a] );
      ASSERT_EQ( arr[a], arrSlice4[a] );
    }
  }

  // we want these to fail compilation
  {
//    arrayView< int > & arrView = arrConst;
    arrayView< int > const & arrViewConst = arrConst;
    for( int a=0 ; a<N ; ++a )
    {
      arrViewConst[a] = 2*a;
      ASSERT_EQ( arr[a], arrViewConst[a] );
    }
  }

  {
  arraySlice<int> arrSlice1 = arr;
  arraySlice<int> const & arrSlice2 = arr;
  arraySlice<int const> arrSlice3 = arr;
  arraySlice<int const> const & arrSlice4 = arr;

  for( int a=0 ; a<N ; ++a )
  {
    ASSERT_EQ( arr[a], arrSlice1[a] );
    ASSERT_EQ( arr[a], arrSlice2[a] );
    ASSERT_EQ( arr[a], arrSlice3[a] );
    ASSERT_EQ( arr[a], arrSlice4[a] );
  }
  }
}





TEST( ArrayView, test_dimReduction )
{
  array2D< int > v( 10, 1 );
  arrayView2D<int> const & vView = v;
  ArrayView< int, 1, int > const vView1d = v.dimReduce();

  for( int a=0 ; a<10 ; ++a )
  {
    v[a][0] = 2*a;
  }

  ASSERT_EQ( vView1d.data(), v.data() );

  for( int a=0 ; a<10 ; ++a )
  {
    ASSERT_EQ( vView[a][0], v[a][0] );
    ASSERT_EQ( vView1d[a], v[a][0] );
  }


}

#ifdef USE_CUDA

#define CUDA_TEST(X, Y)              \
  static void cuda_test_##X##Y();    \
  TEST(X, Y) { cuda_test_##X##Y(); } \
  static void cuda_test_##X##Y()


#ifdef USE_CHAI
#include "chai/util/forall.hpp"
#endif

CUDA_TEST(ArrayView, testChaiManagedArray)
{
  chai::ManagedArray<int> array1(10);
  chai::ManagedArray<int> array2(10);

  std::cout<<"step1"<<std::endl;

  forall(cuda(), 0, 10, [=] __device__(int i)
  {
    array1[i] = i;
  });

  std::cout<<"step2"<<std::endl;
  forall(cuda(), 0, 10, [=] __device__(int i)
  {
    array2[i] = array1[i];
  });

  std::cout<<"step3"<<std::endl;
  forall(sequential(), 0, 10, [=](int i)
  {
    ASSERT_EQ(array1[i], i);
    ASSERT_EQ(array2[i], i);
  });
  std::cout<<"done"<<std::endl;

}


CUDA_TEST(ArrayView, testViewOnCuda)
{
  constexpr int numElems = 2;
  array< int > array1(numElems);
  array< int > array2(numElems);
  arrayView< int > & arrayView1 = array1;
  arrayView< int > & arrayView2 = array2;

  arrayView< int > arrayView3(arrayView1);

  printf("host calls: size(), dim()[0] = %d, %d\n", arrayView1.size(), arrayView1.dims()[0]);

  forall(sequential(), 0, numElems, [=] (int i)
  {
    printf("kernel calls: size(), dim()[0] = %d, %d\n", arrayView1.size(), arrayView1.dims()[0]);
  });


  std::cout<<"step1"<<std::endl;

  forall(cuda(), 0, numElems, [=] __device__(int i)
  {
    printf("kernel calls:          size(), dim()[0]    = %d, %d\n", arrayView1.size(), arrayView1.dims()[0]);
    printf("            :                 m_dims[0]    =     %d\n", arrayView1.m_dims[0]);

    arrayView1[i] = i;
  });

  std::cout<<"step2"<<std::endl;
  forall(cuda(), 0, numElems, [=] __device__(int i)
  {
    arrayView2[i] = arrayView1[i];
  });

  std::cout<<"step3"<<std::endl;
  forall(sequential(), 0, numElems, [=](int i)
  {
    ASSERT_EQ(arrayView1[i], i);
    ASSERT_EQ(arrayView2[i], i);
  });
  std::cout<<"done"<<std::endl;
}

#endif

int main( int argc, char* argv[] )
{
//  MPI_Init( &argc, &argv );
//  logger::InitializeLogger( MPI_COMM_WORLD );

//  cxx_utilities::setSignalHandling(cxx_utilities::handler1);

  int result = 0;
  testing::InitGoogleTest( &argc, argv );
  result = RUN_ALL_TESTS();

//  logger::FinalizeLogger();
//  MPI_Finalize();
  return result;
}
