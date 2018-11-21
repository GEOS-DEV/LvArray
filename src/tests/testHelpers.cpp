
#include "gtest/gtest.h"
#include "Array.hpp"

using namespace LvArray;

TEST( SizeHelper, SizeHelper )
{
  int dims[4] = { 3, 7, 11, 17 };

  int const result0 = size_helper<4,int,0>::f(dims);
  int const result1 = size_helper<4,int,1>::f(dims);
  int const result2 = size_helper<4,int,2>::f(dims);
  int const result3 = size_helper<4,int,3>::f(dims);
  ASSERT_TRUE( result0 == dims[0]*dims[1]*dims[2]*dims[3] );
  ASSERT_TRUE( result1 ==         dims[1]*dims[2]*dims[3] );
  ASSERT_TRUE( result2 ==                 dims[2]*dims[3] );
  ASSERT_TRUE( result3 ==                         dims[3] );

}


template< int NDIM, typename... INDICES >
int indexCaller( int const * const strides, INDICES... indices )
{
  return linearIndex_helper< NDIM, int, INDICES...>::evaluate(strides, indices...);
}

TEST( IndexHelper, IndexHelper )
{
  int dims[4] = { 3, 7, 11, 17 };
  int strides[4] = { dims[1]*dims[2]*dims[3],
                             dims[2]*dims[3],
                                     dims[3],
                                           1 };

  for( int a0=0 ; a0<dims[0] ; ++a0 )
  {
    for( int a1=0 ; a1<dims[1] ; ++a1 )
    {
      for( int a2=0 ; a2<dims[2] ; ++a2 )
      {
        for( int a3=0 ; a3<dims[3] ; ++a3 )
        {
          ASSERT_TRUE( indexCaller<4>( strides, a0,a1,a2,a3 ) ==
              a0*strides[0]+a1*strides[1]+a2*strides[2]+a3*strides[3] );
        }
      }
    }
  }
}


template< int NDIM, typename... INDICES >
void indexChecker( int const * const dims, INDICES... indices )
{
  linearIndex_helper< NDIM, int, INDICES...>::check(dims, indices...);
}

TEST( IndexChecker, IndexChecker )
{
  int dims[4] = { 3, 7, 11, 17 };

  for( int a0=-1 ; a0<=dims[0] ; ++a0 )
  {
    for( int a1=-1 ; a1<=dims[1] ; ++a1 )
    {
      for( int a2=-1 ; a2<=dims[2] ; ++a2 )
      {
        for( int a3=-1 ; a3<=dims[3] ; ++a3 )
        {
          if( a0 < 0 || a0 == dims[0] ||
              a1 < 0 || a1 == dims[1] ||
              a2 < 0 || a2 == dims[2] ||
              a3 < 0 || a3 == dims[3] )
          {
            EXPECT_DEATH_IF_SUPPORTED( indexChecker<4>( dims, a0,a1,a2,a3 ), "" );
          }
          else
          {
            ASSERT_NO_FATAL_FAILURE( indexChecker<4>( dims, a0,a1,a2,a3 ) );
          }
        }
      }
    }
  }
}


TEST( StrideHelper, StrideHelper )
{
  int dims[4] = { 3, 7, 11, 17 };

  int const result0 = stride_helper<4,int>::evaluate(dims);
  int const result1 = stride_helper<3,int>::evaluate(dims+1);
  int const result2 = stride_helper<2,int>::evaluate(dims+2);
  int const result3 = stride_helper<1,int>::evaluate(dims+3);

  std::cout<<dims[1]*dims[2]*dims[3]<<", "<<
                     dims[2]*dims[3]<<", "<<
                             dims[3]<<std::endl;

  ASSERT_TRUE( result0 == dims[1]*dims[2]*dims[3] );
  ASSERT_TRUE( result1 ==         dims[2]*dims[3] );
  ASSERT_TRUE( result2 ==                 dims[3] );
  ASSERT_TRUE( result3 ==                       1 );

}
int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  logger::InitializeLogger( MPI_COMM_WORLD );

  int result = 0;
  testing::InitGoogleTest( &argc, argv );
  result = RUN_ALL_TESTS();

  logger::FinalizeLogger();
  MPI_Finalize();
  return result;
}
