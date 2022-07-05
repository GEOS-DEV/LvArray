/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "typeManipulation.hpp"
#include "testUtils.hpp"
#include "Array.hpp"
#include "SortedArray.hpp"
#include "ArrayOfArrays.hpp"
#include "ArrayOfSets.hpp"
#include "SparsityPattern.hpp"
#include "CRSMatrix.hpp"

// TPL includes
#include <gtest/gtest.h>
#include <RAJA/RAJA.hpp>

// System includes
#include <vector>
#include <map>

// You can't define a nvcc extended host-device or device lambda in a TEST statement.
#define CUDA_TEST( X, Y )                 \
  static void cuda_test_ ## X ## _ ## Y();    \
  TEST( X, Y ) { cuda_test_ ## X ## _ ## Y(); } \
  static void cuda_test_ ## X ## _ ## Y()

namespace LvArray
{
namespace testing
{

CUDA_TEST( typeManipulation, forEachArg )
{
  // Test with no arguments.
  typeManipulation::forEachArg( [] () {} );

  // Test with arguments with the same type.
  int x = 1;
  int y = 2;
  int z = 3;
  typeManipulation::forEachArg( [] ( int & val )
  {
    val *= val;
  }, x, y, z );

  EXPECT_EQ( x, 1 );
  EXPECT_EQ( y, 4 );
  EXPECT_EQ( z, 9 );

  // Test with arguments of different types.
  std::string str = "10";
  double pi = 3;
  typeManipulation::forEachArg( [] ( auto & val )
  {
    val += val;
  }, x, str, pi );

  EXPECT_EQ( x, 2 );
  EXPECT_EQ( str, "1010" );
  EXPECT_EQ( pi, 6 );

#if defined(LVARRAY_USE_CUDA)
  // Test on device.
  RAJA::ReduceSum< RAJA::cuda_reduce, int > intReducer( 1 );
  RAJA::ReduceSum< RAJA::cuda_reduce, float > floatReducer( 3 );
  RAJA::ReduceSum< RAJA::cuda_reduce, double > doubleReducer( 6 );
  forall< parallelDevicePolicy< 32 > >( 1, [intReducer, floatReducer, doubleReducer] LVARRAY_DEVICE ( int )
      {
        // This has to be a host-device lambda to avoid errors.
        typeManipulation::forEachArg( [] LVARRAY_HOST_DEVICE ( auto & reducer )
        {
          reducer += 1;
        }, intReducer, floatReducer, doubleReducer );
      } );

  EXPECT_EQ( intReducer.get(), 2 );
  EXPECT_EQ( floatReducer.get(), 4 );
  EXPECT_EQ( doubleReducer.get(), 7 );
#elif defined(LVARRAY_USE_HIP)
  // Test on device.
  RAJA::ReduceSum< RAJA::hip_reduce, int > intReducer( 1 );
  RAJA::ReduceSum< RAJA::hip_reduce, float > floatReducer( 3 );
  RAJA::ReduceSum< RAJA::hip_reduce, double > doubleReducer( 6 );
  forall< parallelDevicePolicy< 32 > >( 1, [intReducer, floatReducer, doubleReducer] LVARRAY_DEVICE ( int )
      {
        // This has to be a host-device lambda to avoid errors.
        typeManipulation::forEachArg( [] LVARRAY_HOST_DEVICE ( auto & reducer )
        {
          reducer += 1;
        }, intReducer, floatReducer, doubleReducer );
      } );

  EXPECT_EQ( intReducer.get(), 2 );
  EXPECT_EQ( floatReducer.get(), 4 );
  EXPECT_EQ( doubleReducer.get(), 7 );
#endif
}

TEST( typeManipulation, all_of )
{
  static_assert( typeManipulation::all_of< true >::value, "Should be true." );
  static_assert( !typeManipulation::all_of< false >::value, "Should be false." );

  static_assert( typeManipulation::all_of< true, true >::value, "Should be true." );
  static_assert( !typeManipulation::all_of< true, false >::value, "Should be false." );
  static_assert( !typeManipulation::all_of< false, true >::value, "Should be false." );
  static_assert( !typeManipulation::all_of< false, false >::value, "Should be false." );

  static_assert( typeManipulation::all_of< true, true, true >::value, "Should be true." );
  static_assert( !typeManipulation::all_of< true, true, false >::value, "Should be false." );
  static_assert( !typeManipulation::all_of< true, false, true >::value, "Should be false." );
  static_assert( !typeManipulation::all_of< true, false, false >::value, "Should be false." );
  static_assert( !typeManipulation::all_of< false, true, true >::value, "Should be false." );
  static_assert( !typeManipulation::all_of< false, true, false >::value, "Should be false." );
  static_assert( !typeManipulation::all_of< false, false, true >::value, "Should be false." );
  static_assert( !typeManipulation::all_of< false, false, false >::value, "Should be false." );
}

TEST( typeManipulation, all_of_t )
{
  static_assert( typeManipulation::all_of_t< std::true_type >::value, "Should be true." );
  static_assert( !typeManipulation::all_of_t< std::false_type >::value, "Should be false." );

  static_assert( typeManipulation::all_of_t< std::true_type, std::true_type >::value, "Should be true." );
  static_assert( !typeManipulation::all_of_t< std::true_type, std::false_type >::value, "Should be false." );
  static_assert( !typeManipulation::all_of_t< std::false_type, std::true_type >::value, "Should be false." );
  static_assert( !typeManipulation::all_of_t< std::false_type, std::false_type >::value, "Should be false." );

  static_assert( typeManipulation::all_of_t< std::true_type, std::true_type, std::true_type >::value, "Should be true." );
  static_assert( !typeManipulation::all_of_t< std::true_type, std::true_type, std::false_type >::value, "Should be false." );
  static_assert( !typeManipulation::all_of_t< std::true_type, std::false_type, std::true_type >::value, "Should be false." );
  static_assert( !typeManipulation::all_of_t< std::true_type, std::false_type, std::false_type >::value, "Should be false." );
  static_assert( !typeManipulation::all_of_t< std::false_type, std::true_type, std::true_type >::value, "Should be false." );
  static_assert( !typeManipulation::all_of_t< std::false_type, std::true_type, std::false_type >::value, "Should be false." );
  static_assert( !typeManipulation::all_of_t< std::false_type, std::false_type, std::true_type >::value, "Should be false." );
  static_assert( !typeManipulation::all_of_t< std::false_type, std::false_type, std::false_type >::value, "Should be false." );
}

TEST( typeManipulation, is_instantiation_of )
{
  static_assert( typeManipulation::is_instantiation_of< std::vector, std::vector< double > >, "Should be true." );
  static_assert( typeManipulation::is_instantiation_of< std::vector, std::vector< float > >, "Should be true." );
  static_assert( typeManipulation::is_instantiation_of< std::map, std::map< std::string, int > >, "Should be true." );

  static_assert( !typeManipulation::is_instantiation_of< std::vector, std::map< float, int > >, "Should be false." );
  static_assert( !typeManipulation::is_instantiation_of< std::map, int >, "Should be false." );
  static_assert( !typeManipulation::is_instantiation_of< std::map, std::vector< double > >, "Should be false." );
}

template< typename T, int NDIM, typename PERM >
using ArrayT = Array< T, NDIM, PERM, std::ptrdiff_t, DEFAULT_BUFFER >;

template< typename T, int NDIM, int USD >
using ArrayViewT = ArrayView< T, NDIM, USD, std::ptrdiff_t, DEFAULT_BUFFER >;

template< typename T >
using SortedArrayT = SortedArray< T, std::ptrdiff_t, DEFAULT_BUFFER >;

template< typename T >
using SortedArrayViewT = SortedArrayView< T, std::ptrdiff_t, DEFAULT_BUFFER >;

template< typename T >
using ArrayOfArraysT = ArrayOfArrays< T, std::ptrdiff_t, DEFAULT_BUFFER >;

template< typename T, bool CONST_SIZES >
using ArrayOfArraysViewT = ArrayOfArraysView< T, std::ptrdiff_t const, CONST_SIZES, DEFAULT_BUFFER >;

template< typename T >
using ArrayOfSetsT = ArrayOfSets< T, std::ptrdiff_t, DEFAULT_BUFFER >;

template< typename T >
using ArrayOfSetsViewT = ArrayOfSetsView< T, std::ptrdiff_t const, DEFAULT_BUFFER >;

template< typename T >
using SparsityPatternT = SparsityPattern< T, std::ptrdiff_t, DEFAULT_BUFFER >;

template< typename T >
using SparsityPatternViewT = SparsityPatternView< T, std::ptrdiff_t const, DEFAULT_BUFFER >;

template< typename T, typename COL_TYPE >
using CRSMatrixT = CRSMatrix< T, COL_TYPE, std::ptrdiff_t, DEFAULT_BUFFER >;

template< typename T, typename COL_TYPE >
using CRSMatrixViewT = CRSMatrixView< T, COL_TYPE, std::ptrdiff_t const, DEFAULT_BUFFER >;

template< typename T >
struct MethodDetectionTrue : public ::testing::Test
{
  void test()
  {
    static_assert( typeManipulation::HasMemberFunction_toView< T >, "Should be true." );
    static_assert( typeManipulation::HasMemberFunction_toViewConst< T >, "Should be true." );
    static_assert( bufferManipulation::HasMemberFunction_move< T >, "Should be true." );

    // Should be detected on a reference to T as well.
    static_assert( typeManipulation::HasMemberFunction_toView< T & >, "Should be true." );
    static_assert( typeManipulation::HasMemberFunction_toViewConst< T & >, "Should be true." );
    static_assert( bufferManipulation::HasMemberFunction_move< T & >, "Should be true." );

    // All the methods should be const.
    static_assert( typeManipulation::HasMemberFunction_toView< T const >, "Should be true." );
    static_assert( typeManipulation::HasMemberFunction_toViewConst< T const >, "Should be true." );
    static_assert( bufferManipulation::HasMemberFunction_move< T const >, "Should be true." );

    // All the methods should be const.
    static_assert( typeManipulation::HasMemberFunction_toView< T const & >, "Should be true." );
    static_assert( typeManipulation::HasMemberFunction_toViewConst< T const & >, "Should be true." );
    static_assert( bufferManipulation::HasMemberFunction_move< T const & >, "Should be true." );
  }
};

using ObjectsThatHaveTheMethods = ::testing::Types<
  ArrayT< int, 3, RAJA::PERM_IJK >
  , ArrayT< std::string, 4, RAJA::PERM_IKLJ >
  , ArrayT< ArrayT< int, 1, RAJA::PERM_I >, 1, RAJA::PERM_I >
  , ArrayViewT< double, 2, 1 >
  , ArrayViewT< float const, 4, 2 >
  , ArrayViewT< ArrayViewT< std::string, 2, 0 > const, 3, 1 >
  , SortedArrayT< int >
  , SortedArrayT< std::string >
  , SortedArrayViewT< int >
  , SortedArrayViewT< std::string >
  , ArrayOfArraysT< int >
  , ArrayOfArraysT< std::string >
  , ArrayOfArraysViewT< int, false >
  , ArrayOfArraysViewT< std::string, true >
  , ArrayOfArraysViewT< double const, true >
  , SparsityPatternT< int >
  , SparsityPatternT< long >
  , SparsityPatternViewT< int >
  , SparsityPatternViewT< long const >
  , CRSMatrixT< double, int >
  , CRSMatrixT< std::string, long >
  , CRSMatrixViewT< double, int >
  , CRSMatrixViewT< int, long const >
  , CRSMatrixViewT< int const, long const >
  >;

TYPED_TEST_SUITE( MethodDetectionTrue, ObjectsThatHaveTheMethods, );

TYPED_TEST( MethodDetectionTrue, test )
{
  this->test();
}

template< typename T >
struct MethodDetectionFalse : public ::testing::Test
{
  void test()
  {
    static_assert( !typeManipulation::HasMemberFunction_toView< T >, "Should be false." );
    static_assert( !typeManipulation::HasMemberFunction_toViewConst< T >, "Should be false." );
    static_assert( !bufferManipulation::HasMemberFunction_move< T >, "Should be false." );
  }
};

using ObjectsThatDontHaveTheMethods = ::testing::Types<
  int
  , int const
  , int const &
  , int *
  , int const *
  , float
  , char[ 55 ]
  , std::string
  , std::vector< int >
  >;

TYPED_TEST_SUITE( MethodDetectionFalse, ObjectsThatDontHaveTheMethods, );

TYPED_TEST( MethodDetectionFalse, test )
{
  this->test();
}

template< typename T_VIEWS_TUPLE >
struct GetViewTypes : public ::testing::Test
{
  using T = std::tuple_element_t< 0, T_VIEWS_TUPLE >;
  using VIEW = std::tuple_element_t< 1, T_VIEWS_TUPLE >;
  using VIEW_CONST_SIZES = std::tuple_element_t< 2, T_VIEWS_TUPLE >;
  using VIEW_CONST = std::tuple_element_t< 3, T_VIEWS_TUPLE >;
  using NESTED_VIEW = std::tuple_element_t< 4, T_VIEWS_TUPLE >;
  using NESTED_VIEW_CONST = std::tuple_element_t< 5, T_VIEWS_TUPLE >;

  void test()
  {
    static_assert( std::is_same< VIEW, typeManipulation::ViewType< T > >::value, "" );
    static_assert( std::is_same< VIEW_CONST_SIZES, typeManipulation::ViewTypeConstSizes< T > >::value, "" );
    static_assert( std::is_same< VIEW_CONST, typeManipulation::ViewTypeConst< T > >::value, "" );
    static_assert( std::is_same< NESTED_VIEW, typeManipulation::NestedViewType< T > >::value, "" );
    static_assert( std::is_same< NESTED_VIEW_CONST, typeManipulation::NestedViewTypeConst< T > >::value, "" );

    // A reference to T should give the same type.
    static_assert( std::is_same< VIEW, typeManipulation::ViewType< T & > >::value, "" );
    static_assert( std::is_same< VIEW_CONST_SIZES, typeManipulation::ViewTypeConstSizes< T & > >::value, "" );
    static_assert( std::is_same< VIEW_CONST, typeManipulation::ViewTypeConst< T & > >::value, "" );
    static_assert( std::is_same< NESTED_VIEW, typeManipulation::NestedViewType< T & > >::value, "" );
    static_assert( std::is_same< NESTED_VIEW_CONST, typeManipulation::NestedViewTypeConst< T & > >::value, "" );
  }
};

using GetViewTypesTypes = ::testing::Types<
  std::tuple< ArrayT< int, 3, RAJA::PERM_IJK >,
              ArrayViewT< int, 3, 2 >,
              ArrayViewT< int, 3, 2 >,
              ArrayViewT< int const, 3, 2 >,
              ArrayViewT< int, 3, 2 > const,
              ArrayViewT< int const, 3, 2 > const
              >
  , std::tuple< ArrayT< std::string, 4, RAJA::PERM_IKLJ > const,
                ArrayViewT< std::string, 4, 1 >,
                ArrayViewT< std::string, 4, 1 >,
                ArrayViewT< std::string const, 4, 1 >,
                ArrayViewT< std::string, 4, 1 > const,
                ArrayViewT< std::string const, 4, 1 > const
                >
  , std::tuple< ArrayT< ArrayT< int, 1, RAJA::PERM_I >, 1, RAJA::PERM_I >,
                ArrayViewT< ArrayT< int, 1, RAJA::PERM_I >, 1, 0 >,
                ArrayViewT< ArrayT< int, 1, RAJA::PERM_I >, 1, 0 >,
                ArrayViewT< ArrayT< int, 1, RAJA::PERM_I > const, 1, 0 >,
                ArrayViewT< ArrayViewT< int, 1, 0 > const, 1, 0 > const,
                ArrayViewT< ArrayViewT< int const, 1, 0 > const, 1, 0 > const
                >
  , std::tuple< ArrayT< ArrayT< ArrayT< int, 2, RAJA::PERM_JI >, 1, RAJA::PERM_I >, 1, RAJA::PERM_I >,
                ArrayViewT< ArrayT< ArrayT< int, 2, RAJA::PERM_JI >, 1, RAJA::PERM_I >, 1, 0 >,
                ArrayViewT< ArrayT< ArrayT< int, 2, RAJA::PERM_JI >, 1, RAJA::PERM_I >, 1, 0 >,
                ArrayViewT< ArrayT< ArrayT< int, 2, RAJA::PERM_JI >, 1, RAJA::PERM_I > const, 1, 0 >,
                ArrayViewT< ArrayViewT< ArrayViewT< int, 2, 0 > const, 1, 0 > const, 1, 0 > const,
                ArrayViewT< ArrayViewT< ArrayViewT< int const, 2, 0 > const, 1, 0 > const, 1, 0 > const
                >
  , std::tuple< ArrayViewT< double, 2, 1 >,
                ArrayViewT< double, 2, 1 >,
                ArrayViewT< double, 2, 1 >,
                ArrayViewT< double const, 2, 1 >,
                ArrayViewT< double, 2, 1 > const,
                ArrayViewT< double const, 2, 1 > const
                >
  , std::tuple< ArrayViewT< char const, 4, 2 > const,
                ArrayViewT< char const, 4, 2 >,
                ArrayViewT< char const, 4, 2 >,
                ArrayViewT< char const, 4, 2 >,
                ArrayViewT< char const, 4, 2 > const,
                ArrayViewT< char const, 4, 2 > const
                >
  , std::tuple< ArrayViewT< ArrayViewT< std::string, 2, 0 > const, 3, 1 >,
                ArrayViewT< ArrayViewT< std::string, 2, 0 > const, 3, 1 >,
                ArrayViewT< ArrayViewT< std::string, 2, 0 > const, 3, 1 >,
                ArrayViewT< ArrayViewT< std::string, 2, 0 > const, 3, 1 >,
                ArrayViewT< ArrayViewT< std::string, 2, 0 > const, 3, 1 > const,
                ArrayViewT< ArrayViewT< std::string const, 2, 0 > const, 3, 1 > const
                >
  , std::tuple< SortedArrayT< float >,
                SortedArrayViewT< float const >,
                SortedArrayViewT< float const >,
                SortedArrayViewT< float const >,
                SortedArrayViewT< float const >,
                SortedArrayViewT< float const >
                >
  , std::tuple< SortedArrayViewT< std::string >,
                SortedArrayViewT< std::string const >,
                SortedArrayViewT< std::string const >,
                SortedArrayViewT< std::string const >,
                SortedArrayViewT< std::string const >,
                SortedArrayViewT< std::string const >
                >
  , std::tuple< ArrayOfArraysT< int > const,
                ArrayOfArraysViewT< int, false >,
                ArrayOfArraysViewT< int, true >,
                ArrayOfArraysViewT< int const, true >,
                ArrayOfArraysViewT< int, false >,
                ArrayOfArraysViewT< int const, true >
                >
  , std::tuple< ArrayOfArraysViewT< double, true >,
                ArrayOfArraysViewT< double, true >,
                ArrayOfArraysViewT< double, true >,
                ArrayOfArraysViewT< double const, true >,
                ArrayOfArraysViewT< double, true >,
                ArrayOfArraysViewT< double const, true >
                >
  , std::tuple< ArrayOfArraysViewT< std::string const, true > const,
                ArrayOfArraysViewT< std::string const, true >,
                ArrayOfArraysViewT< std::string const, true >,
                ArrayOfArraysViewT< std::string const, true >,
                ArrayOfArraysViewT< std::string const, true >,
                ArrayOfArraysViewT< std::string const, true >
                >
  , std::tuple< SparsityPatternT< int > const,
                SparsityPatternViewT< int >,
                SparsityPatternViewT< int >,
                SparsityPatternViewT< int const >,
                SparsityPatternViewT< int >,
                SparsityPatternViewT< int const >
                >
  , std::tuple< SparsityPatternViewT< long >,
                SparsityPatternViewT< long >,
                SparsityPatternViewT< long >,
                SparsityPatternViewT< long const >,
                SparsityPatternViewT< long >,
                SparsityPatternViewT< long const >
                >
  , std::tuple< SparsityPatternViewT< unsigned char const >,
                SparsityPatternViewT< unsigned char const >,
                SparsityPatternViewT< unsigned char const >,
                SparsityPatternViewT< unsigned char const >,
                SparsityPatternViewT< unsigned char const >,
                SparsityPatternViewT< unsigned char const >
                >
  , std::tuple< CRSMatrixT< double, int >,
                CRSMatrixViewT< double, int >,
                CRSMatrixViewT< double, int const >,
                CRSMatrixViewT< double const, int const >,
                CRSMatrixViewT< double, int >,
                CRSMatrixViewT< double const, int const >
                >
  , std::tuple< CRSMatrixViewT< char, int > const,
                CRSMatrixViewT< char, int >,
                CRSMatrixViewT< char, int const >,
                CRSMatrixViewT< char const, int const >,
                CRSMatrixViewT< char, int >,
                CRSMatrixViewT< char const, int const >
                >
  , std::tuple< CRSMatrixViewT< int, long const >,
                CRSMatrixViewT< int, long const >,
                CRSMatrixViewT< int, long const >,
                CRSMatrixViewT< int const, long const >,
                CRSMatrixViewT< int, long const >,
                CRSMatrixViewT< int const, long const >
                >
  , std::tuple< CRSMatrixViewT< float const, long const >,
                CRSMatrixViewT< float const, long const >,
                CRSMatrixViewT< float const, long const >,
                CRSMatrixViewT< float const, long const >,
                CRSMatrixViewT< float const, long const >,
                CRSMatrixViewT< float const, long const >
                >
  , std::tuple< int,
                int &,
                int &,
                int const &,
                int &,
                int const &
                >
  , std::tuple< int const,
                int const &,
                int const &,
                int const &,
                int const &,
                int const &
                >
  , std::tuple< int const &,
                int const &,
                int const &,
                int const &,
                int const &,
                int const &
                >
  , std::tuple< int *,
                int * &,
                int * &,
                int * const &,
                int * &,
                int * const &
                >
  , std::tuple< int const *,
                int const * &,
                int const * &,
                int const * const &,
                int const * &,
                int const * const &
                >
  , std::tuple< float,
                float &,
                float &,
                float const &,
                float &,
                float const &
                >
  , std::tuple< char[ 55 ],
                char (&)[ 55 ],
                char (&)[ 55 ],
                char const (&)[ 55 ],
                char (&)[ 55 ],
                char const (&)[ 55 ]
                >
  , std::tuple< std::string,
                std::string &,
                std::string &,
                std::string const &,
                std::string &,
                std::string const &
                >
  , std::tuple< std::vector< int >,
                std::vector< int > &,
                std::vector< int > &,
                std::vector< int > const &,
                std::vector< int > &,
                std::vector< int > const &
                >
  >;

TYPED_TEST_SUITE( GetViewTypes, GetViewTypesTypes, );

TYPED_TEST( GetViewTypes, test )
{
  this->test();
}

TEST( Permutations, getDimension )
{
  // 1D
  static_assert( typeManipulation::getDimension< RAJA::PERM_I > == 1, "Dimension should be 1." );

  // 2D
  static_assert( typeManipulation::getDimension< RAJA::PERM_IJ > == 2, "Dimension should be 2." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_JI > == 2, "Dimension should be 2." );

  // 3D
  static_assert( typeManipulation::getDimension< RAJA::PERM_IJK > == 3, "Dimension should be 3." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_JIK > == 3, "Dimension should be 3." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_IKJ > == 3, "Dimension should be 3." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_KIJ > == 3, "Dimension should be 3." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_JKI > == 3, "Dimension should be 3." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_KJI > == 3, "Dimension should be 3." );

  // 4D
  static_assert( typeManipulation::getDimension< RAJA::PERM_IJKL > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_JIKL > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_IKJL > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_KIJL > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_JKIL > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_KJIL > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_IJLK > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_JILK > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_ILJK > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_LIJK > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_JLIK > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_LJIK > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_IKLJ > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_KILJ > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_ILKJ > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_LIKJ > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_KLIJ > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_LKIJ > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_JKLI > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_KJLI > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_JLKI > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_LJKI > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_KLJI > == 4, "Dimension should be 4." );
  static_assert( typeManipulation::getDimension< RAJA::PERM_LKJI > == 4, "Dimension should be 4." );
}

TEST( Permutations, getStrideOneDimension )
{
  // 1D
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_I {} ) == 0, "Incorrect stride one dimension." );

  // 2D
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_IJ {} ) == 1, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_JI {} ) == 0, "Incorrect stride one dimension." );

  // 3D
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_IJK {} ) == 2, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_JIK {} ) == 2, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_IKJ {} ) == 1, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_KIJ {} ) == 1, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_JKI {} ) == 0, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_KJI {} ) == 0, "Incorrect stride one dimension." );

  // 4D
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_IJKL {} ) == 3, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_JIKL {} ) == 3, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_IKJL {} ) == 3, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_KIJL {} ) == 3, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_JKIL {} ) == 3, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_KJIL {} ) == 3, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_IJLK {} ) == 2, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_JILK {} ) == 2, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_ILJK {} ) == 2, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_LIJK {} ) == 2, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_JLIK {} ) == 2, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_LJIK {} ) == 2, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_IKLJ {} ) == 1, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_KILJ {} ) == 1, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_ILKJ {} ) == 1, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_LIKJ {} ) == 1, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_KLIJ {} ) == 1, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_LKIJ {} ) == 1, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_JKLI {} ) == 0, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_KJLI {} ) == 0, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_JLKI {} ) == 0, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_LJKI {} ) == 0, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_KLJI {} ) == 0, "Incorrect stride one dimension." );
  static_assert( typeManipulation::getStrideOneDimension( RAJA::PERM_LKJI {} ) == 0, "Incorrect stride one dimension." );
}

TEST( Permutations, isValidPermutation )
{
  // 1D
  static_assert( typeManipulation::isValidPermutation( camp::idx_seq< 0 > {} ), "This is a valid permutation." );

  static_assert( !typeManipulation::isValidPermutation( camp::idx_seq< 1 > {} ), "This is not a valid permutation." );
  static_assert( !typeManipulation::isValidPermutation( camp::idx_seq< -1 > {} ), "This is not a valid permutation." );

  // 2D
  static_assert( typeManipulation::isValidPermutation( camp::idx_seq< 0, 1 > {} ), "This is a valid permutation." );
  static_assert( typeManipulation::isValidPermutation( camp::idx_seq< 1, 0 > {} ), "This is a valid permutation." );

  static_assert( !typeManipulation::isValidPermutation( camp::idx_seq< 1, 1 > {} ), "This is not a valid permutation." );
  static_assert( !typeManipulation::isValidPermutation( camp::idx_seq< 0, 2 > {} ), "This is not a valid permutation." );
  static_assert( !typeManipulation::isValidPermutation( camp::idx_seq< -1, 0 > {} ), "This is not a valid permutation." );

  // 3D
  static_assert( typeManipulation::isValidPermutation( camp::idx_seq< 0, 1, 2 > {} ), "This is a valid permutation." );
  static_assert( typeManipulation::isValidPermutation( camp::idx_seq< 0, 2, 1 > {} ), "This is a valid permutation." );
  static_assert( typeManipulation::isValidPermutation( camp::idx_seq< 1, 0, 2 > {} ), "This is a valid permutation." );
  static_assert( typeManipulation::isValidPermutation( camp::idx_seq< 1, 2, 0 > {} ), "This is a valid permutation." );
  static_assert( typeManipulation::isValidPermutation( camp::idx_seq< 2, 0, 1 > {} ), "This is a valid permutation." );
  static_assert( typeManipulation::isValidPermutation( camp::idx_seq< 2, 1, 0 > {} ), "This is a valid permutation." );

  static_assert( !typeManipulation::isValidPermutation( camp::idx_seq< 0, 1, 5 > {} ), "This is not a valid permutation." );
  static_assert( !typeManipulation::isValidPermutation( camp::idx_seq< 0, 1, 0 > {} ), "This is not a valid permutation." );
  static_assert( !typeManipulation::isValidPermutation( camp::idx_seq< -6, 1, 0 > {} ), "This is not a valid permutation." );
}


TEST( Permutation, AsArray )
{
  {
    constexpr typeManipulation::CArray< camp::idx_t, 1 > carray = typeManipulation::asArray( RAJA::PERM_I {} );
    constexpr std::array< camp::idx_t, 1 > stdarray = RAJA::as_array< RAJA::PERM_I >::get();
    static_assert( carray[ 0 ] == stdarray[ 0 ], "Should be true." );
  }

  {
    constexpr typeManipulation::CArray< camp::idx_t, 2 > carray = typeManipulation::asArray( RAJA::PERM_IJ {} );
    constexpr std::array< camp::idx_t, 2 > stdarray = RAJA::as_array< RAJA::PERM_IJ >::get();
    static_assert( carray[ 0 ] == stdarray[ 0 ], "Should be true." );
    static_assert( carray[ 1 ] == stdarray[ 1 ], "Should be true." );
  }

  {
    constexpr typeManipulation::CArray< camp::idx_t, 2 > carray = typeManipulation::asArray( RAJA::PERM_JI {} );
    constexpr std::array< camp::idx_t, 2 > stdarray = RAJA::as_array< RAJA::PERM_JI >::get();
    static_assert( carray[ 0 ] == stdarray[ 0 ], "Should be true." );
    static_assert( carray[ 1 ] == stdarray[ 1 ], "Should be true." );
  }

  {
    constexpr typeManipulation::CArray< camp::idx_t, 3 > carray = typeManipulation::asArray( RAJA::PERM_IJK {} );
    constexpr std::array< camp::idx_t, 3 > stdarray = RAJA::as_array< RAJA::PERM_IJK >::get();
    static_assert( carray[ 0 ] == stdarray[ 0 ], "Should be true." );
    static_assert( carray[ 1 ] == stdarray[ 1 ], "Should be true." );
    static_assert( carray[ 2 ] == stdarray[ 2 ], "Should be true." );
  }

  {
    constexpr typeManipulation::CArray< camp::idx_t, 3 > carray = typeManipulation::asArray( RAJA::PERM_IKJ {} );
    constexpr std::array< camp::idx_t, 3 > stdarray = RAJA::as_array< RAJA::PERM_IKJ >::get();
    static_assert( carray[ 0 ] == stdarray[ 0 ], "Should be true." );
    static_assert( carray[ 1 ] == stdarray[ 1 ], "Should be true." );
    static_assert( carray[ 2 ] == stdarray[ 2 ], "Should be true." );
  }

  {
    constexpr typeManipulation::CArray< camp::idx_t, 3 > carray = typeManipulation::asArray( RAJA::PERM_JIK {} );
    constexpr std::array< camp::idx_t, 3 > stdarray = RAJA::as_array< RAJA::PERM_JIK >::get();
    static_assert( carray[ 0 ] == stdarray[ 0 ], "Should be true." );
    static_assert( carray[ 1 ] == stdarray[ 1 ], "Should be true." );
    static_assert( carray[ 2 ] == stdarray[ 2 ], "Should be true." );
  }

  {
    constexpr typeManipulation::CArray< camp::idx_t, 3 > carray = typeManipulation::asArray( RAJA::PERM_JKI {} );
    constexpr std::array< camp::idx_t, 3 > stdarray = RAJA::as_array< RAJA::PERM_JKI >::get();
    static_assert( carray[ 0 ] == stdarray[ 0 ], "Should be true." );
    static_assert( carray[ 1 ] == stdarray[ 1 ], "Should be true." );
    static_assert( carray[ 2 ] == stdarray[ 2 ], "Should be true." );
  }

  {
    constexpr typeManipulation::CArray< camp::idx_t, 3 > carray = typeManipulation::asArray( RAJA::PERM_KIJ {} );
    constexpr std::array< camp::idx_t, 3 > stdarray = RAJA::as_array< RAJA::PERM_KIJ >::get();
    static_assert( carray[ 0 ] == stdarray[ 0 ], "Should be true." );
    static_assert( carray[ 1 ] == stdarray[ 1 ], "Should be true." );
    static_assert( carray[ 2 ] == stdarray[ 2 ], "Should be true." );
  }

  {
    constexpr typeManipulation::CArray< camp::idx_t, 3 > carray = typeManipulation::asArray( RAJA::PERM_KJI {} );
    constexpr std::array< camp::idx_t, 3 > stdarray = RAJA::as_array< RAJA::PERM_KJI >::get();
    static_assert( carray[ 0 ] == stdarray[ 0 ], "Should be true." );
    static_assert( carray[ 1 ] == stdarray[ 1 ], "Should be true." );
    static_assert( carray[ 2 ] == stdarray[ 2 ], "Should be true." );
  }
}

TEST( typeManipulation, convertSize )
{
  EXPECT_EQ( ( typeManipulation::convertSize< char, int >( 40 ) ), 40 * sizeof( int ) / sizeof( char ) );

  EXPECT_DEATH_IF_SUPPORTED( ( typeManipulation::convertSize< double, int >( 41 ) ), "" );

  EXPECT_EQ( ( typeManipulation::convertSize< float[ 2 ], float >( 10 ) ), 10 * sizeof( float ) / sizeof( float[ 2 ] ) );

  EXPECT_EQ( ( typeManipulation::convertSize< float, float[ 2 ] >( 13 ) ), 13 * sizeof( float[ 2 ] ) / sizeof( float ) );
}

} // namespace testing
} // namespace LvArray

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
