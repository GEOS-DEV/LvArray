/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "testArray.hpp"

namespace LvArray
{
namespace testing
{

template< typename ARRAY >
class ArrayViewTest : public ArrayTest< ARRAY >
{
public:

  using T = typename ARRAY::value_type;
  static constexpr int NDIM = ARRAY::NDIM;

  using ViewType = typeManipulation::ViewType< ARRAY >;
  using ViewTypeConst = typeManipulation::ViewTypeConst< ARRAY >;

  using SliceType = std::remove_const_t< typename ARRAY::SliceType >;
  using SliceTypeConst = std::remove_const_t< typename ARRAY::SliceTypeConst >;

  using ParentClass = ArrayTest< ARRAY >;

  static void defaultConstructor()
  {
    {
      ViewType view;
      EXPECT_TRUE( view.empty() );

      for( int i = 0; i < NDIM; ++i )
      {
        EXPECT_EQ( view.size( i ), 0 );
      }

      EXPECT_EQ( view.size(), 0 );
      EXPECT_EQ( view.data(), nullptr );
    }

    {
      ViewTypeConst view;
      EXPECT_TRUE( view.empty() );

      for( int i = 0; i < NDIM; ++i )
      {
        EXPECT_EQ( view.size( i ), 0 );
      }

      EXPECT_EQ( view.size(), 0 );
      EXPECT_EQ( view.data(), nullptr );
    }
  }

  static void copyConstructor()
  {
    std::unique_ptr< ARRAY > array = ParentClass::sizedConstructor();

    {
      ViewType const view = array->toView();
      ParentClass::compare( view, *array, true );
    }

    {
      ViewTypeConst const constView = array->toViewConst();
      ParentClass::compare( constView, *array, true );
    }
  }

  static void copyAssignmentOperator()
  {
    std::unique_ptr< ARRAY > array = ParentClass::sizedConstructor();

    {
      ViewType view;
      view = array->toView();
      ParentClass::compare( view, *array, true );
    }

    {
      ViewTypeConst constView;
      constView = array->toViewConst();
      ParentClass::compare( constView, *array, true );
    }
  }

  static void toView()
  {
    std::unique_ptr< ARRAY > array = ParentClass::sizedConstructor();

    {
      ViewType const view = array->toView();
      ParentClass::compare( view.toView(), *array, true );
    }

    {
      ViewTypeConst const constView = array->toViewConst();
      ParentClass::compare( constView.toView(), *array, true );
    }
  }

  static void toViewConst()
  {
    std::unique_ptr< ARRAY > array = ParentClass::sizedConstructor();

    {
      ViewType const view = array->toView();
      ParentClass::compare( view.toViewConst(), *array, true );
    }

    {
      ViewTypeConst const constView = array->toViewConst();
      ParentClass::compare( constView.toViewConst(), *array, true );
    }
  }

  static void toSlice()
  {
    std::unique_ptr< ARRAY > array = ParentClass::sizedConstructor();

    {
      ViewType const view = array->toView();
      SliceType const slice = view.toSlice();
      ParentClass::compare( slice, *array, true );
    }

    {
      ViewTypeConst const constView = array->toViewConst();
      SliceTypeConst const constSlice = constView.toSlice();
      ParentClass::compare( constSlice, *array, true );
    }
  }

  static void toSliceConst()
  {
    std::unique_ptr< ARRAY > array = ParentClass::sizedConstructor();

    {
      ViewType const view = array->toView();
      SliceTypeConst const slice = view.toSliceConst();
      ParentClass::compare( slice, *array, true );
    }

    {
      ViewTypeConst const constView = array->toViewConst();
      SliceTypeConst const constSlice = constView.toSliceConst();
      ParentClass::compare( constSlice, *array, true );
    }
  }

  static void udcToViewConst()
  {
    std::unique_ptr< ARRAY > array = ParentClass::sizedConstructor();

    {
      ViewType const view = array->toView();
      ViewTypeConst const view2 = view;
      ParentClass::compare( view2, *array, true );
    }

    {
      ViewTypeConst const constView = array->toViewConst();
      ViewTypeConst const view2 = constView;
      ParentClass::compare( view2, *array, true );
    }
  }

  static void udcToSlice()
  {
    std::unique_ptr< ARRAY > array = ParentClass::sizedConstructor();

    {
      ViewType const view = array->toView();
      SliceType const slice = view;
      ParentClass::compare( slice, *array, true );
    }

    {
      ViewTypeConst const constView = array->toViewConst();
      SliceTypeConst const constSlice = constView;
      ParentClass::compare( constSlice, *array, true );
    }
  }

  static void udcToSliceConst()
  {
    std::unique_ptr< ARRAY > array = ParentClass::sizedConstructor();

    {
      ViewType const view = array->toView();
      SliceTypeConst const slice = view;
      ParentClass::compare( slice, *array, true );
    }

    {
      ViewTypeConst const constView = array->toViewConst();
      SliceTypeConst const constSlice = constView;
      ParentClass::compare( constSlice, *array, true );
    }
  }

  IS_VALID_EXPRESSION( hasToSlice, T, std::declval< T >().toSlice() );
};

using ArrayViewTestTypes = ArrayTestTypes;
TYPED_TEST_SUITE( ArrayViewTest, ArrayViewTestTypes, );


template< typename ARRAY_POLICY_PAIR >
class ArrayViewPolicyTest : public ArrayViewTest< typename ARRAY_POLICY_PAIR::first_type >
{
public:
  using ARRAY = typename ARRAY_POLICY_PAIR::first_type;
  using POLICY = typename ARRAY_POLICY_PAIR::second_type;

  using T = typename ARRAY::value_type;
  static constexpr int NDIM = ARRAY::NDIM;

  using ParentClass = ArrayViewTest< ARRAY >;

  using typename ParentClass::ViewType;
  using typename ParentClass::ViewTypeConst;
  using typename ParentClass::SliceType;
  using typename ParentClass::SliceTypeConst;

  static void modifyInKernel()
  {
    std::unique_ptr< ARRAY > array = ParentClass::sizedConstructor();
    ViewType const view = array->toView();
    forall< POLICY >( array->size(), [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          view.data()[ i ] += view.data()[ i ];
        } );

    forall< serialPolicy >( array->size( 0 ), [view] ( INDEX_TYPE const i )
    {
      forValuesInSliceWithIndices( view[ i ], [i] ( T const & value, auto const ... indices )
      {
        T expectedValue = T( ParentClass::getTestingLinearIndex( i, indices ... ) );
        expectedValue += expectedValue;
        EXPECT_EQ( value, expectedValue );
      } );
    } );
  }

  static void readInKernel()
  {
    std::unique_ptr< ARRAY > array = ParentClass::sizedConstructor();
    ViewTypeConst const view = array->toViewConst();
    forall< POLICY >( array->size( 0 ), [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          checkFillDevice( view, i );
        } );

    // Set the data outside of a kernel.
    for( INDEX_TYPE i = 0; i < array->size(); ++i )
    { array->data()[ i ] += array->data()[ i ]; }

    forall< serialPolicy >( array->size( 0 ), [view] ( INDEX_TYPE const i )
    {
      forValuesInSliceWithIndices( view[ i ], [i] ( T const & value, auto const ... indices )
      {
        T expectedValue = T( ParentClass::getTestingLinearIndex( i, indices ... ) );
        expectedValue += expectedValue;
        EXPECT_EQ( value, expectedValue );
      } );
    } );
  }

  static void move()
  {
    std::unique_ptr< ARRAY > array = ParentClass::sizedConstructor();
    array->move( RAJAHelper< POLICY >::space );
    T * const dataPointer = array->data();
    forall< POLICY >( array->size(), [dataPointer] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          dataPointer[ i ] += dataPointer[ i ];
        } );

    array->move( MemorySpace::host, false );

    forValuesInSliceWithIndices( array->toSliceConst(), [] ( T const & value, auto const ... indices )
    {
      T expectedValue = T( ParentClass::getTestingLinearIndex( indices ... ) );
      expectedValue += expectedValue;
      EXPECT_EQ( value, expectedValue );
    } );
  }

  static void moveNoTouch()
  {
    std::unique_ptr< ARRAY > array = ParentClass::sizedConstructor();
    T * const hostPointer = array->data();
    array->move( RAJAHelper< POLICY >::space, false );
    T const * const devicePointer = array->data();
    auto layout = array->layout();

    forall< POLICY >( array->size( 0 ), [devicePointer, layout] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          SliceTypeConst slice( devicePointer, layout );
          checkFillDevice( slice, i );
        } );

    // Set the data outside of a kernel.
    for( INDEX_TYPE i = 0; i < array->size(); ++i )
    { hostPointer[ i ] += hostPointer[ i ]; }

    array->move( MemorySpace::host, false );
    EXPECT_EQ( array->data(), hostPointer );
    forValuesInSliceWithIndices( array->toSliceConst(), [] ( T const & value, auto const ... indices )
    {
      T expectedValue = T( ParentClass::getTestingLinearIndex( indices ... ) );
      expectedValue += expectedValue;
      EXPECT_EQ( value, expectedValue );
    } );
  }

  static void modifyInMultipleKernels()
  {
    std::unique_ptr< ARRAY > array = ParentClass::sizedConstructor();

    ViewType const view = array->toView();
    forall< POLICY >( array->size(), [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          view.data()[ i ] += view.data()[ i ];
        } );

    forall< POLICY >( array->size(), [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          view.data()[ i ] += view.data()[ i ];
        } );

    forall< serialPolicy >( array->size(), [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          view.data()[ i ] += view.data()[ i ];
        } );

    forall< POLICY >( array->size(), [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          view.data()[ i ] += view.data()[ i ];
        } );

    forall< serialPolicy >( array->size( 0 ), [view] ( INDEX_TYPE const i )
    {
      forValuesInSliceWithIndices( view[ i ], [i] ( T const & value, auto const ... indices )
      {
        T expectedValue = T( ParentClass::getTestingLinearIndex( i, indices ... ) );
        expectedValue += expectedValue;
        expectedValue += expectedValue;
        expectedValue += expectedValue;
        expectedValue += expectedValue;
        EXPECT_EQ( value, expectedValue );
      } );
    } );
  }

  static void moveMultipleTimes()
  {
    std::unique_ptr< ARRAY > array = ParentClass::sizedConstructor();

    array->move( RAJAHelper< POLICY >::space );
    T * dataPointer = array->data();
    forall< POLICY >( array->size(), [dataPointer] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          dataPointer[ i ] += dataPointer[ i ];
        } );

    forall< POLICY >( array->size(), [dataPointer] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          dataPointer[ i ] += dataPointer[ i ];
        } );

    array->move( MemorySpace::host );
    dataPointer = array->data();
    forall< serialPolicy >( array->size(), [dataPointer] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          dataPointer[ i ] += dataPointer[ i ];
        } );

    array->move( RAJAHelper< POLICY >::space );
    dataPointer = array->data();
    forall< POLICY >( array->size(), [dataPointer] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          dataPointer[ i ] += dataPointer[ i ];
        } );

    array->move( MemorySpace::host, false );

    forValuesInSliceWithIndices( array->toSliceConst(), [] ( T const & value, auto const ... indices )
    {
      T expectedValue = T( ParentClass::getTestingLinearIndex( indices ... ) );
      expectedValue += expectedValue;
      expectedValue += expectedValue;
      expectedValue += expectedValue;
      expectedValue += expectedValue;
      EXPECT_EQ( value, expectedValue );
    } );
  }

  static void emptyMove()
  {
    ARRAY array;
    array.move( RAJAHelper< POLICY >::space );
    array.move( MemorySpace::host );
  }

  static void setValues()
  {
    T const value = T( 3.14 );

    std::unique_ptr< ARRAY > array = ParentClass::sizedConstructor();

    T const * const initialPtr = array->data();
    INDEX_TYPE totalSize = 1;
    std::array< INDEX_TYPE, NDIM > sizes;
    for( int dim = 0; dim < NDIM; ++dim )
    {
      sizes[ dim ] = array->size( dim );
      totalSize *= array->size( dim );
    }

    array->template setValues< POLICY >( value );

    ViewTypeConst const view = array->toViewConst();
    forall< POLICY >( array->size(), [view, value] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          PORTABLE_EXPECT_EQ( view.data()[ i ], value );
        } );

    EXPECT_EQ( array->size(), totalSize );
    EXPECT_EQ( array->capacity(), totalSize );
    EXPECT_EQ( array->data(), initialPtr );

    for( int dim = 0; dim < NDIM; ++dim )
    { EXPECT_EQ( array->size( dim ), sizes[ dim ] ); }
  }

  static void zero()
  {
    std::unique_ptr< ARRAY > array = ParentClass::sizedConstructor();

    INDEX_TYPE totalSize = 1;
    std::array< INDEX_TYPE, NDIM > sizes;
    for( int dim = 0; dim < NDIM; ++dim )
    {
      sizes[ dim ] = array->size( dim );
      totalSize *= array->size( dim );
    }

    array->move( RAJAHelper< POLICY >::space );
    array->zero();

    ViewTypeConst const view = array->toViewConst();
    forall< POLICY >( array->size(), [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          PORTABLE_EXPECT_EQ( view.data()[ i ], T{} );
        } );

    EXPECT_EQ( array->size(), totalSize );
    EXPECT_EQ( array->capacity(), totalSize );

    for( int dim = 0; dim < NDIM; ++dim )
    { EXPECT_EQ( array->size( dim ), sizes[ dim ] ); }
  }

  static void setValuesFromView()
  {
    ARRAY array;
    std::unique_ptr< ARRAY > arrayToCopy = ParentClass::sizedConstructor();

    array.resize( arrayToCopy->layout().extent() );
    T const * const initialPtr = array.data();

    ViewTypeConst const viewToCopy = arrayToCopy->toViewConst();
    array.template setValues< POLICY >( arrayToCopy->toViewConst() );

    ViewTypeConst const view = array.toViewConst();
    forall< POLICY >( array.size(), [view, viewToCopy] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          PORTABLE_EXPECT_EQ( view.data()[ i ], viewToCopy.data()[ i ] );
        } );


    EXPECT_EQ( array.size(), arrayToCopy->size() );
    EXPECT_EQ( array.capacity(), arrayToCopy->size() );
    EXPECT_EQ( array.data(), initialPtr );

    array.move( MemorySpace::host );
    ParentClass::checkFill( array );
  }

protected:

  template< typename ARRAY >
  class CheckIndices
  {
  public:
    LVARRAY_HOST_DEVICE CheckIndices( ARRAY const slice, INDEX_TYPE const i ):
      m_slice( slice ),
      m_i( i )
    {}

    template< typename ... INDICES >
    LVARRAY_HOST_DEVICE void operator()( T const & value, INDICES const ... indices )
    {
      PORTABLE_EXPECT_EQ( value, T( ParentClass::getTestingLinearIndex( m_i, indices ... ) ) );
      PORTABLE_EXPECT_EQ( &value, &m_slice( m_i, indices ... ) );
    }

  private:
    ARRAY const m_slice;
    INDEX_TYPE const m_i;
  };

  static LVARRAY_HOST_DEVICE void checkFillDevice( ViewTypeConst const & view,
                                                   INDEX_TYPE const i )
  { forValuesInSliceWithIndices( view[ i ], CheckIndices< ViewTypeConst >( view, i ) ); }

  //template< int DIM, int USD >
  static LVARRAY_HOST_DEVICE void checkFillDevice( SliceTypeConst const slice,
                                                   INDEX_TYPE const i )
  { forValuesInSliceWithIndices( slice[ i ], CheckIndices< SliceTypeConst >( slice, i ) ); }
};

using ArrayViewPolicyTestTypes = ::testing::Types<
  std::pair< Array< int, DynamicExtent< 1, INDEX_TYPE >, RAJA::PERM_I, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< int, DynamicExtent< 2, INDEX_TYPE >, RAJA::PERM_IJ, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< int, DynamicExtent< 2, INDEX_TYPE >, RAJA::PERM_JI, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< int, DynamicExtent< 3, INDEX_TYPE >, RAJA::PERM_IJK, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< int, DynamicExtent< 3, INDEX_TYPE >, RAJA::PERM_IKJ, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< int, DynamicExtent< 3, INDEX_TYPE >, RAJA::PERM_JIK, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< int, DynamicExtent< 3, INDEX_TYPE >, RAJA::PERM_JKI, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< int, DynamicExtent< 3, INDEX_TYPE >, RAJA::PERM_KIJ, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< int, DynamicExtent< 3, INDEX_TYPE >, RAJA::PERM_KJI, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< int, DynamicExtent< 4, INDEX_TYPE >, RAJA::PERM_IJKL, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< int, DynamicExtent< 4, INDEX_TYPE >, RAJA::PERM_LKJI, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< Tensor, DynamicExtent< 1, INDEX_TYPE >, RAJA::PERM_I, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< Tensor, DynamicExtent< 2, INDEX_TYPE >, RAJA::PERM_IJ, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< Tensor, DynamicExtent< 2, INDEX_TYPE >, RAJA::PERM_JI, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< Tensor, DynamicExtent< 3, INDEX_TYPE >, RAJA::PERM_IJK, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< Tensor, DynamicExtent< 3, INDEX_TYPE >, RAJA::PERM_KJI, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< Tensor, DynamicExtent< 4, INDEX_TYPE >, RAJA::PERM_IJKL, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< Tensor, DynamicExtent< 4, INDEX_TYPE >, RAJA::PERM_LKJI, DEFAULT_BUFFER >, serialPolicy >

#if defined(LVARRAY_USE_CUDA) && defined(LVARRAY_USE_CHAI)
  , std::pair< Array< int, DynamicExtent< 1, INDEX_TYPE >, RAJA::PERM_I, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< int, DynamicExtent< 2, INDEX_TYPE >, RAJA::PERM_IJ, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< int, DynamicExtent< 2, INDEX_TYPE >, RAJA::PERM_JI, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< int, DynamicExtent< 3, INDEX_TYPE >, RAJA::PERM_IJK, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< int, DynamicExtent< 3, INDEX_TYPE >, RAJA::PERM_IKJ, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< int, DynamicExtent< 3, INDEX_TYPE >, RAJA::PERM_JIK, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< int, DynamicExtent< 3, INDEX_TYPE >, RAJA::PERM_JKI, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< int, DynamicExtent< 3, INDEX_TYPE >, RAJA::PERM_KIJ, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< int, DynamicExtent< 3, INDEX_TYPE >, RAJA::PERM_KJI, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< int, DynamicExtent< 4, INDEX_TYPE >, RAJA::PERM_IJKL, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< int, DynamicExtent< 4, INDEX_TYPE >, RAJA::PERM_LKJI, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< Tensor, DynamicExtent< 1, INDEX_TYPE >, RAJA::PERM_I, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< Tensor, DynamicExtent< 2, INDEX_TYPE >, RAJA::PERM_IJ, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< Tensor, DynamicExtent< 2, INDEX_TYPE >, RAJA::PERM_JI, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< Tensor, DynamicExtent< 3, INDEX_TYPE >, RAJA::PERM_IJK, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< Tensor, DynamicExtent< 3, INDEX_TYPE >, RAJA::PERM_KJI, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< Tensor, DynamicExtent< 4, INDEX_TYPE >, RAJA::PERM_IJKL, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< Tensor, DynamicExtent< 4, INDEX_TYPE >, RAJA::PERM_LKJI, ChaiBuffer >, parallelDevicePolicy< 32 > >
#endif
  >;

TYPED_TEST_SUITE( ArrayViewPolicyTest, ArrayViewPolicyTestTypes, );

} // namespace testing
} // namespace LvArray
