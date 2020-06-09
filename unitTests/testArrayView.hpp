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
  static constexpr int NDIM = ARRAY::ndim;

  using ViewType = std::remove_const_t< typename ARRAY::ViewType >;
  using ViewTypeConst = std::remove_const_t< typename ARRAY::ViewTypeConst >;

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
      EXPECT_EQ( &view, &view.toView() );
      ParentClass::compare( view.toView(), *array, true );
    }

    {
      ViewTypeConst const constView = array->toViewConst();
      EXPECT_EQ( &constView, &constView.toView() );
      ParentClass::compare( constView.toView(), *array, true );
    }
  }

  static void toViewConst()
  {
    std::unique_ptr< ARRAY > array = ParentClass::sizedConstructor();

    {
      ViewType const view = array->toView();
      EXPECT_EQ( static_cast< void const * >( &view ), static_cast< void const * >( &view.toViewConst() ) );
      ParentClass::compare( view.toViewConst(), *array, true );
    }

    {
      ViewTypeConst const constView = array->toViewConst();
      EXPECT_EQ( &constView, &constView.toViewConst() );
      ParentClass::compare( constView.toViewConst(), *array, true );
    }
  }

  static void udcToViewConst()
  {
    std::unique_ptr< ARRAY > array = ParentClass::sizedConstructor();

    {
      ViewType const view = array->toView();
      ViewTypeConst const & view2 = view;
      EXPECT_EQ( static_cast< void const * >( &view2 ), static_cast< void const * >( &view ) );
      ParentClass::compare( view2, *array, true );
    }

    {
      ViewTypeConst const constView = array->toViewConst();
      ViewTypeConst const & view2 = constView;
      EXPECT_EQ( &view2, &constView );
      ParentClass::compare( view2, *array, true );
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
  static constexpr int NDIM = ARRAY::ndim;

  using ParentClass = ArrayViewTest< ARRAY >;

  using typename ParentClass::ViewType;
  using typename ParentClass::ViewTypeConst;
  using typename ParentClass::SliceType;
  using typename ParentClass::SliceTypeConst;

  static void modifyInKernel()
  {
    std::unique_ptr< ARRAY > array = ParentClass::sizedConstructor();
    ViewType const & view = array->toView();
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
    ViewTypeConst const & view = array->toViewConst();
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

    array->move( MemorySpace::CPU, false );

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

    INDEX_TYPE sizes[ NDIM ];
    INDEX_TYPE strides[ NDIM ];
    for( int dim = 0; dim < NDIM; ++dim )
    {
      sizes[ dim ] = array->size( dim );
      strides[ dim ] = array->strides()[ dim ];
    }

    forall< POLICY >( array->size( 0 ), [devicePointer, sizes, strides] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          SliceTypeConst slice( devicePointer, sizes, strides );
          checkFillDevice( slice, i );
        } );

    // Set the data outside of a kernel.
    for( INDEX_TYPE i = 0; i < array->size(); ++i )
    { hostPointer[ i ] += hostPointer[ i ]; }

    array->move( MemorySpace::CPU, false );
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

    ViewType const & view = array->toView();
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

    array->move( MemorySpace::CPU );
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

    array->move( MemorySpace::CPU, false );

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
    array.move( MemorySpace::CPU );
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

  template< int DIM, int USD >
  static LVARRAY_HOST_DEVICE void checkFillDevice( ArraySlice< T const, DIM, USD, INDEX_TYPE > const slice,
                                                   INDEX_TYPE const i )
  { forValuesInSliceWithIndices( slice[ i ], CheckIndices< ArraySlice< T const, DIM, USD, INDEX_TYPE > >( slice, i ) ); }
};

using ArrayViewPolicyTestTypes = ::testing::Types<
  std::pair< Array< int, 1, RAJA::PERM_I, INDEX_TYPE, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< int, 2, RAJA::PERM_IJ, INDEX_TYPE, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< int, 2, RAJA::PERM_JI, INDEX_TYPE, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< int, 3, RAJA::PERM_IJK, INDEX_TYPE, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< int, 3, RAJA::PERM_IKJ, INDEX_TYPE, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< int, 3, RAJA::PERM_JIK, INDEX_TYPE, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< int, 3, RAJA::PERM_JKI, INDEX_TYPE, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< int, 3, RAJA::PERM_KIJ, INDEX_TYPE, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< int, 3, RAJA::PERM_KJI, INDEX_TYPE, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< int, 4, RAJA::PERM_IJKL, INDEX_TYPE, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< int, 4, RAJA::PERM_LKJI, INDEX_TYPE, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< Tensor, 1, RAJA::PERM_I, INDEX_TYPE, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< Tensor, 2, RAJA::PERM_IJ, INDEX_TYPE, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< Tensor, 2, RAJA::PERM_JI, INDEX_TYPE, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< Tensor, 3, RAJA::PERM_IJK, INDEX_TYPE, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< Tensor, 3, RAJA::PERM_KJI, INDEX_TYPE, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< Tensor, 4, RAJA::PERM_IJKL, INDEX_TYPE, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< Array< Tensor, 4, RAJA::PERM_LKJI, INDEX_TYPE, DEFAULT_BUFFER >, serialPolicy >

#if defined(USE_CUDA) && defined(USE_CHAI)
  , std::pair< Array< int, 1, RAJA::PERM_I, INDEX_TYPE, NewChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< int, 2, RAJA::PERM_IJ, INDEX_TYPE, NewChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< int, 2, RAJA::PERM_JI, INDEX_TYPE, NewChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< int, 3, RAJA::PERM_IJK, INDEX_TYPE, NewChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< int, 3, RAJA::PERM_IKJ, INDEX_TYPE, NewChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< int, 3, RAJA::PERM_JIK, INDEX_TYPE, NewChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< int, 3, RAJA::PERM_JKI, INDEX_TYPE, NewChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< int, 3, RAJA::PERM_KIJ, INDEX_TYPE, NewChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< int, 3, RAJA::PERM_KJI, INDEX_TYPE, NewChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< int, 4, RAJA::PERM_IJKL, INDEX_TYPE, NewChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< int, 4, RAJA::PERM_LKJI, INDEX_TYPE, NewChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< Tensor, 1, RAJA::PERM_I, INDEX_TYPE, NewChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< Tensor, 2, RAJA::PERM_IJ, INDEX_TYPE, NewChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< Tensor, 2, RAJA::PERM_JI, INDEX_TYPE, NewChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< Tensor, 3, RAJA::PERM_IJK, INDEX_TYPE, NewChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< Tensor, 3, RAJA::PERM_KJI, INDEX_TYPE, NewChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< Tensor, 4, RAJA::PERM_IJKL, INDEX_TYPE, NewChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array< Tensor, 4, RAJA::PERM_LKJI, INDEX_TYPE, NewChaiBuffer >, parallelDevicePolicy< 32 > >
#endif
  >;

TYPED_TEST_SUITE( ArrayViewPolicyTest, ArrayViewPolicyTestTypes, );

} // namespace testing
} // namespace LvArray
