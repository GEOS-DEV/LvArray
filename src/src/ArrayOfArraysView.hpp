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

/**
 * @file ArrayOfArraysView.hpp
 */

#ifndef ARRAYOFARRAYSVIEW_HPP_
#define ARRAYOFARRAYSVIEW_HPP_

#include "ChaiVector.hpp"
#include "arrayManipulation.hpp"
#include "ArraySlice.hpp"
#include "templateHelpers.hpp"

#ifdef USE_ARRAY_BOUNDS_CHECK

#undef CONSTEXPRFUNC
#define CONSTEXPRFUNC

#define ARRAYOFARRAYS_CHECK_BOUNDS( i ) \
  GEOS_ERROR_IF( !arrayManipulation::isPositive( i ) || i >= this->size(), \
                 "Bounds Check Failed: i=" << i << " size()=" << this->size() )

#define ARRAYOFARRAYS_CHECK_BOUNDS2( i, j ) \
  GEOS_ERROR_IF( !arrayManipulation::isPositive( i ) || i >= this->size() || \
                 !arrayManipulation::isPositive( j ) || j >= this->sizeOfArray( i ), \
                 "Bounds Check Failed: i=" << i << " size()=" << this->size() << \
                 " j=" << j << " sizeOfArray( i )=" << this->sizeOfArray( i ) )

#define ARRAYOFARRAYS_CHECK_INSERT_BOUNDS( i ) \
  GEOS_ERROR_IF( !arrayManipulation::isPositive( i ) || i > this->size(), \
                 "Insert Bounds Check Failed: i=" << i << " size()=" << this->size() )

#define ARRAYOFARRAYS_CHECK_INSERT_BOUNDS2( i, j ) \
  GEOS_ERROR_IF( !arrayManipulation::isPositive( i ) || i >= this->size() || \
                 !arrayManipulation::isPositive( j ) || j > this->sizeOfArray( i ), \
                 "Insert Bounds Check Failed: i=" << i << " size()=" << this->size() << \
                 " j=" << j << " sizeOfArray( i )=" << this->sizeOfArray( i ) )

#define ARRAYOFARRAYS_CAPACITY_CHECK( i, increase ) \
  GEOS_ERROR_IF( this->sizeOfArray( i ) + increase > this->capacityOfArray( i ), \
                 "Capacity Check Failed: i=" << i << " increase=" << increase << \
                 " sizeOfArray( i )=" << this->sizeOfArray( i ) << " capacityOfArray( i )=" << \
                 this->capacityOfArray( i ) )

#else // USE_ARRAY_BOUNDS_CHECK

#define ARRAYOFARRAYS_CHECK_BOUNDS( i )
#define ARRAYOFARRAYS_CHECK_BOUNDS2( i, j )
#define ARRAYOFARRAYS_CHECK_INSERT_BOUNDS( i )
#define ARRAYOFARRAYS_CHECK_INSERT_BOUNDS2( i, j )
#define ARRAYOFARRAYS_CAPACITY_CHECK( i, increase )

#endif // USE_ARRAY_BOUNDS_CHECK

namespace LvArray
{

namespace internal
{
template< class U >
using PairOfVectors = std::pair< ChaiVector< U > &, ChaiVector< U const > const & >;
}

/**
 * @class ArrayOfArraysView
 * @brief This class provides a view into an array of arrays like object.
 * @tparam T the type stored in the arrays.
 * @tparam INDEX_TYPE the integer to use for indexing.
 * @tparam CONST_SIZES true iff the size of each array is constant.
 *
 * When INDEX_TYPE is const m_offsets is not copied between memory spaces.
 * When accessing this class directly (not through an ArrayOfArrays object)
 * INDEX_TYPE should always be const since ArrayOfArraysView is not allowed to modify the offsets.
 *
 * When CONST_SIZES is true m_sizes is not copied between memory spaces.
 */
template< class T, class INDEX_TYPE=std::ptrdiff_t, bool CONST_SIZES=false >
class ArrayOfArraysView
#ifdef USE_CHAI
  : public chai::CHAICopyable
#endif
{
public:
  static_assert( !std::is_const< T >::value || (std::is_const< INDEX_TYPE >::value && CONST_SIZES),
                 "When T is const INDEX_TYPE must also be const and CONST_SIZES must be true" );
  static_assert( std::is_integral< INDEX_TYPE >::value, "INDEX_TYPE must be integral." );

  using INDEX_TYPE_NC = typename std::remove_const< INDEX_TYPE >::type;

  /**
   * @brief Default copy constructor. Performs a shallow copy and calls the
   *        chai::ManagedArray copy constructor.
   * @param [in] src the ArrayOfArraysView to be copied.
   */
  inline
  ArrayOfArraysView( ArrayOfArraysView const & src ) = default;

  /**
   * @brief Default move constructor.
   * @param [in/out] src the ArrayOfArraysView to be moved from.
   */
  inline
  ArrayOfArraysView( ArrayOfArraysView && src ) = default;

  /**
   * @brief User defined conversion to make CONST_SIZES true.
   */
  template< bool SIZES_ARE_CONST=CONST_SIZES >
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  operator typename std::enable_if< !SIZES_ARE_CONST,
                                    ArrayOfArraysView< T, INDEX_TYPE const, true > const & >::type
    () const restrict_this
  { return reinterpret_cast< ArrayOfArraysView< T, INDEX_TYPE const, true > const & >(*this); }

  /**
   * @brief Method to make CONST_SIZES true. Use this method when the above UDC
   *        isn't invoked, this usually occurs with template argument deduction.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  ArrayOfArraysView< T, INDEX_TYPE const, true > const & toViewC() const restrict_this
  { return *this; }

  /**
   * @brief User defined conversion to move from T to T const.
   */
  template< class U=T >
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  operator typename std::enable_if< !std::is_const< U >::value,
                                    ArrayOfArraysView< T const, INDEX_TYPE const, true > const & >::type
    () const restrict_this
  { return reinterpret_cast< ArrayOfArraysView< T const, INDEX_TYPE const, true > const & >(*this); }

  /**
   * @brief Method to convert T to T const. Use this method when the above UDC
   *        isn't invoked, this usually occurs with template argument deduction.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  ArrayOfArraysView< T const, INDEX_TYPE const, true > const & toViewCC() const restrict_this
  { return *this; }

  /**
   * @brief Default copy assignment operator, this does a shallow copy.
   * @param [in] src the SparsityPatternView to be copied from.
   */
  inline
  ArrayOfArraysView & operator=( ArrayOfArraysView const & src ) = default;

  /**
   * @brief Default move assignment operator, this does a shallow copy.
   * @param [in/out] src the SparsityPatternView to be moved from.
   */
  inline
  ArrayOfArraysView & operator=( ArrayOfArraysView && src ) = default;

  /**
   * @brief Return the number of arrays.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  INDEX_TYPE_NC size() const restrict_this
  { return m_sizes.size(); }

  /**
   * @brief Return the number of (zero length) arrays that can be stored before reallocation.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  INDEX_TYPE_NC capacity() const restrict_this
  {
    GEOS_ASSERT( m_sizes.capacity() < m_offsets.capacity());
    return m_sizes.capacity();
  }

  /**
   * @brief Return the size of the given array.
   * @param [in] i the array to querry.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  INDEX_TYPE_NC sizeOfArray( INDEX_TYPE const i ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );
    return m_sizes[i];
  }

  /**
   * @brief Return the capacity of the given array.
   * @param [in] i the array to querry.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  INDEX_TYPE_NC capacityOfArray( INDEX_TYPE const i ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );
    return m_offsets[i + 1] - m_offsets[i];
  }

  /**
   * @brief Return an ArraySlice1d (pointer) to the values of the given array.
   * @param [in] i the array to access.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  ArraySlice1d_rval< T, INDEX_TYPE_NC > operator[]( INDEX_TYPE const i ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );
    return createArraySlice1d< T, INDEX_TYPE_NC >( m_values.data() + m_offsets[i], &m_sizes[i], nullptr );
  }

  /**
   * @brief Return a reference to the value at the given position in the given array.
   * @param [in] i the array to access.
   * @param [in] j the index within the array to access.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  T & operator()( INDEX_TYPE const i, INDEX_TYPE const j ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS2( i, j );
    return m_values[m_offsets[i] + j];
  }

  /**
   * @brief Append a value to an array.
   * @param [in] i the array to append to.
   * @param [in] value the value to append.
   *
   * @note Since the ArrayOfArraysView can't do reallocation or shift the offsets it is
   *       up to the user to ensure that the given row has enough space for the new entries.
   *       If USE_ARRAY_BOUNDS_CHECK is defined a lack of space will result in an error,
   *       otherwise the values in the subsequent array will be overwritten.
   */
  LVARRAY_HOST_DEVICE inline
  void appendToArray( INDEX_TYPE const i, T const & value ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );
    ARRAYOFARRAYS_CAPACITY_CHECK( i, 1 );

    T * const ptr = m_values.data() + m_offsets[ i ];
    arrayManipulation::append( ptr, sizeOfArray( i ), value );
    m_sizes[i]++;
  }

  /**
   * @brief Append a value to an array.
   * @param [in] i the array to append to.
   * @param [in/out] value the value to append.
   *
   * @note Since the ArrayOfArraysView can't do reallocation or shift the offsets it is
   *       up to the user to ensure that the given row has enough space for the new entries.
   *       If USE_ARRAY_BOUNDS_CHECK is defined a lack of space will result in an error,
   *       otherwise the values in the subsequent array will be overwritten.
   */
  LVARRAY_HOST_DEVICE inline
  void appendToArray( INDEX_TYPE const i, T && value ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );
    ARRAYOFARRAYS_CAPACITY_CHECK( i, 1 );

    T * const ptr = m_values.data() + m_offsets[ i ];
    arrayManipulation::append( ptr, sizeOfArray( i ), std::move( value ) );
    m_sizes[i]++;
  }

  /**
   * @brief Append values to an array.
   * @param [in] i the array to append to.
   * @param [in] values the values to append.
   * @param [in] n the number of values to append.
   *
   * @note Since the ArrayOfArraysView can't do reallocation or shift the offsets it is
   *       up to the user to ensure that the given row has enough space for the new entries.
   *       If USE_ARRAY_BOUNDS_CHECK is defined a lack of space will result in an error,
   *       otherwise the values in the subsequent array will be overwritten.
   */
  LVARRAY_HOST_DEVICE inline
  void appendToArray( INDEX_TYPE const i, T const * const values, INDEX_TYPE const n ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );
    ARRAYOFARRAYS_CAPACITY_CHECK( i, n );

    T * const ptr = m_values.data() + m_offsets[ i ];
    arrayManipulation::append( ptr, sizeOfArray( i ), values, n );
    m_sizes[i] += n;
  }

  /**
   * @brief Insert a value into an array.
   * @param [in] i the array to insert into.
   * @param [in] j the position at which to insert.
   * @param [in] value the value to insert.
   *
   * @note Since the ArrayOfArraysView can't do reallocation or shift the offsets it is
   *       up to the user to ensure that the given row has enough space for the new entries.
   *       If USE_ARRAY_BOUNDS_CHECK is defined a lack of space will result in an error,
   *       otherwise the values in the subsequent array will be overwritten.
   */
  LVARRAY_HOST_DEVICE inline
  void insertIntoArray( INDEX_TYPE const i, INDEX_TYPE const j, T const & value ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_INSERT_BOUNDS2( i, j );
    ARRAYOFARRAYS_CAPACITY_CHECK( i, 1 );

    T * const ptr = m_values.data() + m_offsets[ i ];
    arrayManipulation::insert( ptr, sizeOfArray( i ), j, value );
    m_sizes[i]++;
  }

  /**
   * @brief Insert a value into an array.
   * @param [in] i the array to insert into.
   * @param [in] j the position at which to insert.
   * @param [in/out] value the value to insert.
   *
   * @note Since the ArrayOfArraysView can't do reallocation or shift the offsets it is
   *       up to the user to ensure that the given row has enough space for the new entries.
   *       If USE_ARRAY_BOUNDS_CHECK is defined a lack of space will result in an error,
   *       otherwise the values in the subsequent array will be overwritten.
   */
  LVARRAY_HOST_DEVICE inline
  void insertIntoArray( INDEX_TYPE const i, INDEX_TYPE const j, T && value ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_INSERT_BOUNDS2( i, j );
    ARRAYOFARRAYS_CAPACITY_CHECK( i, 1 );

    T * const ptr = m_values.data() + m_offsets[ i ];
    arrayManipulation::insert( ptr, sizeOfArray( i ), j, std::move( value ) );
    m_sizes[i]++;
  }

  /**
   * @brief Insert values into an array.
   * @param [in] i the array to insert into.
   * @param [in] j the position at which to insert.
   * @param [in] values the values to insert.
   * @param [in] n the number of values to insert.
   *
   * @note Since the ArrayOfArraysView can't do reallocation or shift the offsets it is
   *       up to the user to ensure that the given row has enough space for the new entries.
   *       If USE_ARRAY_BOUNDS_CHECK is defined a lack of space will result in an error,
   *       otherwise the values in the subsequent array will be overwritten.
   */
  LVARRAY_HOST_DEVICE inline
  void insertIntoArray( INDEX_TYPE const i, INDEX_TYPE const j, T const * const values, INDEX_TYPE const n ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_INSERT_BOUNDS2( i, j );
    ARRAYOFARRAYS_CAPACITY_CHECK( i, n );

    T * const ptr = m_values.data() + m_offsets[ i ];
    arrayManipulation::insert( ptr, sizeOfArray( i ), j, values, n );
    m_sizes[i] += n;
  }

  /**
   * @brief Erase values from an array.
   * @param [in] i the array to erase values from.
   * @param [in] j the position at which to begin erasing.
   * @param [in] n the number of values to erase.
   */
  LVARRAY_HOST_DEVICE inline
  void eraseFromArray( INDEX_TYPE const i, INDEX_TYPE const j, INDEX_TYPE const n=1 ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS2( i, j );

    T * const ptr = m_values.data() + m_offsets[ i ];
    arrayManipulation::erase( ptr, sizeOfArray( i ), j, n );
    m_sizes[i] -= n;
  }

protected:

  /**
   * @brief Default constructor. Made protected since every ArrayOfArraysView should
   *        either be the base of a ArrayOfArrays or copied from another ArrayOfArraysView.
   */
  ArrayOfArraysView() = default;

  /**
   * @brief Set the number of arrays.
   * @tparam VECTORS variadic template where each type is a ChaiVector.
   * @param [in] newSize the new number of arrays.
   * @param [in] defaultArrayCapacity the default capacity for each new array.
   * @param [in/out] vectors variadic parameter pack where each argument is a ChaiVector that should be treated
   *        simlarly to m_values.
   * @note this is to be use by the non-view derived classes.
   */
  template< class ... VECTORS >
  void resize( INDEX_TYPE const newSize, INDEX_TYPE const defaultArrayCapacity, VECTORS & ... vectors )
  {
    GEOS_ASSERT( arrayManipulation::isPositive( newSize ) );

    INDEX_TYPE const prevSize = size();
    if( newSize < prevSize )
    {
      destroyValues( newSize, prevSize, vectors ... );
      m_offsets.resize( newSize + 1, 0 );
      m_sizes.resize( newSize, 0 );
      return;
    }

    // The ternary here accounts for the case where m_offsets hasn't been allocated yet (when calling from a
    // constructor).
    INDEX_TYPE const originalOffset = (prevSize == 0) ? 0 : m_offsets[prevSize];
    m_offsets.resize( newSize + 1, originalOffset );
    m_sizes.resize( newSize, 0 );

    if( defaultArrayCapacity > 0 )
    {
      for( INDEX_TYPE i = prevSize + 1 ; i < newSize + 1 ; ++i )
      {
        m_offsets[i] = originalOffset + i * defaultArrayCapacity;
      }

      INDEX_TYPE const totalSize = m_offsets[ newSize ];

      for_each_arg(
        [totalSize]( auto & vector )
        {
          vector.reserve( totalSize );
          vector.setSize( totalSize );
        },
        m_values, vectors ... );
    }
  }

  /**
   * @brief Destroy all the objects held by this array and free all associated memory.
   * @tparam VECTORS variadic template where each type is a ChaiVector.
   * @param [in/out] vectors variadic parameter pack where each argument is a ChaiVector that should be treated
   *        simlarly to m_values.
   * @note this is to be use by the non-view derived classes.
   */
  template< class ... VECTORS >
  void free( VECTORS & ... vectors )
  {
    destroyValues( 0, size(), vectors ... );

    for_each_arg(
      []( auto & vector )
      {
        vector.releaseAllocation();
      },
      m_values, vectors ...
      );

    m_sizes.free();
    m_offsets.free();
  }

  /**
   * @brief Set this ArrayOfArraysView equal to the provided arrays.
   * @tparam PAIRS_OF_VECTORS variadic template where each type is an internal::PairOfVectors.
   * @param [in] srcOffsets the source offsets array.
   * @param [in] srcSizes the source sizes array.
   * @param [in] srcValues the source values array.
   * @param [in/out] pairs variadic parameter pack where each argument is an internal::PairOfVectors where each pair
   * should be treated similarly to {m_values, srcValues}.
   * @note this is to be use by the non-view derived classes.
   */
  template< class ... PAIRS_OF_VECTORS >
  void setEqualTo( ChaiVector< INDEX_TYPE const > const & srcOffsets, ChaiVector< INDEX_TYPE const > const & srcSizes,
                   ChaiVector< T const > const & srcValues, PAIRS_OF_VECTORS & ... pairs )
  {
    destroyValues( 0, size(), pairs.first ... );

    srcOffsets.copy_into( m_offsets );
    srcSizes.copy_into( m_sizes );

    INDEX_TYPE const newNumValues = srcValues.size();

    for_each_arg(
      [newNumValues]( auto & dstVector )
      {
        dstVector.reserve( newNumValues );
        dstVector.setSize( newNumValues );
      },
      m_values, pairs.first ...
      );

    internal::PairOfVectors< T > valuesPair( m_values, srcValues );

    INDEX_TYPE const newNumArrays = size();
    for( INDEX_TYPE_NC i = 0 ; i < newNumArrays ; ++i )
    {
      INDEX_TYPE const offset = m_offsets[i];
      INDEX_TYPE const arraySize = sizeOfArray( i );
      for( INDEX_TYPE_NC j = 0 ; j < arraySize ; ++j )
      {
        for_each_arg(
          [offset, j]( auto & pair )
          {
            auto & dstVector = pair.first;
            auto const & srcVector = pair.second;
            using U = typename std::remove_reference< decltype(dstVector[0]) >::type;

            new (&dstVector[offset + j]) U( srcVector[offset + j] );
          },
          valuesPair, pairs ...
          );
      }
    }
  }

#ifdef USE_CHAI
  /**
   * @brief Move this ArrayOfArrays view to the given memory space.
   * @tparam VECTORS variadic template where each type is a ChaiVector.
   * @param [in] space the memory space to move to.
   * @param [in/out] vectors variadic parameter pack where each argument is a ChaiVector that should be treated
   *        simlarly to m_values.
   * @note this is to be use by the non-view derived classes.
   */
  template< class ... VECTORS >
  void move( chai::ExecutionSpace const space, VECTORS & ... vectors )
  {
    for_each_arg(
      [space]( auto & vector )
      {
        vector.move( space );
      },
      m_values, m_sizes, m_offsets, vectors ...
      );
  }

  /**
   * @brief Touch this ArrayOfArrays view in the given memory space.
   * @tparam VECTORS variadic template where each type is a ChaiVector.
   * @param [in] space the memory space to touch.
   * @param [in/out] vectors variadic parameter pack where each argument is a ChaiVector that should be treated
   *        simlarly to m_values.
   * @note this is to be use by the non-view derived classes.
   */
  template< class ... VECTORS >
  void registerTouch( chai::ExecutionSpace const space, VECTORS & ... vectors )
  {
    for_each_arg(
      [space]( auto & vector )
      {
        vector.registerTouch( space );
      },
      m_values, m_sizes, m_offsets, vectors ...
      );
  }
#endif

  /**
   * @brief Compress the arrays so that the values of each array are contiguous with no extra capacity in between.
   * @tparam VECTORS variadic template where each type is a ChaiVector.
   * @param [in/out] vectors variadic parameter pack where each argument is a ChaiVector that should be treated
   *        simlarly to m_values.
   * @note this is to be use by the non-view derived classes.
   * @note This method doesn't free any memory.
   */
  template< class ... VECTORS >
  void compress( VECTORS & ... vectors )
  {
    INDEX_TYPE const nArrays = size();
    for( INDEX_TYPE i = 0 ; i < nArrays - 1 ; ++i )
    {
      INDEX_TYPE const curOffset = m_offsets[ i ];
      INDEX_TYPE const nextOffset = m_offsets[ i + 1 ];
      INDEX_TYPE const sizeOfCurArray = sizeOfArray( i );
      INDEX_TYPE const sizeOfNextArray = sizeOfArray( i + 1 );

      // Shift the values in the next array down.
      for( INDEX_TYPE j = 0 ; j < sizeOfNextArray ; ++j )
      {
        for_each_arg(
          [curOffset, sizeOfCurArray, nextOffset, j]( auto & vector )
          {
            using U = typename std::remove_reference< decltype(vector[0]) >::type;
            new (&vector[curOffset + sizeOfCurArray + j]) U( std::move( vector[nextOffset + j] ));
            vector[nextOffset + j].~U();
          },
          m_values, vectors ...
          );
      }

      // And update the offsets.
      m_offsets[ i + 1 ] = curOffset + sizeOfArray( i );
    }

    // Update the last offset.
    m_offsets[ nArrays ] = m_offsets[ nArrays - 1 ] + sizeOfArray( nArrays - 1 );
  }

  /**
   * @brief Reserve space for the given number of arrays.
   * @param [in] newCapacity the new minimum capacity for the number of arrays.
   */
  void reserve( INDEX_TYPE const newCapacity )
  {
    m_offsets.reserve( newCapacity );
    m_sizes.reserve( newCapacity );
  }

  /**
   * @brief Reserve space for the given number of values.
   * @tparam VECTORS variadic template where each type is a ChaiVector.
   * @param [in] newValueCapacity the new minimum capacity for the number of values across all arrays.
   * @param [in/out] vectors variadic parameter pack where each argument is a ChaiVector that should be treated
   *        simlarly to m_values.
   */
  template< class ... VECTORS >
  void reserveValues( INDEX_TYPE const newValueCapacity, VECTORS & ... vectors )
  {
    for_each_arg(
      [newValueCapacity] ( auto & vector )
      {
        vector.reserve( newValueCapacity );
      },
      m_values, vectors ...
      );
  }

  /**
   * @brief Set the capacity of the given array.
   * @tparam VECTORS variadic template where each type is a ChaiVector.
   * @param [in] i the array to set the capacity of.
   * @param [in] newCapacity the value to set the capacity of the given array to.
   * @param [in/out] vectors variadic parameter pack where each argument is a ChaiVector that should be treated
   *        simlarly to m_values.
   * @note this is to be use by the non-view derived classes.
   */
  template< class ... VECTORS >
  void setCapacityOfArray( INDEX_TYPE const i, INDEX_TYPE const newCapacity, VECTORS & ... vectors )
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );
    GEOS_ASSERT( arrayManipulation::isPositive( newCapacity ) );

    INDEX_TYPE const numArrays = size();
    INDEX_TYPE const arrayOffset = m_offsets[i];
    INDEX_TYPE const arrayCapacity = capacityOfArray( i );
    INDEX_TYPE const capacityIncrease = newCapacity - arrayCapacity;
    if( capacityIncrease == 0 ) return;

    if( capacityIncrease > 0 )
    {
      // Increase the size of the vectors.
      INDEX_TYPE const maxOffset = m_offsets[numArrays];
      for_each_arg(
        [maxOffset, capacityIncrease]( auto & vector )
        {
          vector.dynamicResize( maxOffset + capacityIncrease );
          vector.setSize( maxOffset + capacityIncrease );
        },
        m_values, vectors ...
        );

      // Shift up the values of the arrays.
      for( INDEX_TYPE array = numArrays - 1 ; array > i ; --array )
      {
        INDEX_TYPE const curArraySize = sizeOfArray( array );
        INDEX_TYPE const curArrayOffset = m_offsets[array];

        for( INDEX_TYPE j = curArraySize - 1 ; j >= 0 ; --j )
        {
          for_each_arg(
            [curArrayOffset, capacityIncrease, j]( auto & vector )
            {
              using U = typename std::remove_reference< decltype(vector[0]) >::type;
              new (&vector[curArrayOffset + capacityIncrease + j]) U( std::move( vector[curArrayOffset + j] ));
              vector[curArrayOffset + j].~U();
            },
            m_values, vectors ...
            );
        }
      }
    }
    else
    {
      INDEX_TYPE const capacityDecrease = -capacityIncrease;

      INDEX_TYPE const prevArraySize = sizeOfArray( i );
      INDEX_TYPE const newArraySize = min( prevArraySize, newCapacity );

      // Delete the values at the end of the array.
      m_sizes[i] = newArraySize;
      for( INDEX_TYPE j = newArraySize ; j < prevArraySize ; ++j )
      {
        for_each_arg(
          [arrayOffset, j]( auto & vector )
          {
            using U = typename std::remove_reference< decltype(vector[0]) >::type;
            vector[arrayOffset + j].~U();
          },
          m_values, vectors ...
          );
      }

      // Shift down the values of subsequent arrays.
      for( INDEX_TYPE array = i + 1 ; array < numArrays ; ++array )
      {
        INDEX_TYPE const curArraySize = sizeOfArray( array );
        INDEX_TYPE const curArrayOffset = m_offsets[array];

        for( INDEX_TYPE j = 0 ; j < curArraySize ; ++j )
        {
          for_each_arg(
            [curArrayOffset, capacityDecrease, j]( auto & vector )
            {
              using U = typename std::remove_reference< decltype(vector[0]) >::type;
              new (&vector[curArrayOffset - capacityDecrease + j]) U( std::move( vector[curArrayOffset + j] ));
              vector[curArrayOffset + j].~U();
            },
            m_values, vectors ...
            );
        }
      }
    }

    // Update the offsets array
    for( INDEX_TYPE array = i + 1 ; array < numArrays + 1 ; ++array )
    {
      m_offsets[array] += capacityIncrease;
    }
  }

  // Holds the offset of each array, of length size() + 1. Array i begins at
  // m_offsets[i] and has capacity m_offsets[i+1] - m_offsets[i].
  ChaiVector< INDEX_TYPE > m_offsets;

  // Holds the size of each row, of length size().
  using SIZE_TYPE = std::conditional_t< CONST_SIZES, INDEX_TYPE const, INDEX_TYPE_NC >;
  ChaiVector< SIZE_TYPE > m_sizes;

  // Holds the values of each array, of length numValues().
  ChaiVector< T > m_values;

private:

  /**
   * @brief Destroy the values in arrays in the range [begin, end).
   * @tparam VECTORS variadic template where each type is a ChaiVector.
   * @param [in] begin the array to start with.
   * @param [in] end where to stop destroying values.
   * @param [in/out] vectors variadic parameter pack where each argument is a ChaiVector that should be treated
   *        simlarly to m_values.
   * @note this is to be use by the non-view derived classes.
   * @note this method doesn't free any memory.
   */
  template< class ... VECTORS >
  void destroyValues( INDEX_TYPE const begin, INDEX_TYPE const end, VECTORS & ... vectors )
  {
    ARRAYOFARRAYS_CHECK_INSERT_BOUNDS( begin );
    ARRAYOFARRAYS_CHECK_INSERT_BOUNDS( end );

    for( INDEX_TYPE i = begin ; i < end ; ++i )
    {
      INDEX_TYPE const offset = m_offsets[i];
      INDEX_TYPE const arraySize = sizeOfArray( i );
      for( INDEX_TYPE j = 0 ; j < arraySize ; ++j )
      {
        for_each_arg(
          [offset, j]( auto & vector )
          {
            using U = typename std::remove_reference< decltype(vector[0]) >::type;
            (vector[offset + j]).~U();
          },
          m_values, vectors ...
          );
      }
    }
  }
};

} /* namespace LvArray */

#endif /* ARRAYOFARRAYSVIEW_HPP_ */
