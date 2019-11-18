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

#pragma once

#include "ChaiBuffer.hpp"
#include "bufferManipulation.hpp"
#include "arrayManipulation.hpp"
#include "ArraySlice.hpp"
#include "templateHelpers.hpp"
#include "RAJA/RAJA.hpp"

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

#define ARRAYOFARRAYS_ATOMIC_CAPACITY_CHECK( i, previousSize, increase ) \
  GEOS_ERROR_IF( previousSize + increase > this->capacityOfArray( i ), \
                 "Capacity Check Failed: i=" << i << " increase=" << increase << \
                 " sizeOfArray( i )=" << previousSize << " capacityOfArray( i )=" << \
                 this->capacityOfArray( i ) )

#else // USE_ARRAY_BOUNDS_CHECK

#define ARRAYOFARRAYS_CHECK_BOUNDS( i )
#define ARRAYOFARRAYS_CHECK_BOUNDS2( i, j )
#define ARRAYOFARRAYS_CHECK_INSERT_BOUNDS( i )
#define ARRAYOFARRAYS_CHECK_INSERT_BOUNDS2( i, j )
#define ARRAYOFARRAYS_CAPACITY_CHECK( i, increase )
#define ARRAYOFARRAYS_ATOMIC_CAPACITY_CHECK( i, previousSize, increase )

#endif // USE_ARRAY_BOUNDS_CHECK

namespace LvArray
{

namespace internal
{
template <class U>
using PairOfBuffers = std::pair<ChaiBuffer<U> &, ChaiBuffer<U> const &>;
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
template <class T, class INDEX_TYPE=std::ptrdiff_t, bool CONST_SIZES=false>
class ArrayOfArraysView
{
  template <class U, class INDEX>
  friend class ArrayOfSets;

  template <class U, class INDEX>
  friend class ArrayOfArrays;

public:
  static_assert(!std::is_const<T>::value || (std::is_const<INDEX_TYPE>::value && CONST_SIZES),
                "When T is const INDEX_TYPE must also be const and CONST_SIZES must be true" );
  static_assert(std::is_integral<INDEX_TYPE>::value, "INDEX_TYPE must be integral.");

  using INDEX_TYPE_NC = typename std::remove_const<INDEX_TYPE>::type;

  // Holds the size of each row, of length size().
  using SIZE_TYPE = std::conditional_t<CONST_SIZES, INDEX_TYPE const, INDEX_TYPE_NC>;

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
  template<bool SIZES_ARE_CONST=CONST_SIZES>
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  operator typename std::enable_if<!SIZES_ARE_CONST,
                                   ArrayOfArraysView<T, INDEX_TYPE const, true> const &>::type
  () const restrict_this
  { return reinterpret_cast<ArrayOfArraysView<T, INDEX_TYPE const, true> const &>(*this); }

  /**
   * @brief Method to make CONST_SIZES true. Use this method when the above UDC
   *        isn't invoked, this usually occurs with template argument deduction.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  ArrayOfArraysView<T, INDEX_TYPE const, true> const & toViewC() const restrict_this
  { return *this; }

  /**
   * @brief User defined conversion to move from T to T const.
   */
  template<class U=T>
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  operator typename std::enable_if<!std::is_const<U>::value,
                                   ArrayOfArraysView<T const, INDEX_TYPE const, true> const &>::type
  () const restrict_this
  { return reinterpret_cast<ArrayOfArraysView<T const, INDEX_TYPE const, true> const &>(*this); }

  /**
   * @brief Method to convert T to T const. Use this method when the above UDC
   *        isn't invoked, this usually occurs with template argument deduction.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  ArrayOfArraysView<T const, INDEX_TYPE const, true> const & toViewCC() const restrict_this
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
  { return m_numArrays; }

  /**
   * @brief Return the number of (zero length) arrays that can be stored before reallocation.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  INDEX_TYPE_NC capacity() const restrict_this
  {
    GEOS_ASSERT(m_sizes.capacity() < m_offsets.capacity());
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
  ArraySlice<T, 1, 0, INDEX_TYPE_NC> operator[]( INDEX_TYPE const i ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );
    return ArraySlice<T, 1, 0, INDEX_TYPE_NC>( m_values.data() + m_offsets[i], &m_sizes[i], nullptr );
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

  class IterableArray
  {
  public:
    LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
    IterableArray( T * const data, INDEX_TYPE const size ) :
      m_data( data ),
      m_size( size )
    {}

    LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
    T * begin() const restrict_this
    { return m_data; }

    LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
    T * end() const restrict_this
    { return m_data + m_size; }

  private:
    T * const m_data;
    INDEX_TYPE const m_size;
  };

  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  IterableArray getIterableArray( INDEX_TYPE const i ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );
    return IterableArray( m_values.data() + m_offsets[i], m_sizes[i] );
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
    m_sizes[i] += 1;
  }

  template< class POLICY >
  LVARRAY_HOST_DEVICE inline
  void atomicAppendToArray( POLICY, INDEX_TYPE const i, T const & value ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );

    T * const ptr = m_values.data() + m_offsets[ i ];
    INDEX_TYPE const previousSize = RAJA::atomicInc< POLICY >( &m_sizes[ i ] );
    ARRAYOFARRAYS_ATOMIC_CAPACITY_CHECK( i, previousSize, 1 );

    arrayManipulation::append( ptr, previousSize, value );
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
  void insertIntoArray( INDEX_TYPE const i, INDEX_TYPE const j, T const & value) const restrict_this
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
  void insertIntoArray( INDEX_TYPE const i, INDEX_TYPE const j, T && value) const restrict_this
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
  void insertIntoArray( INDEX_TYPE const i, INDEX_TYPE const j, T const * const values, INDEX_TYPE const n) const restrict_this
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

  friend std::ostream& operator<< ( std::ostream& stream, ArrayOfArraysView const & array )
  {
    stream << "{" << std::endl;

    for (INDEX_TYPE_NC i = 0; i < array.size(); ++i)
    {
      stream << i << "\t{";
      for (INDEX_TYPE_NC j = 0; j < array.sizeOfArray(i); ++j)
      {
        stream << array(i, j) << ", ";
      }

      // for (INDEX_TYPE_NC j = array.sizeOfArray(i); j < array.capacityOfArray(i); ++j)
      // {
      //   stream << "X" << ", ";
      // }

      stream << "}" << std::endl;
    }

    stream << "}" << std::endl;
    return stream;
  }

protected:

  /**
   * @brief Default constructor.
   */
  ArrayOfArraysView():
    m_numArrays( 0 ),
    m_offsets( true ),
    m_sizes( true ),
    m_values( true )
  {}

  /**
   * @brief Set the number of arrays.
   * @tparam BUFFERS variadic template where each type is a ChaiBuffer.
   * @param [in] newSize the new number of arrays.
   * @param [in] defaultArrayCapacity the default capacity for each new array.
   * @param [in/out] buffers variadic parameter pack where each argument is a ChaiBuffer that should be treated
   *        similarly to m_values. 
   * @note this is to be use by the non-view derived classes.
   */
  template <class ...BUFFERS>
  void resize(INDEX_TYPE const newSize, INDEX_TYPE const defaultArrayCapacity, BUFFERS & ...buffers)
  {
    GEOS_ASSERT( arrayManipulation::isPositive( newSize ) );

    INDEX_TYPE const offsetsSize = ( m_numArrays == 0 ) ? 0 : m_numArrays + 1;

    if (newSize < m_numArrays)
    {
      destroyValues(newSize, m_numArrays, buffers...);
      bufferManipulation::resize( m_offsets, offsetsSize, newSize + 1, 0 );
      bufferManipulation::resize( m_sizes, m_numArrays, newSize, 0 );
    }
    else
    {
      // The ternary here accounts for the case where m_offsets hasn't been allocated yet (when calling from a constructor).
      INDEX_TYPE const originalOffset = (m_numArrays == 0) ? 0 : m_offsets[m_numArrays];
      bufferManipulation::resize( m_offsets, offsetsSize, newSize + 1, originalOffset );
      bufferManipulation::resize( m_sizes, m_numArrays, newSize, 0 );

      if( defaultArrayCapacity > 0 )
      {
        for( INDEX_TYPE i = m_numArrays + 1; i < newSize + 1 ; ++i )
        {
          m_offsets[i] = originalOffset + i * defaultArrayCapacity;
        }

        INDEX_TYPE const totalSize = m_offsets[ newSize ];
        
        INDEX_TYPE const maxOffset = m_offsets[ m_numArrays ];
        for_each_arg(
          [totalSize, maxOffset](auto & buffer)
          {
            bufferManipulation::reserve( buffer, maxOffset, totalSize );
          },
          m_values, buffers...
        );
      }
    }

    m_numArrays = newSize;
  }

  /**
   * @brief Destroy all the objects held by this array and free all associated memory.
   * @tparam BUFFERS variadic template where each type is a ChaiBuffer.
   * @param [in/out] buffers variadic parameter pack where each argument is a ChaiBuffer that should be treated
   *        similarly to m_values.
   * @note this is to be use by the non-view derived classes.
   */
  template <class ...BUFFERS>
  void free(BUFFERS & ...buffers)
  {
    destroyValues(0, m_numArrays, buffers...);
    
    for_each_arg(
      [](auto & buffer)
      {
        buffer.free();
      },
      m_sizes, m_offsets, m_values, buffers...
    );

    m_numArrays = 0;
  }

  /**
   * @brief Set this ArrayOfArraysView equal to the provided arrays.
   * @tparam PAIRS_OF_BUFFERS variadic template where each type is an internal::PairOfBuffers.
   * @param [in] srcOffsets the source offsets array.
   * @param [in] srcSizes the source sizes array.
   * @param [in] srcValues the source values array.
   * @param [in/out] pairs variadic parameter pack where each argument is an internal::PairOfBuffers where each pair
   *                 should be treated similarly to {m_values, srcValues}.
   * @note this is to be use by the non-view derived classes.
   */
  template <class ...PAIRS_OF_BUFFERS>
  void setEqualTo( INDEX_TYPE const srcNumArrays,
                   INDEX_TYPE const srcMaxOffset,
                   ChaiBuffer<INDEX_TYPE> const & srcOffsets,
                   ChaiBuffer<INDEX_TYPE> const & srcSizes,
                   ChaiBuffer<T> const & srcValues,
                   PAIRS_OF_BUFFERS & ...pairs )
  {
    destroyValues(0, m_numArrays, pairs.first...);

    INDEX_TYPE const offsetsSize = ( m_numArrays == 0 ) ? 0 : m_numArrays + 1;

    bufferManipulation::copyInto( m_offsets, offsetsSize, srcOffsets, srcNumArrays + 1 );
    bufferManipulation::copyInto( m_sizes, m_numArrays, srcSizes, srcNumArrays );

    INDEX_TYPE const maxOffset = m_offsets[ m_numArrays ];
    for_each_arg(
      [maxOffset, srcMaxOffset](auto & dstBuffer)
      {
        bufferManipulation::reserve( dstBuffer, maxOffset, srcMaxOffset );
      },
      m_values, pairs.first...
    );

    m_numArrays = srcNumArrays;

    internal::PairOfBuffers<T> valuesPair(m_values, srcValues);

    for_each_arg(
      [this] ( auto & pair )
      {
        auto & dstBuffer = pair.first; 
        auto const & srcBuffer = pair.second;

        for (INDEX_TYPE_NC i = 0; i < m_numArrays; ++i)
        {
          INDEX_TYPE const offset = m_offsets[i];
          INDEX_TYPE const arraySize = sizeOfArray(i);
          arrayManipulation::uninitializedCopy( &dstBuffer[ offset ], arraySize, &srcBuffer[ offset ] );
        }
      },
      valuesPair, pairs...
    );
  }

  /**
   * @brief Move this ArrayOfArrays view to the given memory space.
   * @tparam BUFFERS variadic template where each type is a ChaiBuffer.
   * @param [in] space the memory space to move to.
   * @param [in/out] buffers variadic parameter pack where each argument is a ChaiBuffer that should be treated
   *        similarly to m_values.
   * @note this is to be use by the non-view derived classes.
   */
  template <class ...BUFFERS>
  void move(chai::ExecutionSpace const space, bool const touch, BUFFERS & ...buffers)
  {
    for_each_arg(
      [space, touch]( auto & buffer )
      {
        buffer.move( space, touch );
      },
      m_values, m_sizes, m_offsets, buffers...
    );
  }

  /**
   * @brief Touch this ArrayOfArrays view in the given memory space.
   * @tparam BUFFERS variadic template where each type is a ChaiBuffer.
   * @param [in] space the memory space to touch.
   * @param [in/out] buffers variadic parameter pack where each argument is a ChaiBuffer that should be treated
   *        similarly to m_values.
   * @note this is to be use by the non-view derived classes.
   */
  template <class ...BUFFERS>
  void registerTouch(chai::ExecutionSpace const space, BUFFERS & ...buffers)
  {
    for_each_arg(
      [space](auto & buffer)
      {
        buffer.registerTouch( space );
      },
      m_values, m_sizes, m_offsets, buffers...
    );
  }

  /**
   * @brief Compress the arrays so that the values of each array are contiguous with no extra capacity in between.
   * @tparam BUFFERS variadic template where each type is a ChaiBuffer.
   * @param [in/out] buffers variadic parameter pack where each argument is a ChaiBuffer that should be treated
   *        similarly to m_values.
   * @note this is to be use by the non-view derived classes.
   * @note This method doesn't free any memory. 
   */
  template <class ...BUFFERS>
  void compress(BUFFERS & ...buffers)
  {
    for (INDEX_TYPE i = 0; i < m_numArrays - 1; ++i)
    {
      INDEX_TYPE const nextOffset = m_offsets[ i + 1 ];
      INDEX_TYPE const shiftAmount = nextOffset - m_offsets[ i ] - sizeOfArray( i );
      INDEX_TYPE const sizeOfNextArray = sizeOfArray( i + 1 );

      // Shift the values in the next array down.
      for_each_arg(
        [sizeOfNextArray, nextOffset, shiftAmount] ( auto & buffer )
        {
          arrayManipulation::uninitializedShiftDown( &buffer[ nextOffset ], sizeOfNextArray, shiftAmount );
        },
        m_values, buffers...
      );

      // And update the offsets.
      m_offsets[ i + 1 ] -= shiftAmount;
    }

    // Update the last offset.
    m_offsets[ m_numArrays ] = m_offsets[ m_numArrays - 1 ] + sizeOfArray( m_numArrays - 1 );
  }

  /**
   * @brief Reserve space for the given number of arrays.
   * @param [in] newCapacity the new minimum capacity for the number of arrays.
   */
  void reserve( INDEX_TYPE const newCapacity )
  {
    bufferManipulation::reserve( m_offsets, m_numArrays + 1, newCapacity + 1 );
    bufferManipulation::reserve( m_sizes, m_numArrays, newCapacity );
  }

  /**
   * @brief Reserve space for the given number of values.
   * @tparam BUFFERS variadic template where each type is a ChaiBuffer.
   * @param [in] newValueCapacity the new minimum capacity for the number of values across all arrays.
   * @param [in/out] buffers variadic parameter pack where each argument is a ChaiBuffer that should be treated
   *        similarly to m_values.
   */
  template <class ...BUFFERS>
  void reserveValues( INDEX_TYPE const newValueCapacity, BUFFERS & ...buffers )
  {
    INDEX_TYPE const maxOffset = m_offsets[ m_numArrays ];
    for_each_arg(
      [newValueCapacity, maxOffset] (auto & buffer)
      {
        bufferManipulation::reserve( buffer, maxOffset, newValueCapacity );
      },
      m_values, buffers...
    );
  }

  /**
   * @brief Set the capacity of the given array.
   * @tparam BUFFERS variadic template where each type is a ChaiBuffer.
   * @param [in] i the array to set the capacity of.
   * @param [in] newCapacity the value to set the capacity of the given array to.
   * @param [in/out] buffers variadic parameter pack where each argument is a ChaiBuffer that should be treated
   *        similarly to m_values.
   * @note this is to be use by the non-view derived classes.
   */
  template <class ...BUFFERS>
  void setCapacityOfArray(INDEX_TYPE const i, INDEX_TYPE const newCapacity, BUFFERS & ...buffers)
  {
    ARRAYOFARRAYS_CHECK_BOUNDS(i);
    GEOS_ASSERT( arrayManipulation::isPositive( newCapacity ) );

    INDEX_TYPE const arrayCapacity = capacityOfArray(i);
    INDEX_TYPE const capacityIncrease = newCapacity - arrayCapacity;
    if( capacityIncrease == 0 ) return;

    if( capacityIncrease > 0 )
    {
      INDEX_TYPE const maxOffset = m_offsets[ m_numArrays ];
      for_each_arg(
        [this, i, maxOffset, capacityIncrease](auto & buffer)
        {
          // Increase the size of the buffer.
          bufferManipulation::dynamicReserve( buffer, maxOffset, maxOffset + capacityIncrease );

           // Shift up the values.
          for (INDEX_TYPE array = m_numArrays - 1; array > i; --array)
          {
            INDEX_TYPE const curArraySize = sizeOfArray( array );
            INDEX_TYPE const curArrayOffset = m_offsets[ array ];
            arrayManipulation::uninitializedShiftUp( &buffer[ curArrayOffset ], curArraySize, capacityIncrease );
          }
        },
        m_values, buffers...
      );
    }
    else
    {
      INDEX_TYPE const arrayOffset = m_offsets[i];
      INDEX_TYPE const capacityDecrease = -capacityIncrease;
      
      INDEX_TYPE const prevArraySize = sizeOfArray(i);
      INDEX_TYPE const newArraySize = min(prevArraySize, newCapacity);

      m_sizes[i] = newArraySize;
      for_each_arg(
        [this, i, capacityDecrease, arrayOffset, newArraySize, prevArraySize] ( auto & buffer )
        {
          // Delete the values at the end of the array.
          arrayManipulation::destroy( &buffer[ arrayOffset + newArraySize ], prevArraySize - newArraySize );

          // Shift down the values of subsequent arrays.
          for (INDEX_TYPE array = i + 1; array < m_numArrays; ++array)
          {
            INDEX_TYPE const curArraySize = sizeOfArray(array);
            INDEX_TYPE const curArrayOffset = m_offsets[array];
            arrayManipulation::uninitializedShiftDown( &buffer[ curArrayOffset ], curArraySize, capacityDecrease );
          }
        },
        m_values, buffers...
      );
    }

    // Update the offsets array
    for( INDEX_TYPE array = i + 1 ; array < m_numArrays + 1 ; ++array )
    {
      m_offsets[array] += capacityIncrease;
    }
  }

  template< typename U >
  void setUserCallBack(std::string const & name)
  {
    m_offsets.template setUserCallBack< U >( name + "/m_offsets" );
    m_sizes.template setUserCallBack< U >( name + "/m_sizes" );
    m_values.template setUserCallBack< U >( name + "/m_values" );
  }

  INDEX_TYPE m_numArrays = 0;

  // Holds the offset of each array, of length size() + 1. Array i begins at
  // m_offsets[i] and has capacity m_offsets[i+1] - m_offsets[i].
  ChaiBuffer<INDEX_TYPE> m_offsets;

  ChaiBuffer<SIZE_TYPE> m_sizes;

  // Holds the values of each array, of length numValues().
  ChaiBuffer<T> m_values;

private:

  /**
   * @brief Destroy the values in arrays in the range [begin, end).
   * @tparam BUFFERS variadic template where each type is a ChaiBuffer.
   * @param [in] begin the array to start with.
   * @param [in] end where to stop destroying values.
   * @param [in/out] buffers variadic parameter pack where each argument is a ChaiBuffer that should be treated
   *        similarly to m_values.
   * @note this is to be use by the non-view derived classes.
   * @note this method doesn't free any memory.
   */
  template <class ...BUFFERS>
  void destroyValues(INDEX_TYPE const begin, INDEX_TYPE const end, BUFFERS & ...buffers)
  {
    ARRAYOFARRAYS_CHECK_INSERT_BOUNDS(begin);
    ARRAYOFARRAYS_CHECK_INSERT_BOUNDS(end);
  
    for_each_arg(
      [this, begin, end] ( auto & buffer )
      {
        for( INDEX_TYPE i = begin ; i < end ; ++i )
        {
          INDEX_TYPE const offset = m_offsets[ i ];
          INDEX_TYPE const arraySize = sizeOfArray( i );
          arrayManipulation::destroy( &buffer[ offset ], arraySize );
        }
      },
      m_values, buffers...
    );
  }
};

} /* namespace LvArray */
