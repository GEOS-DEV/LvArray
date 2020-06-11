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
 * @file ArrayOfArrays.hpp
 */

#pragma once

#include "ArrayOfArraysView.hpp"

namespace LvArray
{

// Forward declaration of the ArrayOfSets class so that we can define the stealFrom method.
template< class T, class INDEX_TYPE >
class ArrayOfSets;

/**
 * @class ArrayOfArrays
 * @brief This class implements an array of arrays like object with contiguous storage.
 * @tparam T the type stored in the arrays.
 * @tparam INDEX_TYPE the integer to use for indexing.
 */
template< class T, typename INDEX_TYPE=std::ptrdiff_t >
class ArrayOfArrays : protected ArrayOfArraysView< T, INDEX_TYPE >
{
public:

  /// An alias for the parent class ArrayOfArraysView< T, INDEX_TYPE >.
  using ParentClass = ArrayOfArraysView< T, INDEX_TYPE >;

  // Aliasing public methods of ArrayOfArraysView.
  using ParentClass::sizeOfArray;
  using ParentClass::capacity;
  using ParentClass::valueCapacity;
  using ParentClass::capacityOfArray;
  using ParentClass::operator[];
  using ParentClass::operator();
  using ParentClass::getIterableArray;
  using ParentClass::atomicAppendToArray;
  using ParentClass::eraseFromArray;

  /**
   * @brief Constructor.
   * @param numArrays the number of arrays.
   * @param defaultArrayCapacity the initial capacity of each array.
   */
  inline
  ArrayOfArrays( INDEX_TYPE const numArrays=0, INDEX_TYPE const defaultArrayCapacity=0 ) LVARRAY_RESTRICT_THIS:
    ParentClass()
  {
    resize( numArrays, defaultArrayCapacity );
    setName( "" );
  }

  /**
   * @brief Copy constructor, performs a deep copy.
   * @param src the ArrayOfArrays to copy.
   */
  inline
  ArrayOfArrays( ArrayOfArrays const & src ):
    ParentClass()
  { *this = src; }

  /**
   * @brief Default move constructor, performs a shallow copy.
   */
  inline
  ArrayOfArrays( ArrayOfArrays && ) = default;

  /**
   * @brief Destructor, frees the values, sizes and offsets buffers.
   */
  ~ArrayOfArrays()
  { ParentClass::free(); }

  /**
   * @brief Copy assignment operator, performs a deep copy.
   * @param src the ArrayOfArrays to copy.
   * @return *this.
   */
  inline
  ArrayOfArrays & operator=( ArrayOfArrays const & src ) LVARRAY_RESTRICT_THIS
  {
    ParentClass::setEqualTo( src.m_numArrays,
                             src.m_offsets[ src.m_numArrays ],
                             src.m_offsets,
                             src.m_sizes,
                             src.m_values );
    return *this;
  }

  /**
   * @brief Default move assignment operator, performs a shallow copy.
   * @param src The ArrayOfArrays to be moved from.
   * @return *this
   */
  inline
  ArrayOfArrays & operator=( ArrayOfArrays && src )
  {
    ParentClass::free();
    ParentClass::operator=( std::move( src ) );
    return *this;
  }

  /**
   * @brief Steal the resources from an ArrayOfSets and convert it to an ArrayOfArrays.
   * @param src the ArrayOfSets to convert.
   */
  inline
  void stealFrom( ArrayOfSets< T, INDEX_TYPE > && src )
  {
    ParentClass::free();
    ParentClass::stealFrom( reinterpret_cast< ParentClass && >( src ) );
  }

  /**
   * @brief @return Return a reference to *this converted to ArrayOfArraysView<T, INDEX_TYPE const>.
   * @note Duplicated for SFINAE needs.
   */
  constexpr inline
  ArrayOfArraysView< T, INDEX_TYPE const > const & toView() const LVARRAY_RESTRICT_THIS
  { return reinterpret_cast< ArrayOfArraysView< T, INDEX_TYPE const > const & >( *this ); }

  /**
   * @brief @return Return a reference to *this converted to ArrayOfArraysView<T, INDEX_TYPE const, true>.
   * @note Duplicated for SFINAE needs.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  ArrayOfArraysView< T, INDEX_TYPE const, true > const & toViewConstSizes() const LVARRAY_RESTRICT_THIS
  { return ParentClass::toViewConstSizes(); }

  /**
   * @brief @return Return a reference to *this converted to ArrayOfArraysView<T const, INDEX_TYPE const, true>.
   * @note Duplicated for SFINAE needs.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  ArrayOfArraysView< T const, INDEX_TYPE const, true > const & toViewConst() const LVARRAY_RESTRICT_THIS
  { return ParentClass::toViewConst(); }

  /**
   * @brief @return Return the number of arrays.
   * @note Duplicated for SFINAE needs.
   */
  inline
  INDEX_TYPE size() const LVARRAY_RESTRICT_THIS
  { return ParentClass::size(); }

  /**
   * @brief Reserve space for the given number of arrays.
   * @param newCapacity the new minimum capacity for the number of arrays.
   */
  void reserve( INDEX_TYPE const newCapacity )
  { ParentClass::reserve( newCapacity ); }

  /**
   * @brief Reserve space for the given number of values.
   * @param newValueCapacity the new minimum capacity for the number of values across all arrays.
   */
  void reserveValues( INDEX_TYPE const newValueCapacity )
  { ParentClass::reserveValues( newValueCapacity ); }

  /**
   * @brief Set the number of arrays.
   * @param numSubArrays The new number of arrays.
   * @param defaultArrayCapacity The default capacity for each new array.
   */
  void resize( INDEX_TYPE const numSubArrays, INDEX_TYPE const defaultArrayCapacity=0 )
  { ParentClass::resize( numSubArrays, defaultArrayCapacity ); }

  /**
   * @tparam POLICY The RAJA policy used to convert @p capacities into the offsets array.
   *   Should NOT be a device policy.
   * @brief Clears the array and creates a new array with the given number of sub-arrays.
   * @param numSubArrays The new number of arrays.
   * @param capacities A pointer to an array of length @p numSubArrays containing the capacity
   *   of each new sub array.
   */
  template< typename POLICY >
  void resizeFromCapacities( INDEX_TYPE const numSubArrays, INDEX_TYPE const * const capacities )
  { ParentClass::template resizeFromCapacities< POLICY >( numSubArrays, capacities ); }

  /**
   * @brief Append an array.
   * @param n the size of the array.
   */
  void appendArray( INDEX_TYPE const n ) LVARRAY_RESTRICT_THIS
  {
    LVARRAY_ASSERT( arrayManipulation::isPositive( n ) );

    INDEX_TYPE const maxOffset = m_offsets[ m_numArrays ];
    bufferManipulation::pushBack( m_offsets, m_numArrays + 1, maxOffset );
    bufferManipulation::pushBack( m_sizes, m_numArrays, 0 );
    ++m_numArrays;

    resizeArray( m_numArrays - 1, n );
  }

  /**
   * @brief Append an array.
   * @param values the values of the array to append.
   * @param n the number of values.
   */
  void appendArray( T const * const values, INDEX_TYPE const n ) LVARRAY_RESTRICT_THIS
  {
    INDEX_TYPE const maxOffset = m_offsets[ m_numArrays ];
    bufferManipulation::pushBack( m_offsets, m_numArrays + 1, maxOffset );
    bufferManipulation::pushBack( m_sizes, m_numArrays, 0 );
    ++m_numArrays;

    appendToArray( m_numArrays - 1, values, n );
  }

  /**
   * @brief Insert an array.
   * @param i the position to insert the array.
   * @param values the values of the array to insert.
   * @param n the number of values.
   */
  void insertArray( INDEX_TYPE const i, T const * const values, INDEX_TYPE const n )
  {
    ARRAYOFARRAYS_CHECK_INSERT_BOUNDS( i );
    LVARRAY_ASSERT( arrayManipulation::isPositive( n ) );
    LVARRAY_ASSERT( n == 0 || values != nullptr );

    // Insert an array of capacity zero at the given location
    INDEX_TYPE const offset = m_offsets[i];
    bufferManipulation::insert( m_offsets, m_numArrays + 1, i + 1, offset );
    bufferManipulation::insert( m_sizes, m_numArrays, i, 0 );
    ++m_numArrays;

    // Append to the new array
    appendToArray( i, values, n );
  }

  /**
   * @brief Erase an array.
   * @param i the position of the array to erase.
   */
  void eraseArray( INDEX_TYPE const i )
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );

    setCapacityOfArray( i, 0 );
    bufferManipulation::erase( m_offsets, m_numArrays + 1, i + 1 );
    bufferManipulation::erase( m_sizes, m_numArrays, i );
    --m_numArrays;
  }

  /**
   * @brief Compress the arrays so that the values of each array are contiguous with no extra capacity in between.
   * @note This method doesn't free any memory.
   */
  void compress()
  { ParentClass::compress(); }

  /**
   * @brief Append a value to an array.
   * @param i the array to append to.
   * @param value the value to append.
   */
  void appendToArray( INDEX_TYPE const i, T const & value ) LVARRAY_RESTRICT_THIS
  {
    dynamicallyGrowArray( i, 1 );
    ParentClass::appendToArray( i, value );
  }

  /**
   * @brief Append a value to an array.
   * @param i the array to append to.
   * @param value the value to append.
   */
  void appendToArray( INDEX_TYPE const i, T && value ) LVARRAY_RESTRICT_THIS
  {
    dynamicallyGrowArray( i, 1 );
    ParentClass::appendToArray( i, std::move( value ) );
  }

  /**
   * @brief Append values to an array.
   * @param i the array to append to.
   * @param values the values to append.
   * @param n the number of values to append.
   */
  void appendToArray( INDEX_TYPE const i, T const * const values, INDEX_TYPE const n ) LVARRAY_RESTRICT_THIS
  {
    dynamicallyGrowArray( i, n );
    ParentClass::appendToArray( i, values, n );
  }

  /**
   * @brief Insert a value into an array.
   * @param i the array to insert into.
   * @param j the position at which to insert.
   * @param value the value to insert.
   */
  void insertIntoArray( INDEX_TYPE const i, INDEX_TYPE const j, T const & value )
  {
    dynamicallyGrowArray( i, 1 );
    ParentClass::insertIntoArray( i, j, value );
  }

  /**
   * @brief Insert a value into an array.
   * @param i the array to insert into.
   * @param j the position at which to insert.
   * @param value the value to insert.
   */
  void insertIntoArray( INDEX_TYPE const i, INDEX_TYPE const j, T && value )
  {
    dynamicallyGrowArray( i, 1 );
    ParentClass::insertIntoArray( i, j, std::move( value ) );
  }

  /**
   * @brief Insert values into an array.
   * @param i the array to insert into.
   * @param j the position at which to insert.
   * @param values the values to insert.
   * @param n the number of values to insert.
   */
  void insertIntoArray( INDEX_TYPE const i, INDEX_TYPE const j, T const * const values, INDEX_TYPE const n )
  {
    dynamicallyGrowArray( i, n );
    ParentClass::insertIntoArray( i, j, values, n );
  }

  /**
   * @brief Set the number of values in an array.
   * @tparam ARGS variadic template parameter of the types used to initialize any new values with.
   * @param i the array to resize.
   * @param newSize the value to set the size of the array to.
   * @param args variadic parameter pack of the arguments used to initialize any new values with.
   */
  template< class ... ARGS >
  void resizeArray( INDEX_TYPE const i, INDEX_TYPE const newSize, ARGS && ... args )
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );
    LVARRAY_ASSERT( arrayManipulation::isPositive( newSize ) );

    if( newSize > capacityOfArray( i ) )
    {
      setCapacityOfArray( i, newSize );
    }

    INDEX_TYPE const prevSize = sizeOfArray( i );
    T * const values = (*this)[i];
    arrayManipulation::resize( values, prevSize, newSize, std::forward< ARGS >( args )... );
    m_sizes[ i ] = newSize;
  }

  /**
   * @brief Clear the given array.
   * @param i The index of the array to clear.
   */
  void clearArray( INDEX_TYPE const i )
  { resizeArray( i, 0 ); }

  /**
   * @brief Set the capacity of an array.
   * @param i the array to set the capacity of.
   * @param newCapacity the value to set the capacity of the array to.
   */
  void setCapacityOfArray( INDEX_TYPE const i, INDEX_TYPE const newCapacity )
  { ParentClass::setCapacityOfArray( i, newCapacity ); }

  /**
   * @brief Set the name to be displayed whenever the underlying Buffer's user call back is called.
   * @param name The name to display.
   */
  void setName( std::string const & name )
  { ParentClass::template setName< decltype( *this ) >( name ); }

  /**
   * @brief Move this ArrayOfArrays to the given memory space.
   * @param space The memory space to move to.
   * @param touch If true touch the values, sizes and offsets in the new space.
   * @note When moving to the GPU since the offsets can't be modified on device they are not touched.
   * @note Duplicated for SFINAE needs.
   */
  void move( chai::ExecutionSpace const space, bool touch=true ) const
  { ParentClass::move( space, touch ); }

private:

  /**
   * @brief Dynamically grow the capacity of an array.
   * @param i the array to grow.
   * @param increase the increase in the size of the array.
   */
  void dynamicallyGrowArray( INDEX_TYPE const i, INDEX_TYPE const increase )
  {
    LVARRAY_ASSERT( arrayManipulation::isPositive( increase ) );

    INDEX_TYPE const newArraySize = sizeOfArray( i ) + increase;
    if( newArraySize > capacityOfArray( i ))
    {
      setCapacityOfArray( i, 2 * newArraySize );
    }
  }

  using ParentClass::m_numArrays;
  using ParentClass::m_offsets;
  using ParentClass::m_sizes;
  using ParentClass::m_values;
};

} /* namespace LvArray */
