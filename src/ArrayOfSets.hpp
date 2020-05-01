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
 * @file SparsityPattern.hpp
 */

#pragma once

#include "ArrayOfSetsView.hpp"

namespace LvArray
{

// Forward declaration of the ArrayOfArrays class so that we can define the stealFrom method.
template< class T, typename INDEX_TYPE >
class ArrayOfArrays;


/**
 * @class ArrayOfSets
 * @brief This class implements an array of sets like object with contiguous storage.
 * @tparam T the type stored in the sets.
 * @tparam INDEX_TYPE the integer to use for indexing.
 */
template< class T, class INDEX_TYPE=std::ptrdiff_t >
class ArrayOfSets : protected ArrayOfSetsView< T, INDEX_TYPE >
{

  /// An alias for the parent class.
  using ParentClass = ArrayOfSetsView< T, INDEX_TYPE >;

public:

  // Aliasing public methods of ArrayOfSetsView.
  using ParentClass::capacity;
  using ParentClass::valueCapacity;
  using ParentClass::sizeOfSet;
  using ParentClass::capacityOfSet;
  using ParentClass::operator();
  using ParentClass::operator[];
  using ParentClass::getSetValues;

  using ParentClass::getIterableSet;
  using ParentClass::removeFromSet;
  using ParentClass::contains;
  using ParentClass::consistencyCheck;

  /**
   * @brief Constructor.
   * @param nsets the number of sets.
   * @param defaultSetCapacity the initial capacity of each set.
   */
  inline
  ArrayOfSets( INDEX_TYPE const nsets=0, INDEX_TYPE defaultSetCapacity=0 ) LVARRAY_RESTRICT_THIS:
    ArrayOfSetsView< T, INDEX_TYPE >()
  {
    resize( nsets, defaultSetCapacity );
    setName( "" );
  }

  /**
   * @brief Copy constructor, performs a deep copy.
   * @param src the ArrayOfSets to copy.
   */
  inline
  ArrayOfSets( ArrayOfSets const & src ) LVARRAY_RESTRICT_THIS:
    ArrayOfSetsView< T, INDEX_TYPE >()
  { *this = src; }

  /**
   * @brief Default move constructor, performs a shallow copy.
   * @return *this.
   */
  inline
  ArrayOfSets( ArrayOfSets && ) = default;

  /**
   * @brief Destructor, frees the values, sizes and offsets Buffers.
   */
  inline
  ~ArrayOfSets() LVARRAY_RESTRICT_THIS
  { ParentClass::free(); }

  /**
   * @brief Copy assignment operator, performs a deep copy.
   * @param src the ArrayOfSets to copy.
   * @return *this.
   */
  inline
  ArrayOfSets & operator=( ArrayOfSets const & src ) LVARRAY_RESTRICT_THIS
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
   * @return *this.
   */
  inline
  ArrayOfSets & operator=( ArrayOfSets && ) = default;

  /**
   * @brief @return A reference to *this reinterpreted as an ArrayOfSetsView< T, INDEX_TYPE const >.
   */
  inline
  ArrayOfSetsView< T, INDEX_TYPE const > const & toView() const LVARRAY_RESTRICT_THIS
  { return reinterpret_cast< ArrayOfSetsView< T, INDEX_TYPE const > const & >(*this); }

  /**
   * @brief @return A reference to *this reinterpreted as an ArrayOfSetsView< T const, INDEX_TYPE const >.
   * @note Duplicated for SFINAE needs.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  ArrayOfSetsView< T const, INDEX_TYPE const > const & toViewConst() const LVARRAY_RESTRICT_THIS
  { return ParentClass::toViewConst(); }

  /**
   * @brief @return A reference to *this reinterpreted as an ArrayOfArraysView< T const, INDEX_TYPE const, true >.
   * @note Duplicated for SFINAE needs.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  ArrayOfArraysView< T const, INDEX_TYPE const, true > const & toArrayOfArraysView() const LVARRAY_RESTRICT_THIS
  { return ParentClass::toArrayOfArraysView(); }

  /**
   * @brief @return Return the number sets.
   * @note Duplicated for SFINAE needs.
   */
  inline
  INDEX_TYPE size() const LVARRAY_RESTRICT_THIS
  { return ParentClass::size(); }

  /**
   * @brief Steal the resources from an ArrayOfArrays and convert it to an ArrayOfSets.
   * @param src the ArrayOfArrays to convert.
   * @param desc describes the type of data in the source.
   * @note This would be prime for omp parallelism.
   */
  inline
  void stealFrom( ArrayOfArrays< T, INDEX_TYPE > && src, sortedArrayManipulation::Description const desc ) LVARRAY_RESTRICT_THIS
  {
    ParentClass::free();
    ParentClass::stealFrom( reinterpret_cast< ArrayOfArraysView< T, INDEX_TYPE > && >( src ) );

    INDEX_TYPE const numSets = size();
    if( desc == sortedArrayManipulation::UNSORTED_NO_DUPLICATES )
    {
      for( INDEX_TYPE i = 0; i < numSets; ++i )
      {
        T * const setValues = getSetValues( i );
        INDEX_TYPE const numValues = sizeOfSet( i );
        std::sort( setValues, setValues + numValues );
      }
    }
    if( desc == sortedArrayManipulation::SORTED_WITH_DUPLICATES )
    {
      for( INDEX_TYPE i = 0; i < numSets; ++i )
      {
        T * const setValues = getSetValues( i );
        INDEX_TYPE const numValues = sizeOfSet( i );

        INDEX_TYPE const numUniqueValues = sortedArrayManipulation::removeDuplicates( setValues, setValues + numValues );
        arrayManipulation::resize( setValues, numValues, numUniqueValues );
        m_sizes[ i ] = numUniqueValues;
      }
    }
    if( desc == sortedArrayManipulation::UNSORTED_WITH_DUPLICATES )
    {
      for( INDEX_TYPE i = 0; i < numSets; ++i )
      {
        T * const setValues = getSetValues( i );
        INDEX_TYPE const numValues = sizeOfSet( i );

        INDEX_TYPE const numUniqueValues = sortedArrayManipulation::makeSortedUnique( setValues, setValues + numValues );
        arrayManipulation::resize( setValues, numValues, numUniqueValues );
        m_sizes[ i ] = numUniqueValues;
      }
    }

#ifdef ARRAY_BOUNDS_CHECK
    consistencyCheck();
#endif
  }

  /**
   * @brief Clear a set.
   * @param i the index of the set to clear.
   */
  void clearSet( INDEX_TYPE const i ) LVARRAY_RESTRICT_THIS
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );

    INDEX_TYPE const prevSize = sizeOfSet( i );
    T * const values = getSetValues( i );
    arrayManipulation::resize( values, prevSize, INDEX_TYPE( 0 ) );
    m_sizes[ i ] = 0;
  }

  /**
   * @brief Reserve space for the given number of sets.
   * @param newCapacity the new minimum capacity for the number of sets.
   */
  inline
  void reserve( INDEX_TYPE const newCapacity ) LVARRAY_RESTRICT_THIS
  { ParentClass::reserve( newCapacity ); }

  /**
   * @brief Reserve space for the given number of values.
   * @param newValueCapacity the new minimum capacity for the number of values across all sets.
   */
  inline
  void reserveValues( INDEX_TYPE const newValueCapacity ) LVARRAY_RESTRICT_THIS
  { ParentClass::reserveValues( newValueCapacity ); }

  /**
   * @brief Set the capacity of a set.
   * @param i the set to set the capacity of.
   * @param newCapacity the value to set the capacity of the set to.
   */
  inline
  void setCapacityOfSet( INDEX_TYPE const i, INDEX_TYPE newCapacity ) LVARRAY_RESTRICT_THIS
  { ParentClass::setCapacityOfArray( i, newCapacity ); }

  /**
   * @brief Reserve space in a set.
   * @param i the set to reserve space in.
   * @param newCapacity the number of values to reserve space for.
   */
  inline
  void reserveCapacityOfSet( INDEX_TYPE const i, INDEX_TYPE newCapacity ) LVARRAY_RESTRICT_THIS
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );
    if( newCapacity > capacityOfSet( i ) ) setCapacityOfSet( i, newCapacity );
  }

  /**
   * @brief Compress the arrays so that the values of each array are contiguous with no extra capacity in between.
   * @note This method doesn't free any memory.
   */
  inline
  void compress() LVARRAY_RESTRICT_THIS
  { ParentClass::compress(); }

  /**
   * @brief Set the number of sets.
   * @param numSubSets The new number of sets.
   * @param defaultSetCapacity The default capacity for each new array.
   */
  void resize( INDEX_TYPE const numSubSets, INDEX_TYPE const defaultSetCapacity=0 ) LVARRAY_RESTRICT_THIS
  { ParentClass::resize( numSubSets, defaultSetCapacity ); }

  /**
   * @brief Append an set with the given capacity.
   * @param n the capacity of the set.
   */
  inline
  void appendSet( INDEX_TYPE const n=0 ) LVARRAY_RESTRICT_THIS
  {
    INDEX_TYPE const maxOffset = m_offsets[ m_numArrays ];
    bufferManipulation::pushBack( m_offsets, m_numArrays + 1, maxOffset );
    bufferManipulation::pushBack( m_sizes, m_numArrays, 0 );
    ++m_numArrays;

    setCapacityOfSet( m_numArrays - 1, n );
  }

  /**
   * @brief Insert a set with the given capacity.
   * @param i the position to insert the set.
   * @param n the capacity of the set.
   */
  inline
  void insertSet( INDEX_TYPE const i, INDEX_TYPE const n=0 ) LVARRAY_RESTRICT_THIS
  {
    ARRAYOFARRAYS_CHECK_INSERT_BOUNDS( i );
    LVARRAY_ASSERT( arrayManipulation::isPositive( n ) );

    // Insert an set of capacity zero at the given location
    INDEX_TYPE const offset = m_offsets[i];
    bufferManipulation::insert( m_offsets, m_numArrays + 1, i + 1, offset );
    bufferManipulation::insert( m_sizes, m_numArrays, i, 0 );
    ++m_numArrays;

    // Set the capacity of the new set
    setCapacityOfSet( i, n );
  }

  /**
   * @brief Erase a set.
   * @param i the position of the set to erase.
   */
  inline
  void eraseSet( INDEX_TYPE const i ) LVARRAY_RESTRICT_THIS
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );

    setCapacityOfSet( i, 0 );
    bufferManipulation::erase( m_offsets, m_numArrays + 1, i + 1 );
    bufferManipulation::erase( m_sizes, m_numArrays, i );
    --m_numArrays;
  }

  /**
   * @brief Insert a value into the given set.
   * @param i the set to insert into.
   * @param val the value to insert.
   * @return True iff the value was inserted (the set did not already contain the value).
   */
  inline
  bool insertIntoSet( INDEX_TYPE const i, T const & val ) LVARRAY_RESTRICT_THIS
  { return ParentClass::insertIntoSetImpl( i, val, CallBacks( *this, i ) ); }

  /**
   * @tparam ITER An iterator type.
   * @brief Inserts multiple values into the given set.
   * @param i The set to insert into.
   * @param first An iterator to the first value to insert.
   * @param last An iterator to the end of the values to insert.
   * @return The number of values inserted.
   * @pre The values to insert [first, last) must be sorted and contain no duplicates.
   */
  template< typename ITER >
  inline
  INDEX_TYPE insertIntoSet( INDEX_TYPE const i, ITER const first, ITER const last ) LVARRAY_RESTRICT_THIS
  { return ParentClass::insertIntoSetImpl( i, first, last, CallBacks( *this, i ) ); }

  /**
   * @brief Set the name to be displayed whenever the underlying Buffer's user call back is called.
   * @param name the name to display.
   */
  void setName( std::string const & name )
  { ArrayOfArraysView< T, INDEX_TYPE >::template setName< decltype( *this ) >( name ); }

  /**
   * @brief Move this ArrayOfSetsView to the given memory space and touch the values, sizes and offsets.
   * @param space the memory space to move to.
   * @param touch If true touch the values, sizes and offsets in the new space.
   * @note When moving to the GPU since the offsets can't be modified on device they are not touched.
   * @note Duplicated for SFINAE needs.
   */
  void move( chai::ExecutionSpace const space, bool const touch=true ) const
  { return ParentClass::move( space, touch ); }

private:

  /**
   * @class CallBacks
   * @brief This class provides the callbacks for the sortedArrayManipulation routines.
   */
  class CallBacks : public sortedArrayManipulation::CallBacks< T >
  {
public:

    /**
     * @brief Constructor.
     * @param aos the ArrayOfSets this CallBacks is associated with.
     * @param i the set this CallBacks is associated with.
     */
    inline
    CallBacks( ArrayOfSets< T, INDEX_TYPE > & sp, INDEX_TYPE const i ):
      m_aos( sp ),
      m_i( i )
    {}

    /**
     * @brief Callback signaling that the size of the set has increased.
     * @param curPtr the current pointer to the array.
     * @param nToAdd the increase in the size.
     * @note This method doesn't actually change the size, it just checks if the new size
     *       exceeds the capacity of the set and if so reserves more space.
     * @return a pointer to the sets values.
     */
    inline
    T * incrementSize( T * const LVARRAY_UNUSED_ARG( curPtr ),
                       INDEX_TYPE const nToAdd ) const LVARRAY_RESTRICT_THIS
    {
      INDEX_TYPE const newNNZ = m_aos.sizeOfSet( m_i ) + nToAdd;
      if( newNNZ > m_aos.capacityOfSet( m_i ) )
      {
        m_aos.setCapacityOfSet( m_i, 2 * newNNZ );
      }

      return m_aos.getSetValues( m_i );
    }

private:
    /// A reference to the associated ArrayOfSets.
    ArrayOfSets< T, INDEX_TYPE > & m_aos;

    /// The index of the associated set.
    INDEX_TYPE const m_i;
  };

  // Aliasing protected members of ArrayOfSetsView.
  using ParentClass::m_numArrays;
  using ParentClass::m_offsets;
  using ParentClass::m_sizes;
  using ParentClass::m_values;
};

} /* namespace LvArray */
