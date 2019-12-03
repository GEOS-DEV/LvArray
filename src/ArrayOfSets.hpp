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
public:

  // Aliasing public methods of ArrayOfSetsView.
  using ArrayOfSetsView< T, INDEX_TYPE >::toViewC;
  using ArrayOfSetsView< T, INDEX_TYPE >::toArrayOfArraysView;
  using ArrayOfSetsView< T, INDEX_TYPE >::capacity;
  using ArrayOfSetsView< T, INDEX_TYPE >::sizeOfSet;
  using ArrayOfSetsView< T, INDEX_TYPE >::capacityOfSet;
  using ArrayOfSetsView< T, INDEX_TYPE >::operator();
  using ArrayOfSetsView< T, INDEX_TYPE >::operator[];
  using ArrayOfSetsView< T, INDEX_TYPE >::getSetValues;

  using ArrayOfSetsView< T, INDEX_TYPE >::getIterableSet;
  using ArrayOfSetsView< T, INDEX_TYPE >::removeFromSet;
  using ArrayOfSetsView< T, INDEX_TYPE >::removeSortedFromSet;
  using ArrayOfSetsView< T, INDEX_TYPE >::contains;
  using ArrayOfSetsView< T, INDEX_TYPE >::consistencyCheck;

  /**
   * @brief Return the number sets.
   * @note This needs is duplicated here for the intel compiler on cori.
   */
  inline
  INDEX_TYPE size() const restrict_this
  { return ArrayOfSetsView< T, INDEX_TYPE >::size(); }

  /**
   * @brief Constructor.
   * @param [in] nsets the number of sets.
   * @param [in] defaultSetCapacity the initial capacity of each set.
   */
  inline
  ArrayOfSets( INDEX_TYPE const nsets=0, INDEX_TYPE defaultSetCapacity=0 ) restrict_this:
    ArrayOfSetsView< T, INDEX_TYPE >()
  {
    resize( nsets, defaultSetCapacity );
    setName( "" );
  }

  /**
   * @brief Copy constructor, performs a deep copy.
   * @param [in] src the ArrayOfSets to copy.
   */
  inline
  ArrayOfSets( ArrayOfSets const & src ) restrict_this:
    ArrayOfSetsView< T, INDEX_TYPE >()
  { *this = src; }

  /**
   * @brief Default move constructor, performs a shallow copy.
   * @param [in/out] src the ArrayOfSets to be moved from.
   */
  inline
  ArrayOfSets( ArrayOfSets && src ) = default;

  /**
   * @brief Destructor, frees the values, sizes and offsets Buffers.
   */
  inline
  ~ArrayOfSets() restrict_this
  { ArrayOfSetsView< T, INDEX_TYPE >::free(); }

  /**
   * @brief Conversion operator to an ArrayOfArraysView.
   */
  CONSTEXPRFUNC inline
  operator ArrayOfSetsView< T, INDEX_TYPE const > const &
  () const restrict_this
  { return reinterpret_cast< ArrayOfSetsView< T, INDEX_TYPE const > const & >(*this); }

  /**
   * @brief Conversion operator to an ArrayOfArraysView.
   */
  inline
  ArrayOfSetsView< T, INDEX_TYPE const > const & toView() const restrict_this
  { return *this; }

  /**
   * @brief Method to convert to an immutable ArrayOfSetsView.
   */
  CONSTEXPRFUNC inline
  operator ArrayOfSetsView< T const, INDEX_TYPE const > const &
  () const restrict_this
  { return toViewC(); }

  /**
   * @brief Conversion operator to an immutable ArrayOfArraysView.
   */
  template< class U=T >
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  operator ArrayOfArraysView< T const, INDEX_TYPE const, true >
  () const restrict_this
  { return toArrayOfArraysView(); }

  /**
   * @brief Copy assignment operator, performs a deep copy.
   * @param [in] src the ArrayOfSets to copy.
   */
  inline
  ArrayOfSets & operator=( ArrayOfSets const & src ) restrict_this
  {
    ArrayOfSetsView< T, INDEX_TYPE >::setEqualTo( src.m_numArrays,
                                                  src.m_offsets[ src.m_numArrays ],
                                                  src.m_offsets,
                                                  src.m_sizes,
                                                  src.m_values );
    return *this;
  }

  /**
   * @brief Default move assignment operator, performs a shallow copy.
   * @param [in] src the ArrayOfSets to be moved from.
   */
  inline
  ArrayOfSets & operator=( ArrayOfSets && src ) = default;

  /**
   * @brief Steal the resources from an ArrayOfArrays and convert it to an ArrayOfSets.
   * @param [in/out] src the ArrayOfArrays to convert.
   * @param [in] desc describes the type of data in the source.
   * @note This would be prime for omp parallelism.
   */
  inline
  void stealFrom( ArrayOfArrays< T, INDEX_TYPE > && src, sortedArrayManipulation::Description const desc ) restrict_this
  {
    ArrayOfSetsView< T, INDEX_TYPE >::free();

    // Reinterpret cast to ArrayOfArraysView so that we don't have to include ArrayOfArrays.hpp.
    ArrayOfArraysView< T, INDEX_TYPE > && srcView = reinterpret_cast< ArrayOfArraysView< T, INDEX_TYPE > && >( src );

    m_numArrays = srcView.m_numArrays;
    srcView.m_numArrays = 0;

    m_offsets = std::move( srcView.m_offsets );
    m_sizes = std::move( srcView.m_sizes );
    m_values = std::move( srcView.m_values );

    INDEX_TYPE const numSets = size();
    if( desc == sortedArrayManipulation::UNSORTED_NO_DUPLICATES )
    {
      for( INDEX_TYPE i = 0 ; i < numSets ; ++i )
      {
        T * const setValues = getSetValues( i );
        INDEX_TYPE const numValues = sizeOfSet( i );
        std::sort( setValues, setValues + numValues );
      }
    }
    if( desc == sortedArrayManipulation::SORTED_WITH_DUPLICATES )
    {
      for( INDEX_TYPE i = 0 ; i < numSets ; ++i )
      {
        T * const setValues = getSetValues( i );
        INDEX_TYPE const numValues = sizeOfSet( i );

        INDEX_TYPE const numUniqueValues = sortedArrayManipulation::removeDuplicates( setValues, numValues );
        arrayManipulation::resize( setValues, numValues, numUniqueValues );
        m_sizes[ i ] = numUniqueValues;
      }
    }
    if( desc == sortedArrayManipulation::UNSORTED_WITH_DUPLICATES )
    {
      for( INDEX_TYPE i = 0 ; i < numSets ; ++i )
      {
        T * const setValues = getSetValues( i );
        INDEX_TYPE const numValues = sizeOfSet( i );
        std::sort( setValues, setValues + numValues );

        INDEX_TYPE const numUniqueValues = sortedArrayManipulation::removeDuplicates( setValues, numValues );
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
   * @param[in] i the index of the set to clear.
   */
  void clearSet( INDEX_TYPE const i ) restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );

    INDEX_TYPE const prevSize = sizeOfSet( i );
    T * const values = getSetValues( i );
    arrayManipulation::resize( values, prevSize, INDEX_TYPE( 0 ) );
    m_sizes[ i ] = 0;
  }

  /**
   * @brief Move to the given memory space, optionally touching it.
   * @param [in] space the memory space to move to.
   * @param [in] touch whether to touch the memory in the space or not.
   */
  inline
  void move( chai::ExecutionSpace const space, bool const touch=true ) restrict_this
  { ArrayOfSetsView< T, INDEX_TYPE >::move( space, touch ); }

  /**
   * @brief Touch in the given memory space.
   * @param [in] space the memory space to touch.
   */
  inline
  void registerTouch( chai::ExecutionSpace const space ) restrict_this
  { ArrayOfSetsView< T, INDEX_TYPE >::registerTouch( space ); }

  /**
   * @brief Reserve space for the given number of sets.
   * @param [in] newCapacity the new minimum capacity for the number of sets.
   */
  inline
  void reserve( INDEX_TYPE const newCapacity ) restrict_this
  { ArrayOfSetsView< T, INDEX_TYPE >::reserve( newCapacity ); }

  /**
   * @brief Reserve space for the given number of values.
   * @param [in] newValueCapacity the new minimum capacity for the number of values across all sets.
   */
  inline
  void reserveValues( INDEX_TYPE const nnz ) restrict_this
  { m_values.reserve( nnz ); }

  /**
   * @brief Set the capacity of a set.
   * @param [in] i the set to set the capacity of.
   * @param [in] newCapacity the value to set the capacity of the set to.
   */
  inline
  void setCapacityOfSet( INDEX_TYPE const i, INDEX_TYPE newCapacity ) restrict_this
  { ArrayOfSetsView< T, INDEX_TYPE >::setCapacityOfArray( i, newCapacity ); }

  /**
   * @brief Reserve space in a set.
   * @param [in] i the set to reserve space in.
   * @param [in] newCapacity the number of values to reserve space for.
   */
  inline
  void reserveCapacityOfSet( INDEX_TYPE const i, INDEX_TYPE newCapacity ) restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );
    if( newCapacity > capacityOfSet( i ) ) setCapacityOfSet( i, newCapacity );
  }

  /**
   * @brief Compress the arrays so that the values of each array are contiguous with no extra capacity in between.
   * @note This method doesn't free any memory.
   */
  inline
  void compress() restrict_this
  { ArrayOfSetsView< T, INDEX_TYPE >::compress(); }

  /**
   * @brief Set the number of sets.
   * @param [in] numSets the new number of sets.
   * @note We need this method in addition to the following resize method because of SFINAE requirements.
   */
  void resize( INDEX_TYPE const numSets ) restrict_this
  { ArrayOfSetsView< T, INDEX_TYPE >::resize( numSets, 0 ); }

  /**
   * @brief Set the number of sets.
   * @param [in] newSize the new number of sets.
   * @param [in] defaultSetCapacity the default capacity for each new array.
   */
  void resize( INDEX_TYPE const numSets, INDEX_TYPE const defaultSetCapacity ) restrict_this
  { ArrayOfSetsView< T, INDEX_TYPE >::resize( numSets, defaultSetCapacity ); }

  /**
   * @brief Append an set with the given capacity.
   * @param [in] n the capacity of the set.
   */
  inline
  void appendSet( INDEX_TYPE const n=0 ) restrict_this
  {
    INDEX_TYPE const maxOffset = m_offsets[ m_numArrays ];
    bufferManipulation::pushBack( m_offsets, m_numArrays + 1, maxOffset );
    bufferManipulation::pushBack( m_sizes, m_numArrays, 0 );
    ++m_numArrays;

    setCapacityOfSet( m_numArrays - 1, n );
  }

  /**
   * @brief Insert a set with the given capacity.
   * @param [in] i the position to insert the set.
   * @param [in] n the capacity of the set.
   */
  inline
  void insertSet( INDEX_TYPE const i, INDEX_TYPE const n=0 ) restrict_this
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
   * @param [in] i the position of the set to erase.
   */
  inline
  void eraseSet( INDEX_TYPE const i ) restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );

    setCapacityOfSet( i, 0 );
    bufferManipulation::erase( m_offsets, m_numArrays + 1, i + 1 );
    bufferManipulation::erase( m_sizes, m_numArrays, i );
    --m_numArrays;
  }

  /**
   * @brief Insert a value into the given set.
   * @param [in] i the set to insert into.
   * @param [in] val the value to insert.
   * @return True iff the value was inserted (the set did not already contain the value).
   */
  inline
  bool insertIntoSet( INDEX_TYPE const i, T const & val ) restrict_this
  { return ArrayOfSetsView< T, INDEX_TYPE >::insertIntoSetImpl( i, val, CallBacks( *this, i ) ); }

  /**
   * @brief Insert values into the given set.
   * @param [in] i the set to insert into.
   * @param [in] vals the values to insert.
   * @param [in] n the number of values to insert.
   * @return The number of values inserted.
   */
  inline
  INDEX_TYPE insertIntoSet( INDEX_TYPE const i, T const * const vals, INDEX_TYPE const n ) restrict_this
  { return ArrayOfSetsView< T, INDEX_TYPE >::insertIntoSetImpl( i, vals, n, CallBacks( *this, i ) ); }

  /**
   * @brief Insert values into the given set.
   * @param [in] i the set to insert into.
   * @param [in] vals the values to insert. Must be sorted
   * @param [in] n the number of values to insert.
   * @return The number of values inserted.
   */
  inline
  INDEX_TYPE insertSortedIntoSet( INDEX_TYPE const i, T const * const vals, INDEX_TYPE const n ) restrict_this
  { return ArrayOfSetsView< T, INDEX_TYPE >::insertSortedIntoSetImpl( i, vals, n, CallBacks( *this, i ) ); }

  void setName( std::string const & name )
  {
    ArrayOfArraysView< T, INDEX_TYPE >::template setName< decltype( *this ) >( name );
  }

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
     * @param [in/out] aos the ArrayOfSets this CallBacks is associated with.
     * @param [in] i the set this CallBacks is associated with.
     */
    inline
    CallBacks( ArrayOfSets< T, INDEX_TYPE > & sp, INDEX_TYPE const i ):
      m_aos( sp ),
      m_i( i )
    {}

    /**
     * @brief Callback signaling that the size of the set has increased.
     * @param [in] nToAdd the increase in the size.
     * @note This method doesn't actually change the size, it just checks if the new size
     *       exceeds the capacity of the set and if so reserves more space.
     * @return a pointer to the sets values.
     */
    inline
    T * incrementSize( INDEX_TYPE const nToAdd ) const restrict_this
    {
      INDEX_TYPE const newNNZ = m_aos.sizeOfSet( m_i ) + nToAdd;
      if( newNNZ > m_aos.capacityOfSet( m_i ) )
      {
        m_aos.setCapacityOfSet( m_i, 2 * newNNZ );
      }

      return m_aos.getSetValues( m_i );
    }

private:
    ArrayOfSets< T, INDEX_TYPE > & m_aos;
    INDEX_TYPE const m_i;
  };

  // Aliasing protected members of ArrayOfSetsView.
  using ArrayOfSetsView< T, INDEX_TYPE >::m_numArrays;
  using ArrayOfSetsView< T, INDEX_TYPE >::m_offsets;
  using ArrayOfSetsView< T, INDEX_TYPE >::m_sizes;
  using ArrayOfSetsView< T, INDEX_TYPE >::m_values;
};

} /* namespace LvArray */
