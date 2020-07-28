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
 * @file ArrayOfSets.hpp
 * @brief Contains the implementation of LvArray::ArrayOfSets
 */

#pragma once

#include "ArrayOfSetsView.hpp"

namespace LvArray
{

// Forward declaration of the ArrayOfArrays class so that we can define the assimilate method.
template< typename T, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
class ArrayOfArrays;


/**
 * @class ArrayOfSets
 * @brief This class implements an array of sets like object with contiguous storage.
 * @tparam T the type stored in the sets.
 * @tparam INDEX_TYPE the integer to use for indexing.
 */
template< typename T,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
class ArrayOfSets : protected ArrayOfSetsView< T, INDEX_TYPE, BUFFER_TYPE >
{

  /// An alias for the parent class.
  using ParentClass = ArrayOfSetsView< T, INDEX_TYPE, BUFFER_TYPE >;

public:
  using typename ParentClass::ValueType;
  using typename ParentClass::IndexType;
  using typename ParentClass::value_type;
  using typename ParentClass::size_type;

  /**
   * @name Constructors and the destructor.
   */
  ///@{

  /**
   * @brief Constructor.
   * @param nsets the number of sets.
   * @param defaultSetCapacity the initial capacity of each set.
   */
  inline
  ArrayOfSets( INDEX_TYPE const nsets=0, INDEX_TYPE defaultSetCapacity=0 ) LVARRAY_RESTRICT_THIS:
    ParentClass()
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
    ParentClass()
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

  ///@}

  /**
   * @name Methods to construct the array from scratch.
   */
  ///@{

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
   * @param src The ArrayOfSets to be moved from.
   * @return *this.
   */
  inline
  ArrayOfSets & operator=( ArrayOfSets && src )
  {
    ParentClass::free();
    ParentClass::operator=( std::move( src ) );
    return *this;
  }

  /**
   * @brief Steal the resources from an ArrayOfArrays and convert it to an ArrayOfSets.
   * @param src the ArrayOfArrays to convert.
   * @param desc describes the type of data in the source.
   * TODO: Add a RAJA policy template parameter.
   */
  inline
  void assimilate( ArrayOfArrays< T, INDEX_TYPE, BUFFER_TYPE > && src,
                   sortedArrayManipulation::Description const desc ) LVARRAY_RESTRICT_THIS
  {
    ParentClass::free();
    ParentClass::assimilate( reinterpret_cast< ParentClass && >( src ) );

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
        this->m_sizes[ i ] = numUniqueValues;
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
        this->m_sizes[ i ] = numUniqueValues;
      }
    }

#ifdef ARRAY_BOUNDS_CHECK
    consistencyCheck();
#endif
  }

  using ParentClass::resizeFromCapacities;

  ///@}

  /**
   * @name ArrayOfArraysView creation methods
   */
  ///@{

  /**
   * @copydoc ParentClass::toView
   * @note This is just a wrapper around the ArrayOfSetsView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  constexpr inline
  ArrayOfSetsView< T, INDEX_TYPE const, BUFFER_TYPE > const &
  toView() const LVARRAY_RESTRICT_THIS
  { return ParentClass::toView(); }

  /**
   * @copydoc ParentClass::toViewConst
   * @note This is just a wrapper around the ArrayOfSetsView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  constexpr inline
  ArrayOfSetsView< T const, INDEX_TYPE const, BUFFER_TYPE > const &
  toViewConst() const LVARRAY_RESTRICT_THIS
  { return ParentClass::toViewConst(); }

  /**
   * @copydoc ParentClass::toArrayOfArraysView
   * @note This is just a wrapper around the ArrayOfSetsView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  constexpr inline
  ArrayOfArraysView< T const, INDEX_TYPE const, true, BUFFER_TYPE > const &
  toArrayOfArraysView() const LVARRAY_RESTRICT_THIS
  { return ParentClass::toArrayOfArraysView(); }

  ///@}

  /**
   * @name Attribute querying methods
   */
  ///@{

  /**
   * @copydoc ParentClass::size
   * @note This is just a wrapper around the ArrayOfSetsView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  inline
  INDEX_TYPE size() const LVARRAY_RESTRICT_THIS
  { return ParentClass::size(); }

  using ParentClass::sizeOfSet;
  using ParentClass::capacity;
  using ParentClass::capacityOfSet;
  using ParentClass::valueCapacity;
  using ParentClass::contains;
  using ParentClass::consistencyCheck;

  ///@}


  /**
   * @name Methods that provide access to the data
   */
  ///@{

  using ParentClass::operator[];
  using ParentClass::operator();

  ///@}

  /**
   * @name Methods to modify the outer array.
   */
  ///@{

  using ParentClass::reserve;
  using ParentClass::reserveValues;

  /**
   * @copydoc ArrayOfArraysView::resize
   * @note This is just a wrapper around the ArrayOfArraysView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  void resize( INDEX_TYPE const newSize, INDEX_TYPE const defaultArrayCapacity=0 )
  { return ParentClass::resize( newSize, defaultArrayCapacity ); }

  using ParentClass::compress;

  ///@}

  /**
   * @name Methods to create or remove an inner set.
   */
  ///@{

  /**
   * @brief Append a set with capacity @p setCapacity.
   * @param setCapacity The capacity of the set.
   */
  inline
  void appendSet( INDEX_TYPE const setCapacity=0 ) LVARRAY_RESTRICT_THIS
  {
    INDEX_TYPE const maxOffset = this->m_offsets[ this->m_numArrays ];
    bufferManipulation::emplaceBack( this->m_offsets, this->m_numArrays + 1, maxOffset );
    bufferManipulation::emplaceBack( this->m_sizes, this->m_numArrays, 0 );
    ++this->m_numArrays;

    setCapacityOfSet( this->m_numArrays - 1, setCapacity );
  }

  /**
   * @brief Insert a set at position @p i with capacity @p setCapacity.
   * @param i The position to insert the set.
   * @param setCapacity The capacity of the set.
   */
  inline
  void insertSet( INDEX_TYPE const i, INDEX_TYPE const setCapacity=0 ) LVARRAY_RESTRICT_THIS
  {
    ARRAYOFARRAYS_CHECK_INSERT_BOUNDS( i );
    LVARRAY_ASSERT( arrayManipulation::isPositive( setCapacity ) );

    // Insert an set of capacity zero at the given location
    INDEX_TYPE const offset = this->m_offsets[i];
    bufferManipulation::emplace( this->m_offsets, this->m_numArrays + 1, i + 1, offset );
    bufferManipulation::emplace( this->m_sizes, this->m_numArrays, i, 0 );
    ++this->m_numArrays;

    // Set the capacity of the new set
    setCapacityOfSet( i, setCapacity );
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
    bufferManipulation::erase( this->m_offsets, this->m_numArrays + 1, i + 1 );
    bufferManipulation::erase( this->m_sizes, this->m_numArrays, i );
    --this->m_numArrays;
  }

  ///@}

  /**
   * @name Methods to modify an inner set.
   */
  ///@{

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
   * @copydoc ParentClass::removeFromSet
   * @note This is not brought in with a @c using statement because it breaks doxygen.
   */
  LVARRAY_HOST_DEVICE inline
  bool removeFromSet( INDEX_TYPE const i, T const & value ) const LVARRAY_RESTRICT_THIS
  { return ParentClass::removeFromSet( i, value ); }

  /**
   * @tparam ITER An iterator type.
   * @brief Removes multiple values from the given set.
   * @param i The set to remove from.
   * @param first An iterator to the first value to remove.
   * @param last An iterator to the end of the values to remove.
   * @return The number of values removed.
   * @pre The values to remove [ @p first, @p last ) must be sorted and contain no duplicates.
   * @note This is not brought in with a @c using statement because it breaks doxygen. Furthermore I'm not
   *   sure how to get copydoc to work with an overloaded template like this.
   */
  template< typename ITER >
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE removeFromSet( INDEX_TYPE const i, ITER const first, ITER const last ) const LVARRAY_RESTRICT_THIS
  { return ParentClass::removeFromSet( i, first, last ); }

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
    this->m_sizes[ i ] = 0;
  }

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

  ///@}

  /**
   * @name Methods dealing with memory spaces
   */
  ///@{

  /**
   * @copydoc ArrayOfSetsView::move
   * @note This is just a wrapper around the ArrayOfSetsView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  void move( MemorySpace const space, bool const touch=true ) const
  { return ParentClass::move( space, touch ); }

  ///@}

  /**
   * @brief Set the name to be displayed whenever the underlying Buffer's user call back is called.
   * @param name the name to display.
   */
  void setName( std::string const & name )
  { ParentClass::template setName< decltype( *this ) >( name ); }

private:

  using ParentClass::getSetValues;

  /**
   * @class CallBacks
   * @brief This class provides the callbacks for the sortedArrayManipulation routines.
   */
  class CallBacks : public sortedArrayManipulation::CallBacks< T >
  {
public:

    /**
     * @brief Constructor.
     * @param aos The ArrayOfSets this CallBacks is associated with.
     * @param i The set this CallBacks is associated with.
     */
    inline
    CallBacks( ArrayOfSets & aos, INDEX_TYPE const i ):
      m_aos( aos ),
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
    T * incrementSize( T * const curPtr, INDEX_TYPE const nToAdd ) const LVARRAY_RESTRICT_THIS
    {
      LVARRAY_UNUSED_VARIABLE( curPtr );
      INDEX_TYPE const newNNZ = m_aos.sizeOfSet( m_i ) + nToAdd;
      if( newNNZ > m_aos.capacityOfSet( m_i ) )
      {
        m_aos.setCapacityOfSet( m_i, 2 * newNNZ );
      }

      return m_aos.getSetValues( m_i );
    }

private:
    /// A reference to the associated ArrayOfSets.
    ArrayOfSets & m_aos;

    /// The index of the associated set.
    INDEX_TYPE const m_i;
  };
};

} /* namespace LvArray */
