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
 * @file ArrayOfSetsView.hpp
 * @brief Contains the implementation of LvArray::ArrayOfSetsView
 */

#pragma once

// Source includes
#include "ArrayOfArraysView.hpp"
#include "arrayManipulation.hpp"
#include "sortedArrayManipulation.hpp"
#include "ArraySlice.hpp"
#include "typeManipulation.hpp"

namespace LvArray
{

/**
 * @class ArrayOfSetsView
 * @brief This class provides a view into an array of sets like object.
 * @tparam T the type stored in the arrays.
 * @tparam INDEX_TYPE the integer to use for indexing.
 *
 * When INDEX_TYPE is const m_offsets is not touched when copied between memory spaces.
 * INDEX_TYPE should always be const since ArrayOfSetsview is not allowed to modify the offsets.
 *
 * When T is const and INDEX_TYPE is const you cannot insert or remove from the View
 * and neither the offsets, sizes, or values are touched when copied between memory spaces.
 */
template< typename T,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
class ArrayOfSetsView : protected ArrayOfArraysView< T, INDEX_TYPE, std::is_const< T >::value, BUFFER_TYPE >
{
protected:
  /// Alias for the parent class
  using ParentClass = ArrayOfArraysView< T, INDEX_TYPE, std::is_const< T >::value, BUFFER_TYPE >;

  /// Since INDEX_TYPE should always be const we need an alias for the non const version.
  using INDEX_TYPE_NC = typename ParentClass::INDEX_TYPE_NC;

public:
  using typename ParentClass::ValueType;
  using typename ParentClass::IndexType;
  using typename ParentClass::value_type;
  using typename ParentClass::size_type;


  /**
   * @name Constructors, destructor and assignment operators
   */
  ///@{

  /**
   * @brief Default copy constructor. Performs a shallow copy and calls the
   *   chai::ManagedArray copy constructor.
   */
  inline
  ArrayOfSetsView( ArrayOfSetsView const & ) = default;

  /**
   * @brief Default move constructor.
   */
  inline
  ArrayOfSetsView( ArrayOfSetsView && ) = default;

  /**
   * @brief Default copy assignment operator, this does a shallow copy.
   * @return *this.
   */
  inline
  ArrayOfSetsView & operator=( ArrayOfSetsView const & ) = default;

  /**
   * @brief Default move assignment operator, this does a shallow copy.
   * @return *this.
   */
  inline
  ArrayOfSetsView & operator=( ArrayOfSetsView && ) = default;

  ///@}

  /**
   * @name ArrayOfSetsView and ArrayOfArraysView creation methods
   */
  ///@{

  /**
   * @return Return a reference to *this reinterpreted as an
   *   ArrayOfSetsView< T, INDEX_TYPE const >.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  ArrayOfSetsView< T, INDEX_TYPE const, BUFFER_TYPE > const &
  toView() const LVARRAY_RESTRICT_THIS
  { return reinterpret_cast< ArrayOfSetsView< T, INDEX_TYPE const, BUFFER_TYPE > const & >(*this);}

  /**
   * @return Return a reference to *this reinterpreted as an
   *   ArrayOfSetsView< T const, INDEX_TYPE const >.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  ArrayOfSetsView< T const, INDEX_TYPE const, BUFFER_TYPE > const &
  toViewConst() const LVARRAY_RESTRICT_THIS
  { return reinterpret_cast< ArrayOfSetsView< T const, INDEX_TYPE const, BUFFER_TYPE > const & >(*this); }

  /**
   * @return Return a reference to *this reinterpreted as an
   *   ArrayOfArraysView< T const, INDEX_TYPE const, true >.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  ArrayOfArraysView< T const, INDEX_TYPE const, true, BUFFER_TYPE > const &
  toArrayOfArraysView() const LVARRAY_RESTRICT_THIS
  { return reinterpret_cast< ArrayOfArraysView< T const, INDEX_TYPE const, true, BUFFER_TYPE > const & >( *this ); }

  ///@}

  /**
   * @name Attribute querying methods
   */
  ///@{

  using ParentClass::size;

  /**
   * @return Return the size of the given set.
   * @param i The set to get the size of.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  INDEX_TYPE_NC sizeOfSet( INDEX_TYPE const i ) const LVARRAY_RESTRICT_THIS
  { return ParentClass::sizeOfArray( i ); }

  using ParentClass::capacity;

  /**
   * @return Return the capacity of the given set.
   * @param i The set to get the capacity of.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  INDEX_TYPE_NC capacityOfSet( INDEX_TYPE const i ) const LVARRAY_RESTRICT_THIS
  { return ParentClass::capacityOfArray( i ); }

  using ParentClass::valueCapacity;

  /**
   * @return Return true iff the given set contains the given value.
   * @param i the set to search.
   * @param value the value to search for.
   */
  LVARRAY_HOST_DEVICE inline
  bool contains( INDEX_TYPE const i, T const & value ) const LVARRAY_RESTRICT_THIS
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );

    INDEX_TYPE const setSize = sizeOfSet( i );
    T const * const setValues = (*this)[ i ];

    return sortedArrayManipulation::contains( setValues, setSize, value );
  }

  /**
   * @brief Verify that the capacity of each set is greater than or equal to the
   *   size and that each set is sorted unique.
   * @note The is intended for debugging.
   */
  void consistencyCheck() const LVARRAY_RESTRICT_THIS
  {
    INDEX_TYPE const numSets = size();
    for( INDEX_TYPE_NC i = 0; i < numSets; ++i )
    {
      LVARRAY_ERROR_IF_GT( sizeOfSet( i ), capacityOfSet( i ) );

      T * const setValues = getSetValues( i );
      INDEX_TYPE const numValues = sizeOfSet( i );
      LVARRAY_ERROR_IF( !sortedArrayManipulation::isSortedUnique( setValues, setValues + numValues ),
                        "Values should be sorted and unique!" );
    }
  }

  ///@}

  /**
   * @name Methods that provide access to the data
   */
  ///@{

  /**
   * @return Return an ArraySlice1d<T const> (pointer to const) to the values of the given array.
   * @param i The set to access.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  ArraySlice< T const, 1, 0, INDEX_TYPE_NC > operator[]( INDEX_TYPE const i ) const LVARRAY_RESTRICT_THIS
  { return ParentClass::operator[]( i ); }

  /**
   * @return Return a const reference to the value at the given position in the given array.
   * @param i The set to access.
   * @param j The index within the set to access.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  T const & operator()( INDEX_TYPE const i, INDEX_TYPE const j ) const LVARRAY_RESTRICT_THIS
  { return ParentClass::operator()( i, j ); }

  ///@}

  /**
   * @name Methods that modify the size of an inner set.
   */
  ///@{

  /**
   * @brief Insert a value into the given set.
   * @param i the set to insert into.
   * @param value the value to insert.
   * @return True iff the value was inserted (the set did not already contain the value).
   * @pre Since the ArrayOfSetsview can't do reallocation or shift the offsets it is
   *   up to the user to ensure that the given set has enough space for the new entries.
   */
  LVARRAY_HOST_DEVICE inline
  bool insertIntoSet( INDEX_TYPE const i, T const & value ) const LVARRAY_RESTRICT_THIS
  { return insertIntoSetImpl( i, value, CallBacks( *this, i ) ); }

  /**
   * @tparam ITER An iterator type.
   * @brief Inserts multiple values into the given set.
   * @param i The set to insert into.
   * @param first An iterator to the first value to insert.
   * @param last An iterator to the end of the values to insert.
   * @return The number of values inserted.
   * @pre The values to insert [ @p first, @p last ) must be sorted and contain no duplicates.
   * @pre Since the ArrayOfSetsView can't do reallocation or shift the offsets it is
   *   up to the user to ensure that the given set has enough space for the new entries.
   */
  template< typename ITER >
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC insertIntoSet( INDEX_TYPE const i, ITER const first, ITER const last ) const LVARRAY_RESTRICT_THIS
  { return insertIntoSetImpl( i, first, last, CallBacks( *this, i ) ); }

  /**
   * @brief Remove a value from the given set.
   * @param i The set to remove from.
   * @param value The value to remove.
   * @return True iff the value was removed (the set previously contained the value).
   */
  LVARRAY_HOST_DEVICE inline
  bool removeFromSet( INDEX_TYPE const i, T const & value ) const LVARRAY_RESTRICT_THIS
  { return removeFromSetImpl( i, value, CallBacks( *this, i ) ); }

  /**
   * @tparam ITER An iterator type.
   * @brief Removes multiple values from the given set.
   * @param i The set to remove from.
   * @param first An iterator to the first value to remove.
   * @param last An iterator to the end of the values to remove.
   * @return The number of values removed.
   * @pre The values to remove [ @p first, @p last ) must be sorted and contain no duplicates.
   */
  template< typename ITER >
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC removeFromSet( INDEX_TYPE const i, ITER const first, ITER const last ) const LVARRAY_RESTRICT_THIS
  { return removeFromSetImpl( i, first, last, CallBacks( *this, i ) ); }

  ///@}

  /**
   * @name Methods dealing with memory spaces
   */
  ///@{

  /**
   * @brief Move this ArrayOfSets to the given memory space.
   * @param space The memory space to move to.
   * @param touch If true touch the values, sizes and offsets in the new space.
   * @note When moving to the GPU since the offsets can't be modified on device they are not touched.
   * @note This is just a wrapper around the ArrayOfArraysView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  void move( MemorySpace const space, bool const touch=true ) const
  { return ParentClass::move( space, touch ); }

  ///@}

protected:

  /**
   * @brief Default constructor.
   * @note Protected since every ArrayOfSetsView should either be the base of a ArrayOfSets
   *   or copied from another ArrayOfSetsView.
   */
  ArrayOfSetsView() = default;

  /**
   * @return Return an ArraySlice1d to the values of the given array.
   * @param i the array to access.
   * @note Protected because it returns a non-const pointer.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  ArraySlice< T, 1, 0, INDEX_TYPE_NC > getSetValues( INDEX_TYPE const i ) const LVARRAY_RESTRICT_THIS
  { return ParentClass::operator[]( i ); }

  /**
   * @name Methods to be used by derived classes
   */
  ///@{

  /**
   * @brief Helper function to insert a value into the given set.
   * @tparam CALLBACKS type of the call-back helper class.
   * @param i the set to insert into.
   * @param value the value to insert.
   * @param cbacks call-back helper class used with the sortedArrayManipulation routines.
   * @return True iff the value was inserted (the set did not already contain the value).
   */
  template< typename CALLBACKS >
  LVARRAY_HOST_DEVICE inline
  bool insertIntoSetImpl( INDEX_TYPE const i, T const & value, CALLBACKS && cbacks ) const LVARRAY_RESTRICT_THIS
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );

    INDEX_TYPE const setSize = sizeOfSet( i );
    T * const setValues = getSetValues( i );

    bool const success = sortedArrayManipulation::insert( setValues, setSize, value, std::move( cbacks ) );
    this->m_sizes[i] += success;
    return success;
  }

  /**
   * @tparam ITER An iterator type.
   * @tparam CALLBACKS type of the call-back helper class.
   * @brief Inserts multiple values into the given set.
   * @param i The set to insert into.
   * @param first An iterator to the first value to insert.
   * @param last An iterator to the end of the values to insert.
   * @param cbacks Helper class used with the sortedArrayManipulation routines.
   * @return The number of values inserted.
   * @note The values to insert [ @p first, @p last ) must be sorted and contain no duplicates.
   */
  template< typename ITER, typename CALLBACKS >
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC insertIntoSetImpl( INDEX_TYPE const i,
                                   ITER const first,
                                   ITER const last,
                                   CALLBACKS && cbacks ) const LVARRAY_RESTRICT_THIS
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );

    INDEX_TYPE const setSize = sizeOfSet( i );
    T * const setValues = getSetValues( i );

    INDEX_TYPE const nInserted = sortedArrayManipulation::insert( setValues,
                                                                  setSize,
                                                                  first,
                                                                  last,
                                                                  std::forward< CALLBACKS >( cbacks ) );
    this->m_sizes[i] += nInserted;
    return nInserted;
  }

  /**
   * @brief Helper function to remove a value from the given set.
   * @tparam CALLBACKS type of the call-back helper class.
   * @param i the set to remove from.
   * @param value the value to remove.
   * @param cbacks call-back helper class used with the sortedArrayManipulation routines.
   * @return True iff the value was removed (the set contained the value).
   */
  template< typename CALLBACKS >
  LVARRAY_HOST_DEVICE inline
  bool removeFromSetImpl( INDEX_TYPE const i, T const & value, CALLBACKS && cbacks ) const LVARRAY_RESTRICT_THIS
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );

    INDEX_TYPE const setSize = sizeOfSet( i );
    T * const setValues = getSetValues( i );

    bool const success = sortedArrayManipulation::remove( setValues, setSize, value, std::move( cbacks ) );
    this->m_sizes[i] -= success;
    return success;
  }

  /**
   * @tparam ITER An iterator type.
   * @tparam CALLBACKS type of the call-back helper class.
   * @brief Removes multiple values from the given set.
   * @param i The set to remove from.
   * @param first An iterator to the first value to remove.
   * @param last An iterator to the end of the values to remove.
   * @param cbacks Helper class used with the sortedArrayManipulation routines.
   * @return The number of values removed.
   * @note The values to remove [ @p first, @p last ) must be sorted and contain no duplicates.
   */
  template< typename ITER, typename CALLBACKS >
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC removeFromSetImpl( INDEX_TYPE const i,
                                   ITER const first,
                                   ITER const last,
                                   CALLBACKS && cbacks ) const LVARRAY_RESTRICT_THIS
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );

    INDEX_TYPE const setSize = sizeOfSet( i );
    T * const setValues = getSetValues( i );

    INDEX_TYPE const nRemoved = sortedArrayManipulation::remove( setValues,
                                                                 setSize,
                                                                 first,
                                                                 last,
                                                                 std::forward< CALLBACKS >( cbacks ) );
    this->m_sizes[i] -= nRemoved;
    return nRemoved;
  }

  ///@}

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
     * @param aos the ArrayOfSetsView this CallBacks is associated with.
     * @param i the set this CallBacks is associated with.
     */
    LVARRAY_HOST_DEVICE inline
    CallBacks( ArrayOfSetsView const & aos, INDEX_TYPE const i ):
      m_aos( aos ),
      m_indexOfSet( i )
    {}

    /**
     * @brief Callback signaling that the size of the set has increased.
     * @param curPtr the current pointer to the array.
     * @param nToAdd the increase in the size.
     * @note This method doesn't actually change the size, it just checks that the new
     *       size doesn't exceed the capacity since the ArrayOfSetsView can't do allocation.
     * @return a pointer to the sets values.
     */
    LVARRAY_HOST_DEVICE inline
    T * incrementSize( T * const curPtr, INDEX_TYPE const nToAdd ) const LVARRAY_RESTRICT_THIS
    {
      LVARRAY_UNUSED_VARIABLE( curPtr );
#ifdef USE_ARRAY_BOUNDS_CHECK
      LVARRAY_ERROR_IF_GT_MSG( m_aos.sizeOfSet( m_indexOfSet ) + nToAdd, m_aos.capacityOfSet( m_indexOfSet ),
                               "ArrayOfSetsView cannot do reallocation." );
#else
      LVARRAY_DEBUG_VAR( nToAdd );
#endif
      return m_aos.getSetValues( m_indexOfSet );
    }

private:
    /// The ArrayOfSetsView the call back is associated with.
    ArrayOfSetsView const & m_aos;

    /// The index of the set the call back is associated with.
    INDEX_TYPE const m_indexOfSet;
  };
};

} // namespace LvArray
