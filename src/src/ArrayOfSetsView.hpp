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
 */

#ifndef ARRAYOFSETSVIEW_HPP_
#define ARRAYOFSETSVIEW_HPP_

#include <limits>
#include "ArrayOfArraysView.hpp"
#include "arrayManipulation.hpp"
#include "sortedArrayManipulation.hpp"
#include "ArraySlice.hpp"
#include "templateHelpers.hpp"

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
template< typename T, typename INDEX_TYPE=std::ptrdiff_t >
class ArrayOfSetsView : protected ArrayOfArraysView< T, INDEX_TYPE, std::is_const<T>::value >
{
  // Alias for the parent class
  using ParentClass = ArrayOfArraysView<T, INDEX_TYPE, std::is_const<T>::value>;

public:

  using INDEX_TYPE_NC = typename std::remove_const<INDEX_TYPE>::type;

  // Aliasing public methods of ArrayOfArraysView.
  using ParentClass::size;

  /**
   * @brief Default copy constructor. Performs a shallow copy and calls the
   *        chai::ManagedArray copy constructor.
   * @param [in] src the ArrayOfArraysView to be copied.
   */
  inline
  ArrayOfSetsView( ArrayOfSetsView const & src ) = default;

  /**
   * @brief Default move constructor.
   * @param [in/out] src the ArrayOfArraysView to be moved from.
   */
  inline
  ArrayOfSetsView( ArrayOfSetsView && src ) = default;

  /**
   * @brief User defined conversion to move from T to T const.
   */
  template<class U=T>
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  operator typename std::enable_if<!std::is_const<U>::value,
                                   ArrayOfSetsView<T const, INDEX_TYPE const> const &>::type
  () const restrict_this
  { return reinterpret_cast<ArrayOfSetsView<T const, INDEX_TYPE const> const &>(*this); }

  /**
   * @brief Method to convert T to T const. Use this method when the above UDC
   *        isn't invoked, this usually occurs with template argument deduction.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  ArrayOfSetsView<T const, INDEX_TYPE const> const & toViewC() const restrict_this
  { return *this; }

  /**
   * @brief Method to convert to an immutable ArrayOfArraysView.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  ArrayOfArraysView<T const, INDEX_TYPE const, true> const & toArrayOfArraysView() const restrict_this
  { return *this; }

  /**
   * @brief Default copy assignment operator, this does a shallow copy.
   * @param [in] src the ArrayOfSetsView to be copied from.
   */
  inline
  ArrayOfSetsView & operator=( ArrayOfSetsView const & src ) = default;

  /**
   * @brief Default move assignment operator, this does a shallow copy.
   * @param [in/out] src the ArrayOfSetsView to be moved from.
   */
  inline
  ArrayOfSetsView & operator=( ArrayOfSetsView && src ) = default;


  /**
   * @brief Return an object provides an iterable interface to the given set.
   * @param [in] i the set to get an iterator for.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  typename ParentClass::IterableArray getIterableSet( INDEX_TYPE const i ) const restrict_this
  { return ParentClass::getIterableArray( i ); }

  /**
   * @brief Return the size of the given set.
   * @param [in] i the set to querry.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  INDEX_TYPE_NC sizeOfSet( INDEX_TYPE const i ) const restrict_this
  { return ParentClass::sizeOfArray( i ); }

  /**
   * @brief Return the capacity of the given set.
   * @param [in] i the set to querry.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  INDEX_TYPE_NC capacityOfSet( INDEX_TYPE const i ) const restrict_this
  { return ParentClass::capacityOfArray( i ); }

  /**
   * @brief Return an ArraySlice1d<T const> (pointer to const) to the values of the given array.
   * @param [in] i the array to access.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  ArraySlice<T const, 1, 0, INDEX_TYPE_NC> operator[]( INDEX_TYPE const i ) const restrict_this
  { return ParentClass::operator[]( i ); }

  /**
   * @brief Return a const reference to the value at the given position in the given array.
   * @param [in] i the array to access.
   * @param [in] j the index within the array to access.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  T const & operator()( INDEX_TYPE const i, INDEX_TYPE const j ) const restrict_this
  { return ParentClass::operator()( i, j ); }

  void consistencyCheck() const restrict_this
  {
    INDEX_TYPE const numSets = size();
    for ( INDEX_TYPE_NC i = 0; i < numSets; ++i )
    {
      GEOS_ERROR_IF_GT(sizeOfSet( i ), capacityOfSet( i ));

      T * const setValues = getSetValues( i );
      INDEX_TYPE const numValues = sizeOfSet( i );
      GEOS_ERROR_IF(!sortedArrayManipulation::isSorted(setValues, numValues), "Values should be sorted!");
      GEOS_ERROR_IF(!sortedArrayManipulation::allUnique(setValues, numValues), "Values should be unique!");
    }
  }

  /**
   * @brief Return true iff the given set contains the given value.
   * @param [in] i the set to search.
   * @param [in] value the value to search for.
   */
  LVARRAY_HOST_DEVICE inline
  bool contains( INDEX_TYPE const i, T const & value ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );

    INDEX_TYPE const setSize = sizeOfSet( i );
    T const * const setValues = (*this)[ i ];

    return sortedArrayManipulation::contains( setValues, setSize, value );
  }

  /**
   * @brief Insert a value into the given set.
   * @param [in] i the set to insert into.
   * @param [in] value the value to insert.
   * @return True iff the value was inserted (the set did not already contain the value).
   *
   * @note Since the ArrayOfSetsview can't do reallocation or shift the offsets it is
   *       up to the user to ensure that the given row has enough space for the new entries.
   *       If USE_ARRAY_BOUNDS_CHECK is defined a lack of space will result in an error,
   *       otherwise the values in the subsequent row will be overwritten.
   */
  LVARRAY_HOST_DEVICE inline
  bool insertIntoSet( INDEX_TYPE const i, T const & value ) const restrict_this
  { return insertIntoSetImpl( i, value, CallBacks( *this, i ) ); }

  /**
   * @brief Insert values into the given set.
   * @param [in] i the set to insert into.
   * @param [in] values the values to insert.
   * @param [in] n the number of values to insert.
   * @return The number of values inserted.
   *
   * @note If possible sort cols first by calling sortedArrayManipulation::makeSorted(values, values + n)
   *       and then call insertSortedIntoSet, this will be substantially faster.
   * @note Since the ArrayOfSetsview can't do reallocation or shift the offsets it is
   *       up to the user to ensure that the given row has enough space for the new entries.
   *       If USE_ARRAY_BOUNDS_CHECK is defined a lack of space will result in an error,
   *       otherwise the values in the subsequent row will be overwritten.
   */
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC insertIntoSet( INDEX_TYPE const i, T const * const values, INDEX_TYPE const n ) const restrict_this
  { return insertIntoSetImpl( i, values, n, CallBacks( *this, i ) ); }

  /**
   * @brief Insert values into the given set.
   * @param [in] i the set to insert into.
   * @param [in] values the values to insert. Must be sorted
   * @param [in] n the number of values to insert.
   * @return The number of values inserted.
   *
   * @note Since the ArrayOfSetsView can't do reallocation or shift the offsets it is
   *       up to the user to ensure that the given row has enough space for the new entries.
   *       If USE_ARRAY_BOUNDS_CHECK is defined a lack of space will result in an error,
   *       otherwise the values in the subsequent row will be overwritten.
   */
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC insertSortedIntoSet( INDEX_TYPE const i, T const * const values, INDEX_TYPE const n ) const restrict_this
  { return insertSortedIntoSetImpl( i, values, n, CallBacks( *this, i ) ); }

  /**
   * @brief Remove a value from the given set.
   * @param [in] i the set to remove from.
   * @param [in] value the value to remove.
   * @return True iff the value was removed (the set previously contained the value).
   */
  LVARRAY_HOST_DEVICE inline
  bool removeFromSet( INDEX_TYPE const i, T const & value ) const restrict_this
  { return removeFromSetImpl( i, value, CallBacks( *this, i ) ); }

  /**
   * @brief Remove values entries from the given set.
   * @param [in] i the set to remove from.
   * @param [in] values the values to remove.
   * @param [in] n the number of values to remove.
   * @return The number of values removed.
   *
   * @note If possible sort cols first by calling sortedArrayManipulation::makeSorted(values, values + n)
   *       and then call removeSortedFromSet, this will be substantially faster.
   */
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC removeFromSet( INDEX_TYPE const i, T const * const values, INDEX_TYPE const n ) const restrict_this
  { return removeFromSetImpl( i, values, n, CallBacks( *this, i ) ); }

/**
   * @brief Remove values entries from the given set.
   * @param [in] i the set to remove from.
   * @param [in] values the values to remove. Must be sorted.
   * @param [in] n the number of values to remove.
   * @return The number of values removed.
   */
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC removeSortedFromSet( INDEX_TYPE const i, T const * const values, INDEX_TYPE const n ) const restrict_this
  { return removeSortedFromSetImpl( i, values, n, CallBacks( *this, i ) ); }

protected:

  /**
   * @brief Default constructor. Made protected since every ArrayOfSetsView should
   *        either be the base of a ArrayOfSets or copied from another ArrayOfSetsView.
   */
  ArrayOfSetsView() = default;

  /**
   * @brief Return an ArraySlice1d (pointer) to the values of the given array.
   * @param [in] i the array to access.
   * @note This method is protected because it returns a non-const pointer.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  ArraySlice<T, 1, 0, INDEX_TYPE_NC> getSetValues( INDEX_TYPE const i ) const restrict_this
  { return ParentClass::operator[]( i ); }

  /**
   * @brief Helper function to insert a value into the given set.
   * @tparam CALLBACKS type of the call-back helper class.
   * @param [in] i the set to insert into.
   * @param [in] value the value to insert.
   * @param [in/out] cbacks call-back helper class used with the sortedArrayManipulation routines.
   * @return True iff the value was inserted (the set did not already contain the value).
   */
  template <class CALLBACKS>
  LVARRAY_HOST_DEVICE inline
  bool insertIntoSetImpl( INDEX_TYPE const i, T const & value, CALLBACKS && cbacks ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );

    INDEX_TYPE const setSize = sizeOfSet( i );
    T const * const setValues = (*this)[i];

    bool const success = sortedArrayManipulation::insert( setValues, setSize, value, std::move( cbacks ) );
    m_sizes[i] += success;
    return success;
  }

  /**
   * @brief Helper function to insert values into the given set.
   * @tparam CALLBACKS type of the call-back helper class.
   * @param [in] i the set to insert into.
   * @param [in] valuesToInsert the values to insert.
   * @param [in] n the number of values to insert.
   * @param [in/out] cbacks call-back helper class used with the sortedArrayManipulation routines.
   * @return The number of values inserted.
   */
  template <class CALLBACKS>
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC insertIntoSetImpl( INDEX_TYPE const i,
                                   T const * const valuesToInsert,
                                   INDEX_TYPE const n,
                                   CALLBACKS && cbacks ) const restrict_this
  {
    constexpr int LOCAL_SIZE = 16;
    T localValueBuffer[LOCAL_SIZE];

    T * const valueBuffer = sortedArrayManipulation::createTemporaryBuffer(valuesToInsert, n, localValueBuffer);
    sortedArrayManipulation::makeSorted(valueBuffer, valueBuffer + n);

    INDEX_TYPE const nInserted = insertSortedIntoSetImpl(i, valueBuffer, n, std::move( cbacks ) );

    sortedArrayManipulation::freeTemporaryBuffer(valueBuffer, n, localValueBuffer);
    return nInserted;
  }
  
  /**
   * @brief Helper function to insert values into the given set.
   * @tparam CALLBACKS type of the call-back helper class.
   * @param [in] i the set to insert into.
   * @param [in] valuesToInsert the values to insert. Must be sorted.
   * @param [in] n the number of values to insert.
   * @param [in/out] cbacks call-back helper class used with the sortedArrayManipulation routines.
   * @return The number of values inserted.
   */
  template <class CALLBACKS>
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC insertSortedIntoSetImpl( INDEX_TYPE const i,
                                         T const * const valuesToInsert,
                                         INDEX_TYPE const n,
                                         CALLBACKS && cbacks ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );
    GEOS_ASSERT( arrayManipulation::isPositive( n ) );
    GEOS_ASSERT( valuesToInsert != nullptr || n == 0 );
    GEOS_ASSERT( sortedArrayManipulation::isSorted( valuesToInsert, n ));

    INDEX_TYPE const setSize = sizeOfSet( i );
    T const * const setValues = (*this)[i];

    INDEX_TYPE const nInserted = sortedArrayManipulation::insertSorted( setValues, setSize, valuesToInsert, n, std::move( cbacks ) );
    m_sizes[i] += nInserted;
    return nInserted;
  }

  /**
   * @brief Helper function to remove a value from the given set.
   * @tparam CALLBACKS type of the call-back helper class.
   * @param [in] i the set to remove from.
   * @param [in] value the value to remove.
   * @param [in/out] cbacks call-back helper class used with the sortedArrayManipulation routines.
   * @return True iff the value was removed (the set contained the value).
   */
  template <class CALLBACKS>
  LVARRAY_HOST_DEVICE inline
  bool removeFromSetImpl( INDEX_TYPE const i, T const & value, CALLBACKS && cbacks ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );

    INDEX_TYPE const setSize = sizeOfSet( i );
    T * const setValues = getSetValues( i );

    bool const success = sortedArrayManipulation::remove( setValues, setSize, value, std::move( cbacks ) );
    m_sizes[i] -= success;
    return success;
  }

  /**
   * @brief Helper function to remove values from the given set.
   * @tparam CALLBACKS type of the call-back helper class.
   * @param [in] i the set to remove from.
   * @param [in] values the values to remove.
   * @param [in] n the number of values to remove.
   * @param [in/out] cbacks call-back helper class used with the sortedArrayManipulation routines.
   * @return The number of values removed.
   */
  template <class CALLBACKS>
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC removeFromSetImpl( INDEX_TYPE const i, T const * const valuesToRemove, INDEX_TYPE const n, CALLBACKS && cbacks ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );
    GEOS_ASSERT( arrayManipulation::isPositive( n ) );
    GEOS_ASSERT( valuesToRemove != nullptr || n == 0 );

    constexpr int LOCAL_SIZE = 16;
    T localValueBuffer[LOCAL_SIZE];

    T * const valueBuffer = sortedArrayManipulation::createTemporaryBuffer(valuesToRemove, n, localValueBuffer);
    sortedArrayManipulation::makeSorted(valueBuffer, valueBuffer + n);

    INDEX_TYPE const nRemoved = removeSortedFromSetImpl( i, valueBuffer, n, std::move( cbacks ) );

    sortedArrayManipulation::freeTemporaryBuffer(valueBuffer, n, localValueBuffer);
    return nRemoved;
  }
  
  /**
   * @brief Helper function to remove values from the given set.
   * @tparam CALLBACKS type of the call-back helper class.
   * @param [in] i the set to remove from.
   * @param [in] values the values to remove. Must be sorted.
   * @param [in] n the number of values to remove.
   * @param [in/out] cbacks call-back helper class used with the sortedArrayManipulation routines.
   * @return The number of values removed.
   */
  template <class CALLBACKS>
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC removeSortedFromSetImpl( INDEX_TYPE const i, T const * const valuesToRemove, INDEX_TYPE const n, CALLBACKS && cbacks ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );
    GEOS_ASSERT( arrayManipulation::isPositive( n ) );
    GEOS_ASSERT( valuesToRemove != nullptr || n == 0 );
    GEOS_ASSERT( sortedArrayManipulation::isSorted( valuesToRemove, n ));

    INDEX_TYPE const setSize = sizeOfSet( i );
    T * const setValues = getSetValues( i );

    INDEX_TYPE const nRemoved = sortedArrayManipulation::removeSorted( setValues, setSize, valuesToRemove, n, std::move( cbacks ) );
    m_sizes[i] -= nRemoved;
    return nRemoved;
  }

  // Aliasing protected members in ArrayOfArraysView
  using ParentClass::m_sizes;
  using ParentClass::m_values;

private:

  /**
   * @class CallBacks
   * @brief This class provides the callbacks for the sortedArrayManipulation routines.
   */
  class CallBacks : public sortedArrayManipulation::CallBacks<T, INDEX_TYPE_NC>
  {
  public:

    /**
     * @brief Constructor.
     * @param [in/out] aos the ArrayOfSetsView this CallBacks is associated with.
     * @param [in] i the set this CallBacks is associated with.
     */
    LVARRAY_HOST_DEVICE inline
    CallBacks( ArrayOfSetsView<T, INDEX_TYPE> const & aos, INDEX_TYPE const i ):
      m_aos( aos ),
      m_indexOfSet( i )
    {}

    /**
     * @brief Callback signaling that the size of the set has increased.
     * @param [in] nToAdd the increase in the size.
     * @note This method doesn't actually change the size, it just checks that the new
     *       size doesn't exceed the capacity since the ArrayOfSetsView can't do allocation.
     * @return a pointer to the sets values.
     */
    LVARRAY_HOST_DEVICE inline
    T * incrementSize( INDEX_TYPE const nToAdd ) const restrict_this
    {
#ifdef USE_ARRAY_BOUNDS_CHECK
      GEOS_ERROR_IF( m_aos.sizeOfSet( m_indexOfSet ) + nToAdd > m_aos.capacityOfSet( m_indexOfSet ),
                     "ArrayOfSetsView cannot do reallocation." );
#else
    CXX_UTILS_DEBUG_VAR( nToAdd );
#endif
      return m_aos.getSetValues( m_indexOfSet );
    }

  private:
    ArrayOfSetsView<T, INDEX_TYPE> const & m_aos;
    INDEX_TYPE const m_indexOfSet;
  };
};

} /* namespace LvArray */

#endif /* ARRAYOFSETSVIEW_HPP_ */
