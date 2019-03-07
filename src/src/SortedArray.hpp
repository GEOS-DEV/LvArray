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
 * @file SortedArray.hpp
 */

#ifndef SRC_COMMON_SORTEDARRAY
#define SRC_COMMON_SORTEDARRAY

#include "SortedArrayView.hpp"
#include "ArrayManipulation.hpp"

namespace geosx
{
namespace dataRepository
{

// Forward declaration for friend class purposes.
template <class U>
class ViewWrapper;

} // namespace dataRepository
} // namespace geosx

namespace LvArray
{

/**
 * @class SortedArray
 * @brief This class provides an interface similar to an std::set.
 * @tparam T type of data that is contained by the array.
 * @tparam INDEX_TYPE the integer to use for indexing the components of the array.
 *
 * The difference between this class and std::set is that the values are stored contiguously
 * in memory. It maintains O(log(N)) lookup time but insertion and removal are O(N).
 *
 * The derivation from SortedArrayView is protected to control the conversion to
 * SortedArrayView. Specifically only conversion to SortedArrayView<T const> is allowed.
 */
template <class T, class INDEX_TYPE=std::ptrdiff_t>
class SortedArray : protected SortedArrayView<T, INDEX_TYPE>
{
public:

  // ViewWrapper needs access to the data and resize methods, however these methods
  // need to be private so ViewWrapper is a friend class.
  template <class U>
  friend class geosx::dataRepository::ViewWrapper;

  // These are needed by ViewWrapper.
  using value_type = T;
  using pointer = T *;
  using const_pointer = T const *;

  // The iterators are defined for compatibility with std::set.
  using iterator = const_pointer;
  using const_iterator = const_pointer;

  // Alias public methods of SortedArrayView.
  using SortedArrayView<T, INDEX_TYPE>::values;
  using SortedArrayView<T, INDEX_TYPE>::operator[];
  using SortedArrayView<T, INDEX_TYPE>::begin;
  using SortedArrayView<T, INDEX_TYPE>::end;
  using SortedArrayView<T, INDEX_TYPE>::empty;
  using SortedArrayView<T, INDEX_TYPE>::size;
  using SortedArrayView<T, INDEX_TYPE>::contains;
  using SortedArrayView<T, INDEX_TYPE>::count;
  using SortedArrayView<T, INDEX_TYPE>::isSorted;

  /**
   * @brief Default constructor, the array is empty.
   */
  inline
  SortedArray() = default;

  /**
   * @brief The copy constructor, performs a deep copy.
   * @param [in] src the SortedArray to copy.
   */
  inline
  SortedArray( SortedArray const & src ):
    SortedArrayView<T, INDEX_TYPE>()
  { *this = src; }

  /**
   * @brief Default move constructor, performs a shallow copy.
   * @param [in/out] src the SortedArray to be moved from.
   */
  inline
  SortedArray( SortedArray && src ) = default;

  /**
   * @brief Destructor, frees the values array.
   */
  inline
  ~SortedArray() restrict_this
  { m_values.free(); }

  /**
   * @brief User defined conversion to SortedArrayView<T const> const.
   */
  template <class U = T>
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  operator SortedArrayView<T const, INDEX_TYPE> const & () const restrict_this
  { return reinterpret_cast<SortedArrayView<T const, INDEX_TYPE> const &>(*this); }

  /**
   * @brief Method to convert to SortedArrayView<T const> const. Use this method when
   * the above UDC isn't invoked, this usually occurs with template argument deduction.
   */
  LVARRAY_HOST_DEVICE inline
  SortedArrayView<T const, INDEX_TYPE> const & toView() const restrict_this
  { return *this; }

  /**
   * @brief Copy assignment operator, performs a deep copy.
   * @param [in] src the SortedArray to copy.
   */
  inline
  SortedArray & operator=( SortedArray const & src ) restrict_this
  {
    src.m_values.copy_into( m_values );
    return *this;
  }

  /**
   * @brief Default move assignment operator, performs a shallow copy.
   * @param [in/out] src the SortedArray to be moved from.
   */
  inline
  SortedArray & operator=( SortedArray && src ) = default;

  /**
   * @brief Remove all the values from the array.
   */
  inline
  void clear() restrict_this
  { m_values.clear(); }

  /**
   * @brief Insert the given value into the array if it doesn't already exist.
   * @param [in] value the value to insert.
   * @return True iff the value was actually inserted.
   */
  inline
  bool insert( T const & value ) restrict_this
  {
    bool const success = ArrayManipulation::insertSorted( data(), size(), value, CallBacks( m_values ));
    m_values.setSize( size() + success );
    return success;
  }

  /**
   * @brief Insert the given values into the array if they don't already exist.
   * @param [in] vals the values to insert, must be sorted.
   * @param [in] nVals the number of values to insert.
   * @return The number of values actually inserted.
   */
  inline
  INDEX_TYPE insertSorted( T const * const vals, INDEX_TYPE const nVals ) restrict_this
  {
    INDEX_TYPE const nInserted = ArrayManipulation::insertSorted( data(), size(), vals, nVals, CallBacks( m_values ));
    m_values.setSize( size() + nInserted );
    return nInserted;
  }

  /**
   * @brief Insert the given values into the array if they don't already exist.
   * @param [in] vals the values to insert.
   * @param [in] nVals the number of values to insert.
   * @return The number of values actually inserted.
   */
  inline
  INDEX_TYPE insert( T const * const vals, INDEX_TYPE const nVals ) restrict_this
  {
    INDEX_TYPE const nInserted = ArrayManipulation::insertSorted2( data(), size(), vals, nVals, CallBacks( m_values ));
    m_values.setSize( size() + nInserted );
    return nInserted;
  }

  /**
   * @brief Remove the given value from the array if it exists.
   * @param [in] value the value to remove.
   * @return True iff the value was actually removed.
   */
  inline
  bool erase( T const & value ) restrict_this
  {
    bool const success = ArrayManipulation::removeSorted( data(), size(), value, CallBacks( m_values ));
    m_values.setSize( size() - success );
    return success;
  }

  /**
   * @brief Remove the given values from the array if they exist.
   * @param [in] vals the values to remove, must be sorted.
   * @param [in] nVals the number of values to remove.
   * @return The number of values actually removed.
   */
  inline
  INDEX_TYPE eraseSorted( T const * const vals, INDEX_TYPE nVals ) restrict_this
  {
    INDEX_TYPE nRemoved = ArrayManipulation::removeSorted( data(), size(), vals, nVals, CallBacks( m_values ));
    m_values.setSize( size() - nRemoved );
    return nRemoved;
  }

  /**
   * @brief Remove the given values from the array if they exist.
   * @param [in] vals the values to remove.
   * @param [in] nVals the number of values to remove.
   * @return The number of values actually removed.
   */
  inline
  INDEX_TYPE erase( T const * const vals, INDEX_TYPE nVals ) restrict_this
  {
    INDEX_TYPE nRemoved = ArrayManipulation::removeSorted2( data(), size(), vals, nVals, CallBacks( m_values ));
    m_values.setSize( size() - nRemoved );
    return nRemoved;
  }

#ifdef USE_CHAI
  /**
   * @brief Moves the SortedArray to the given execution space.
   * @param [in] space the space to move to.
   */
  inline
  void move( chai::ExecutionSpace space ) restrict_this
  { m_values.move( space ); }
#endif

private:

  /**
   * @brief Return a non const pointer to the values.
   * @note This method is private because allowing access to the values in this manner
   * could destroy the sorted nature of the array.
   * @note the friend class ViewWrapper calls this method.
   */
  CONSTEXPRFUNC inline
  T * data() const restrict_this
  { return m_values.data(); }

  /**
   * @brief Return a non const pointer to the values.
   * @note This method is private because allowing access to the values in this manner
   * could destroy the sorted nature of the array.
   * @note the friend class ViewWrapper calls this method.
   */
  inline
  void resize( INDEX_TYPE newSize ) restrict_this
  { return m_values.resize( newSize ); }

  /**
   * @class CallBacks
   * @brief This class provides the callbacks for the ArrayManipulation sorted routines.
   */
  class CallBacks
  {
public:

    /**
     * @brief Constructor.
     * @param [in/out] cv the ChaiVector associated with the SortedArray.
     */
    inline
    CallBacks( ChaiVector<T> & cv ):
      m_cv( cv )
    {}

    /**
     * @brief Callback signaling that the size of the array has increased.
     * @param [in] nToAdd the increase in the size.
     * @note This method doesn't actually change the size but it can do reallocation.
     * @return a pointer to the new array.
     */
    inline
    T * incrementSize( INDEX_TYPE const nToAdd ) const restrict_this
    {
      INDEX_TYPE const newSize = m_cv.size() + nToAdd;
      INDEX_TYPE const capacity = m_cv.capacity();
      if( newSize > capacity )
      {
        m_cv.dynamicRealloc( newSize );
      }

      return m_cv.data();
    }

    /**
     * @brief These methods are placeholder and are no-ops.
     */
    /// @{
    inline
    void set( INDEX_TYPE, INDEX_TYPE ) const restrict_this
    {}

    inline
    void insert( INDEX_TYPE ) const restrict_this
    {}

    inline
    void insert( INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE ) const restrict_this
    {}

    inline
    void remove( INDEX_TYPE ) const restrict_this
    {}

    inline
    void remove( INDEX_TYPE, INDEX_TYPE, INDEX_TYPE ) const restrict_this
    {}
    /// @}

private:
    ChaiVector<T> & m_cv;
  };

  // Alias the protected member of SortedArrayView.
  using SortedArrayView<T, INDEX_TYPE>::m_values;
};

} // namespace LvArray

#endif // SRC_COMMON_SORTEDARRAY
