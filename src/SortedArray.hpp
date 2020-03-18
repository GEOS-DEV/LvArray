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
#include "sortedArrayManipulation.hpp"

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
template< class T, class INDEX_TYPE=std::ptrdiff_t >
class SortedArray : protected SortedArrayView< T, INDEX_TYPE >
{
public:

  // These are needed by Wrapper.
  using value_type = T;
  using pointer = T const *;
  using const_pointer = T const *;

  // The iterators are defined for compatibility with std::set.
  using iterator = const_pointer;
  using const_iterator = const_pointer;

  // Alias public methods of SortedArrayView.
  using SortedArrayView< T, INDEX_TYPE >::data;
  using SortedArrayView< T, INDEX_TYPE >::operator[];
  using SortedArrayView< T, INDEX_TYPE >::begin;
  using SortedArrayView< T, INDEX_TYPE >::end;

  // Duplicating these next two methods because SFINAE macros don't seem to pick them up otherwise.

  // using SortedArrayView<T, INDEX_TYPE>::empty;
  constexpr inline
  bool empty() const
  { return SortedArrayView< T, INDEX_TYPE >::empty(); }

  // using SortedArrayView<T, INDEX_TYPE>::size;
  constexpr inline
  INDEX_TYPE size() const
  { return SortedArrayView< T, INDEX_TYPE >::size(); }

  using SortedArrayView< T, INDEX_TYPE >::contains;
  using SortedArrayView< T, INDEX_TYPE >::count;
  using SortedArrayView< T, INDEX_TYPE >::move;

  using ViewType = SortedArrayView< T const, INDEX_TYPE >;
  using ViewTypeConst = ViewType;

  /**
   * @brief Default constructor, the array is empty.
   */
  inline
  SortedArray():
    SortedArrayView< T, INDEX_TYPE >()
  {
    setName( "" );
  }

  /**
   * @brief The copy constructor, performs a deep copy.
   * @param [in] src the SortedArray to copy.
   */
  inline
  SortedArray( SortedArray const & src ):
    SortedArrayView< T, INDEX_TYPE >()
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
  ~SortedArray() LVARRAY_RESTRICT_THIS
  {
    bufferManipulation::free( m_values, size() );
  }

  /**
   * @brief Copy assignment operator, performs a deep copy.
   * @param [in] src the SortedArray to copy.
   */
  inline
  SortedArray & operator=( SortedArray const & src ) LVARRAY_RESTRICT_THIS
  {
    bufferManipulation::copyInto( m_values, size(), src.m_values, src.size() );
    m_size = src.size();
    return *this;
  }

  /**
   * @brief Default move assignment operator, performs a shallow copy.
   * @param [in/out] src the SortedArray to be moved from.
   */
  inline
  SortedArray & operator=( SortedArray && src ) = default;

  /**
   * @brief User defined conversion to SortedArrayView<T const> const.
   */
  template< typename U = T >
  LVARRAY_HOST_DEVICE constexpr inline
  operator ViewType const & () const LVARRAY_RESTRICT_THIS
  { return reinterpret_cast< ViewType const & >( *this ); }

  /**
   * @brief Method to convert to SortedArrayView<T const> const. Use this method when
   *        the above UDC isn't invoked, this usually occurs with template argument deduction.
   */
  LVARRAY_HOST_DEVICE inline
  ViewType const & toView() const LVARRAY_RESTRICT_THIS
  { return *this; }

  /**
   * @brief Duplicate method to placate Wrapper's SFINAE.
   */
  LVARRAY_HOST_DEVICE inline
  ViewType const & toViewConst() const LVARRAY_RESTRICT_THIS
  { return *this; }

  /**
   * @brief Remove all the values from the array.
   */
  inline
  void clear() LVARRAY_RESTRICT_THIS
  {
    bufferManipulation::resize( m_values, size(), 0 );
    m_size = 0;
  }

  /**
   * @brief Reserve space to store the given number of values without resizing.
   * @param [in] nVals the number of values to reserve space for.
   */
  inline
  void reserve( INDEX_TYPE const nVals ) LVARRAY_RESTRICT_THIS
  {
    bufferManipulation::reserve( m_values, size(), nVals );
  }

  /**
   * @brief Insert the given value into the array if it doesn't already exist.
   * @param [in] value the value to insert.
   * @return True iff the value was actually inserted.
   */
  inline
  bool insert( T const & value ) LVARRAY_RESTRICT_THIS
  {
    bool const success = sortedArrayManipulation::insert( m_values.data(), size(), value, CallBacks( m_values, size() ) );
    m_size += success;
    return success;
  }

  /**
   * @brief Insert the given values into the array if they don't already exist.
   * @param [in] vals the values to insert.
   * @param [in] nVals the number of values to insert.
   * @return The number of values actually inserted.
   *
   * @note If possible sort vals first by calling sortedArrayManipulation::makeSorted(vals, nVals)
   *       and then call insertSorted, this will be substantially faster.
   */
  inline
  INDEX_TYPE insert( T const * const vals, INDEX_TYPE const nVals ) LVARRAY_RESTRICT_THIS
  {
    INDEX_TYPE const nInserted = sortedArrayManipulation::insert( m_values.data(), size(), vals, nVals, CallBacks( m_values, size() ) );
    m_size += nInserted;
    return nInserted;
  }

  /**
   * @brief Insert the given values into the array if they don't already exist.
   * @param [in] vals the values to insert, must be sorted.
   * @param [in] nVals the number of values to insert.
   * @return The number of values actually inserted.
   */
  inline
  INDEX_TYPE insertSorted( T const * const vals, INDEX_TYPE const nVals ) LVARRAY_RESTRICT_THIS
  {
    INDEX_TYPE const nInserted = sortedArrayManipulation::insertSorted( m_values.data(), size(), vals, nVals, CallBacks( m_values, size() ) );
    m_size += nInserted;
    return nInserted;
  }

  /**
   * @brief Remove the given value from the array if it exists.
   * @param [in] value the value to remove.
   * @return True iff the value was actually removed.
   */
  inline
  bool erase( T const & value ) LVARRAY_RESTRICT_THIS
  {
    bool const success = sortedArrayManipulation::remove( m_values.data(), size(), value, CallBacks( m_values, size() ) );
    m_size -= success;
    return success;
  }

  /**
   * @brief Remove the given values from the array if they exist.
   * @param [in] vals the values to remove.
   * @param [in] nVals the number of values to remove.
   * @return The number of values actually removed.
   *
   * @note If possible sort vals first by calling sortedArrayManipulation::makeSorted(vals, nVals)
   *       and then call eraseSorted, this will be substantially faster.
   */
  inline
  INDEX_TYPE erase( T const * const vals, INDEX_TYPE nVals ) LVARRAY_RESTRICT_THIS
  {
    INDEX_TYPE const nRemoved = sortedArrayManipulation::remove( m_values.data(), size(), vals, nVals, CallBacks( m_values, size() ) );
    m_size -= nRemoved;
    return nRemoved;
  }

  /**
   * @brief Remove the given values from the array if they exist.
   * @param [in] vals the values to remove, must be sorted.
   * @param [in] nVals the number of values to remove.
   * @return The number of values actually removed.
   */
  inline
  INDEX_TYPE eraseSorted( T const * const vals, INDEX_TYPE nVals ) LVARRAY_RESTRICT_THIS
  {
    INDEX_TYPE const nRemoved = sortedArrayManipulation::removeSorted( m_values.data(), size(), vals, nVals, CallBacks( m_values, size() ) );
    m_size -= nRemoved;
    return nRemoved;
  }

  void setName( std::string const & name )
  {
    m_values.template setName< decltype( *this ) >( name );
  }

  friend std::ostream & operator<< ( std::ostream & stream, SortedArray const & array )
  {
    stream << array.toView();
    return stream;
  }

private:

  /**
   * @class CallBacks
   * @brief This class provides the callbacks for the sortedArrayManipulation sorted routines.
   */
  class CallBacks : public sortedArrayManipulation::CallBacks< T >
  {
public:

    /**
     * @brief Constructor.
     * @param [in/out] cv the NewChaiBuffer associated with the SortedArray.
     */
    inline
    CallBacks( NewChaiBuffer< T > & cb, INDEX_TYPE const size ):
      m_cb( cb ),
      m_size( size )
    {}

    /**
     * @brief Callback signaling that the size of the array has increased.
     * @param [in] curPtr the current pointer to the array.
     * @param [in] nToAdd the increase in the size.
     * @note This method doesn't actually change the size but it can do reallocation.
     * @return a pointer to the new array.
     */
    inline
    T * incrementSize( T * const LVARRAY_UNUSED_ARG( curPtr ),
                       INDEX_TYPE const nToAdd ) const LVARRAY_RESTRICT_THIS
    {
      bufferManipulation::dynamicReserve( m_cb, m_size, m_size + nToAdd );
      return m_cb.data();
    }

private:
    NewChaiBuffer< T > & m_cb;
    INDEX_TYPE const m_size;
  };

  // Alias the protected members of SortedArrayView.
  using SortedArrayView< T, INDEX_TYPE >::m_values;
  using SortedArrayView< T, INDEX_TYPE >::m_size;
};

} // namespace LvArray

#endif // SRC_COMMON_SORTEDARRAY
