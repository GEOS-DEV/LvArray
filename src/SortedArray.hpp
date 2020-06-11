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

  // Alias public typedefs of SortedArrayView.
  using typename SortedArrayView< T, INDEX_TYPE >::value_type;
  using typename SortedArrayView< T, INDEX_TYPE >::iterator;
  using typename SortedArrayView< T, INDEX_TYPE >::const_iterator;
  using typename SortedArrayView< T, INDEX_TYPE >::pointer;
  using typename SortedArrayView< T, INDEX_TYPE >::const_pointer;

  // Alias public methods of SortedArrayView.
  using SortedArrayView< T, INDEX_TYPE >::operator[];
  using SortedArrayView< T, INDEX_TYPE >::begin;
  using SortedArrayView< T, INDEX_TYPE >::end;
  using SortedArrayView< T, INDEX_TYPE >::contains;
  using SortedArrayView< T, INDEX_TYPE >::count;

  /**
   * @brief Default constructor.
   */
  inline
  SortedArray():
    SortedArrayView< T, INDEX_TYPE >()
  { setName( "" ); }

  /**
   * @brief The copy constructor, performs a deep copy.
   * @param src The SortedArray to copy.
   */
  inline
  SortedArray( SortedArray const & src ):
    SortedArrayView< T, INDEX_TYPE >()
  { *this = src; }

  /**
   * @brief Default move constructor, performs a shallow copy.
   * @param src the SortedArray to be moved from.
   */
  inline
  SortedArray( SortedArray && src ) = default;

  /**
   * @brief Destructor, frees the values array.
   */
  inline
  ~SortedArray() LVARRAY_RESTRICT_THIS
  { bufferManipulation::free( m_values, size() ); }

  /**
   * @brief Copy assignment operator, performs a deep copy.
   * @param src the SortedArray to copy.
   * @return *this.
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
   * @param src the SortedArray to be moved from.
   * @return *this.
   */
  inline
  SortedArray & operator=( SortedArray && src ) = default;

  /**
   * @brief @return A reference to *this reinterpreted as a SortedArrayView<T const> const.
   */
  LVARRAY_HOST_DEVICE inline
  SortedArrayView< T const, INDEX_TYPE > const & toView() const LVARRAY_RESTRICT_THIS
  { return SortedArrayView< T, INDEX_TYPE >::toViewConst(); }

  /**
   * @brief @return A reference to *this reinterpreted as a SortedArrayView<T const> const.
   */
  LVARRAY_HOST_DEVICE inline
  SortedArrayView< T const, INDEX_TYPE > const & toViewConst() const LVARRAY_RESTRICT_THIS
  { return SortedArrayView< T, INDEX_TYPE >::toViewConst(); }

  /**
   * @brief @return Return true iff the SortedArray contains not values.
   * @note Duplicated for SFINAE needs.
   */
  constexpr inline
  bool empty() const
  { return SortedArrayView< T, INDEX_TYPE >::empty(); }

  /**
   * @brief @return Return the number of values in the SortedArray.
   * @note Duplicated for SFINAE needs.
   */
  constexpr inline
  INDEX_TYPE size() const
  { return SortedArrayView< T, INDEX_TYPE >::size(); }

  /**
   * @brief @return Return a pointer to the values.
   * @note Duplicated for SFINAE needs.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  T const * data() const
  { return SortedArrayView< T, INDEX_TYPE >::data(); }

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
   * @param nVals the number of values to reserve space for.
   */
  inline
  void reserve( INDEX_TYPE const nVals ) LVARRAY_RESTRICT_THIS
  { bufferManipulation::reserve( m_values, size(), nVals ); }

  /**
   * @brief Insert the given value into the array if it doesn't already exist.
   * @param value the value to insert.
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
   * @tparam ITER The type of the iterator to use.
   * @brief Insert the values in [ @p first, @p last ) into the array if they don't already exist.
   * @param first Iterator to the first value to insert.
   * @param last Iterator to the end of the values to insert.
   * @return The number of values actually inserted.
   * @note [ @p first, @p last ) must be sorted and unique.
   */
  template< typename ITER >
  INDEX_TYPE insert( ITER const first, ITER const last ) LVARRAY_RESTRICT_THIS
  {
    INDEX_TYPE const nInserted = sortedArrayManipulation::insert( m_values.data(), size(), first, last, CallBacks( m_values, size() ) );
    m_size += nInserted;
    return nInserted;
  }

  /**
   * @brief Remove the given value from the array if it exists.
   * @param value the value to remove.
   * @return True iff the value was actually removed.
   */
  inline
  bool remove( T const & value ) LVARRAY_RESTRICT_THIS
  {
    bool const success = sortedArrayManipulation::remove( m_values.data(), size(), value, CallBacks( m_values, size() ) );
    m_size -= success;
    return success;
  }

  /**
   * @tparam ITER The type of the iterator to use.
   * @brief Remove the values in [ @p first, @p last ) from the array if they exist.
   * @param first Iterator to the first value to remove.
   * @param last Iterator to the end of the values to remove.
   * @return The number of values actually removed.
   * @note [ @p first, @p last ) must be sorted and unique.
   */
  template< typename ITER >
  INDEX_TYPE remove( ITER const first, ITER const last ) LVARRAY_RESTRICT_THIS
  {
    INDEX_TYPE const nRemoved = sortedArrayManipulation::remove( m_values.data(), size(), first, last, CallBacks( m_values, size() ) );
    m_size -= nRemoved;
    return nRemoved;
  }

  /**
   * @brief Set the name to be displayed whenever the underlying Buffer's user call back is called.
   * @param name The name to associate with this SortedArray.
   */
  void setName( std::string const & name )
  { m_values.template setName< decltype( *this ) >( name ); }

  /**
   * @brief Moves the SortedArrayView to the given execution space.
   * @param space the space to move to.
   * @param touch If the values will be modified in the new space.
   * @note Since the SortedArrayView can't be modified on device when moving
   *       to the GPU @p touch is set to false.
   * @note Duplicated for SFINAE needs.
   */
  inline
  void move( chai::ExecutionSpace const space, bool touch=true ) const LVARRAY_RESTRICT_THIS
  { return SortedArrayView< T, INDEX_TYPE >::move( space, touch ); }

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
     * @param cv the NewChaiBuffer associated with the SortedArray.
     */
    inline
    CallBacks( NewChaiBuffer< T > & cb, INDEX_TYPE const size ):
      m_cb( cb ),
      m_size( size )
    {}

    /**
     * @brief Callback signaling that the size of the array has increased.
     * @param curPtr The current pointer to the array.
     * @param nToAdd The increase in the size.
     * @note This method doesn't actually change the size but it can do reallocation.
     * @return A pointer to the new array.
     */
    inline
    T * incrementSize( T * const LVARRAY_UNUSED_ARG( curPtr ),
                       INDEX_TYPE const nToAdd ) const LVARRAY_RESTRICT_THIS
    {
      bufferManipulation::dynamicReserve( m_cb, m_size, m_size + nToAdd );
      return m_cb.data();
    }

private:
    /// The buffer associated with the callback.
    NewChaiBuffer< T > & m_cb;

    /// The number of values in the buffer.
    INDEX_TYPE const m_size;
  };

  // Alias the protected members of SortedArrayView.
  using SortedArrayView< T, INDEX_TYPE >::m_values;
  using SortedArrayView< T, INDEX_TYPE >::m_size;
};

} // namespace LvArray

#endif // SRC_COMMON_SORTEDARRAY
