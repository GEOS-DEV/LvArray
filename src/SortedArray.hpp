/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file SortedArray.hpp
 * @brief Contains the implementation of LvArray::SortedArray.
 */

#pragma once

#include "SortedArrayView.hpp"

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
template< typename T,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
class SortedArray : protected SortedArrayView< T, INDEX_TYPE, BUFFER_TYPE >
{
public:

  /// Alias for the parent class
  using ParentClass = SortedArrayView< T, INDEX_TYPE, BUFFER_TYPE >;

  // Alias public typedefs of SortedArrayView.
  using typename ParentClass::ValueType;
  using typename ParentClass::IndexType;
  using typename ParentClass::value_type;
  using typename ParentClass::size_type;

  /// The view type.
  using ViewType = SortedArrayView< T const, INDEX_TYPE, BUFFER_TYPE > const;

  /**
   * @brief The const view type
   * @note This is the same as the view type since SortedArrayView can't modify the data.
   */
  using ViewTypeConst = SortedArrayView< T const, INDEX_TYPE, BUFFER_TYPE > const;

  // Alias public methods of SortedArrayView.

  /**
   * @name Constructors, destructor and assignment operators.
   */
  ///@{

  /**
   * @brief Default constructor.
   */
  inline
  SortedArray():
    ParentClass()
  { setName( "" ); }

  /**
   * @brief The copy constructor, performs a deep copy.
   * @param src The SortedArray to copy.
   */
  inline
  SortedArray( SortedArray const & src ):
    ParentClass()
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
  ~SortedArray()
  { bufferManipulation::free( this->m_values, size() ); }

  /**
   * @brief Copy assignment operator, performs a deep copy.
   * @param src the SortedArray to copy.
   * @return *this.
   */
  inline
  SortedArray & operator=( SortedArray const & src )
  {
    bufferManipulation::copyInto( this->m_values, size(), src.m_values, src.size() );
    this->m_size = src.size();
    return *this;
  }

  /**
   * @brief Default move assignment operator, performs a shallow copy.
   * @param src the SortedArray to be moved from.
   * @return *this.
   */
  inline
  SortedArray & operator=( SortedArray && src )
  {
    bufferManipulation::free( this->m_values, size() );
    ParentClass::operator=( std::move( src ) );
    return *this;
  }

  ///@}

  /**
   * @name SortedArrayView creation methods
   */
  ///@{

  /**
   * @copydoc ParentClass::toView()
   * @note This is just a wrapper around the SortedArrayView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  LVARRAY_HOST_DEVICE inline
  SortedArrayView< T const, INDEX_TYPE, BUFFER_TYPE > toView() const &
  { return ParentClass::toView(); }

  /**
   * @brief Overload for rvalues that is deleted.
   * @return A null SortedArrayView.
   * @note This cannot be called on a rvalue since the @c SortedArrayView would
   *   contain the buffer of the current @c SortedArray that is about to be destroyed.
   *   This overload prevents that from happening.
   */
  LVARRAY_HOST_DEVICE inline
  SortedArrayView< T const, INDEX_TYPE, BUFFER_TYPE > toView() const && = delete;

  /**
   * @copydoc ParentClass::toViewConst()
   * @note This is just a wrapper around the SortedArrayView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  LVARRAY_HOST_DEVICE inline
  SortedArrayView< T const, INDEX_TYPE, BUFFER_TYPE > toViewConst() const &
  { return ParentClass::toViewConst(); }

  /**
   * @brief Overload for rvalues that is deleted.
   * @return A null SortedArrayView.
   * @note This cannot be called on a rvalue since the @c SortedArrayView would
   *   contain the buffer of the current @c SortedArray that is about to be destroyed.
   *   This overload prevents that from happening.
   */
  LVARRAY_HOST_DEVICE inline
  SortedArrayView< T const, INDEX_TYPE, BUFFER_TYPE > toViewConst() const && = delete;

  using ParentClass::toSlice;

  ///@}

  /**
   * @name Attribute querying methods
   */
  ///@{

  using ParentClass::empty;

  /**
   * @copydoc SortedArrayView::size
   * @note This is just a wrapper around the SortedArrayView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  INDEX_TYPE size() const
  { return ParentClass::size(); }

  using ParentClass::contains;
  using ParentClass::count;

  ///@}

  /**
   * @name Methods that provide access to the data.
   */
  ///@{

  using ParentClass::operator[];
  using ParentClass::operator();

  /**
   * @copydoc SortedArrayView::data
   * @note This is just a wrapper around the SortedArrayView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  T const * data() const
  { return ParentClass::data(); }

  using ParentClass::begin;
  using ParentClass::end;

  ///@}

  /**
   * @name Methods to insert or remove values
   */
  ///@{

  /**
   * @brief Insert the given value into the array if it doesn't already exist.
   * @param value the value to insert.
   * @return True iff the value was actually inserted.
   */
  inline
  bool insert( T const & value )
  {
    bool const success = sortedArrayManipulation::insert( this->m_values.data(),
                                                          size(),
                                                          value,
                                                          CallBacks( this->m_values, size() ) );
    this->m_size += success;
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
  INDEX_TYPE insert( ITER const first, ITER const last )
  {
    INDEX_TYPE const nInserted = sortedArrayManipulation::insert( this->m_values.data(),
                                                                  size(),
                                                                  first,
                                                                  last,
                                                                  CallBacks( this->m_values, size() ) );
    this->m_size += nInserted;
    return nInserted;
  }

  /**
   * @brief Remove the given value from the array if it exists.
   * @param value the value to remove.
   * @return True iff the value was actually removed.
   */
  inline
  bool remove( T const & value )
  {
    bool const success = sortedArrayManipulation::remove( this->m_values.data(),
                                                          size(),
                                                          value,
                                                          CallBacks( this->m_values, size() ) );
    this->m_size -= success;
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
  INDEX_TYPE remove( ITER const first, ITER const last )
  {
    INDEX_TYPE const nRemoved = sortedArrayManipulation::remove( this->m_values.data(),
                                                                 size(),
                                                                 first,
                                                                 last,
                                                                 CallBacks( this->m_values, size() ) );
    this->m_size -= nRemoved;
    return nRemoved;
  }

  ///@}

  /**
   * @name Methods that modify the size or capacity
   */
  ///@{

  /**
   * @brief Remove all the values from the array.
   */
  inline
  void clear()
  {
    bufferManipulation::resize( this->m_values, size(), 0 );
    this->m_size = 0;
  }

  /**
   * @brief Reserve space to store the given number of values without resizing.
   * @param nVals the number of values to reserve space for.
   */
  inline
  void reserve( INDEX_TYPE const nVals )
  { bufferManipulation::reserve( this->m_values, size(), MemorySpace::host, nVals ); }

  ///@}

  /**
   * @name Methods dealing with memory spaces
   */
  ///@{

  /**
   * @copydoc SortedArrayView::move
   * @note This is just a wrapper around the SortedArrayView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  inline
  void move( MemorySpace const space, bool const touch=true ) const
  { ParentClass::move( space, touch ); }

  ///@}

  /**
   * @brief Set the name to be displayed whenever the underlying Buffer's user call back is called.
   * @param name The name to associate with this SortedArray.
   */
  void setName( std::string const & name )
  { this->m_values.template setName< decltype( *this ) >( name ); }

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
     * @param buffer The buffer associated with the SortedArray.
     * @param size The size of the SortedArray.
     */
    inline
    CallBacks( BUFFER_TYPE< T > & buffer, INDEX_TYPE const size ):
      m_buffer( buffer ),
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
    T * incrementSize( T * const curPtr,
                       INDEX_TYPE const nToAdd ) const
    {
      LVARRAY_UNUSED_VARIABLE( curPtr );
      bufferManipulation::dynamicReserve( m_buffer, m_size, m_size + nToAdd );
      return m_buffer.data();
    }

private:
    /// The buffer associated with the callback.
    BUFFER_TYPE< T > & m_buffer;

    /// The number of values in the buffer.
    INDEX_TYPE const m_size;
  };
};

/**
 * @brief True if the template type is a SortedArray.
 */
template< class >
constexpr bool isSortedArray = false;

/**
 * @tparam T The type contained in the SortedArray.
 * @tparam INDEX_TYPE The integral type used as an index.
 * @tparam BUFFER_TYPE The type used to manager the underlying allocation.
 * @brief Specialization of isSortedArrayView for the SortedArray class.
 */
template< class T,
          class INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
constexpr bool isSortedArray< SortedArray< T, INDEX_TYPE, BUFFER_TYPE > > = true;


} // namespace LvArray
