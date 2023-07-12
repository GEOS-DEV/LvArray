/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file SortedArrayView.hpp
 * @brief Contains the implementation of LvArray::SortedArrayView.
 */

#pragma once

// Source includes
#include "ArraySlice.hpp"
#include "bufferManipulation.hpp"
#include "sortedArrayManipulation.hpp"

#ifdef LVARRAY_BOUNDS_CHECK

/**
 * @brief Check that @p index falls within the size of the array.
 * @param index The index to check.
 * @note This is only active when LVARRAY_BOUNDS_CHECK is defined.
 */
#define SORTEDARRAY_CHECK_BOUNDS( index ) \
  LVARRAY_ERROR_IF( index < 0 || index >= size(), \
                    "Array Bounds Check Failed: index=" << index << " size()=" << size())

#else // LVARRAY_BOUNDS_CHECK

/**
 * @brief Check that @p index falls within the size of the first dimension.
 * @param index The index to check.
 * @note This is only active when LVARRAY_BOUNDS_CHECK is defined.
 */
#define SORTEDARRAY_CHECK_BOUNDS( index )

#endif // LVARRAY_BOUNDS_CHECK

namespace LvArray
{

/**
 * @tparam T type of data that is contained by the array.
 * @tparam INDEX_TYPE the integer to use for indexing.
 * @class SortedArrayView
 * @brief This class provides a view into a SortedArray.
 *
 * @details When using a SortedArrayView directly the template parameter T should be const
 *   since the View has no way of modifying the values. This also prevents unnecessary
 *   memory movement.
 */
template< typename T,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
class SortedArrayView
{
public:

  /// The type of the values contained in the SortedArrayView
  using ValueType = T;

  /// The integer type used for indexing.
  using IndexType = INDEX_TYPE;

  /// The type of the values contained in the SortedArrayView, here for stl compatability.
  using value_type = T;

  /// The integer type used for indexing, here for stl compatability.
  using size_type = INDEX_TYPE;

  /**
   * @name Constructors, destructor and assignment operators.
   */
  ///@{

  /**
   * @brief Default copy constructor. Performs a shallow copy and calls the
   *        chai::ManagedArray copy constructor.
   * @param src the SortedArray to copy.
   */
  SortedArrayView( SortedArrayView const & src ) = default;

  /**
   * @brief Default move constructor, performs a shallow copy.
   * @param src The SortedArray to be moved from.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  SortedArrayView( SortedArrayView && src ):
    m_values( std::move( src.m_values ) ),
    m_size( src.m_size )
  {
    src.m_size = 0;
  }

  /**
   * @brief Construct a new SortedArrayView from the given buffer.
   * @param size The number of entries.
   * @param buffer The values buffer, of size @p size.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  SortedArrayView( INDEX_TYPE const size, BUFFER_TYPE< T > const & buffer ):
    m_values( buffer ),
    m_size( size )
  {}

  /**
   * @brief Default copy assignment operator, this does a shallow copy.
   * @param src The SortedArray to copy.
   * @return *this.
   */
  SortedArrayView & operator=( SortedArrayView const & src ) = default;

  /**
   * @brief Default move assignment operator, this does a shallow copy.
   * @param src the SortedArray to be moved from.
   * @return *this.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  SortedArrayView & operator=( SortedArrayView && src )
  {
    m_values = std::move( src.m_values );
    m_size = src.m_size;
    src.m_size = 0;
    return *this;
  }

  ///@}

  /**
   * @name SortedArrayView and ArraySlice creation methods
   */
  ///@{

  /**
   * @return An immutable SortedArrayView.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  SortedArrayView< T const, INDEX_TYPE, BUFFER_TYPE >
  toView() const
  { return SortedArrayView< T const, INDEX_TYPE, BUFFER_TYPE >( size(), m_values ); }

  /**
   * @return An immutable SortedArrayView.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  SortedArrayView< T const, INDEX_TYPE, BUFFER_TYPE >
  toViewConst() const
  { return toView(); }

  /**
   * @return Return an ArraySlice representing this SortedArrayView.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  ArraySlice< T const, 1, 0, INDEX_TYPE > toSlice() const &
  { return ArraySlice< T const, 1, 0, INDEX_TYPE >( data(), &m_size, nullptr ); }

  /**
   * @brief Overload for rvalues that is deleted.
   * @return A null ArraySlice.
   * @note This cannot be called on a rvalue since the @c ArraySlice would
   *   contain pointers to the size of the current @c SortedArrayView that is
   *   about to be destroyed. This overload prevents that from happening.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  ArraySlice< T const, 1, 0, INDEX_TYPE > toSlice() const && = delete;

  ///@}

  /**
   * @name Attribute querying methods
   */
  ///@{

  /**
   * @return Return true if the array holds no values.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  bool empty() const
  { return size() == 0; }

  /**
   * @return Return the number of values in the array.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  INDEX_TYPE size() const
  { return m_size; }

  /**
   * @return Return true if the @p value is in the array.
   * @param value the value to search for.
   */
  LVARRAY_HOST_DEVICE inline
  bool contains( T const & value ) const
  { return sortedArrayManipulation::contains( data(), size(), value ); }

  /**
   * @return Return true if the given value is in the array.
   * @param value the value to search for.
   * @note This is a alias for contains to conform to the std::set interface.
   */
  LVARRAY_HOST_DEVICE inline
  bool count( T const & value ) const
  { return contains( value ); }

  ///@}

  /**
   * @name Methods that provide access to the data.
   */
  ///@{

  /**
   * @return Return the value at position @p i .
   * @param i the index of the value to access.
   */
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  T const & operator[]( INDEX_TYPE const i ) const
  {
    SORTEDARRAY_CHECK_BOUNDS( i );
    return data()[ i ];
  }

  /**
   * @return Return the value at position @p i .
   * @param i the index of the value to access.
   */
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  T const & operator()( INDEX_TYPE const i ) const
  {
    SORTEDARRAY_CHECK_BOUNDS( i );
    return data()[ i ];
  }

  /**
   * @return Return a pointer to the values.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  T const * data() const
  { return m_values.data(); }

  /**
   * @return Return an iterator to the beginning of the array.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  T const * begin() const
  { return data(); }

  /**
   * @return Return an iterator to the end of the array.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  T const * end() const
  { return data() + size(); }

  ///@}

  /**
   * @name Methods dealing with memory spaces
   */
  ///@{

  /**
   * @brief Moves the SortedArrayView to the given execution space.
   * @param space the space to move to.
   * @param touch If the values will be modified in the new space.
   * @note Since the SortedArrayView can't be modified on device when moving
   *   to the GPU @p touch is set to false.
   */
  inline
  void move( MemorySpace const space, bool touch=true ) const
  {
  #if defined(LVARRAY_USE_CUDA)
    if( space == MemorySpace::cuda ) touch = false;
  #endif
  #if defined(LVARRAY_USE_HIP)
    if( space == MemorySpace::hip ) touch = false;
  #endif
    m_values.move( space, touch );
  }

  ///@}

protected:

  /**
   * @brief Default constructor.
   * @note Protected since every SortedArrayView should either be the base of a
   *  SortedArray or copied from another SortedArrayView.
   */
  SortedArrayView():
    m_values( true )
  {}

  /// Holds the array of values.
  BUFFER_TYPE< T > m_values;

  /// The number of values
  INDEX_TYPE m_size = 0;
};

/**
 * @brief True if the template type is a SortedArrayView.
 */
template< class >
constexpr bool isSortedArrayView = false;

/**
 * @tparam T The type contained in the SortedArrayView.
 * @tparam INDEX_TYPE The integral type used as an index.
 * @tparam BUFFER_TYPE The type used to manager the underlying allocation.
 * @brief Specialization of isSortedArrayView for the SortedArrayView class.
 */
template< class T,
          class INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
constexpr bool isSortedArrayView< SortedArrayView< T, INDEX_TYPE, BUFFER_TYPE > > = true;

} // namespace LvArray
