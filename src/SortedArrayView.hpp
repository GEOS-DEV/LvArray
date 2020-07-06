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
 * @file SortedArrayView.hpp
 */

#ifndef SRC_COMMON_SORTEDARRAYVIEW
#define SRC_COMMON_SORTEDARRAYVIEW

// Source includes
#include "bufferManipulation.hpp"
#include "sortedArrayManipulation.hpp"

#ifdef USE_ARRAY_BOUNDS_CHECK

/**
 * @brief Check that @p index falls within the size of the array.
 * @param index The index to check.
 * @note This is only active when USE_ARRAY_BOUNDS_CHECK is defined.
 */
#define SORTEDARRAY_CHECK_BOUNDS( index ) \
  LVARRAY_ERROR_IF( index < 0 || index >= size(), \
                    "Array Bounds Check Failed: index=" << index << " size()=" << size())

#else // USE_ARRAY_BOUNDS_CHECK

/**
 * @brief Check that @p index falls within the size of the first dimension.
 * @param index The index to check.
 * @note This is only active when USE_ARRAY_BOUNDS_CHECK is defined.
 */
#define SORTEDARRAY_CHECK_BOUNDS( index )

#endif // USE_ARRAY_BOUNDS_CHECK

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

  /// The type of the value in the SortedArrayView.
  using value_type = T;

  /// The type of the iterator used.
  using iterator = T const *;

  /// The type of the const iterator used.
  using const_iterator = T const *;

  /// The pointer type.
  using pointer = T *;

  /// The const_pointer type.
  using const_pointer = T const *;

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
  LVARRAY_HOST_DEVICE inline
  SortedArrayView( SortedArrayView && src ):
    m_values( std::move( src.m_values ) ),
    m_size( src.m_size )
  {
    src.m_size = 0;
  }

  /**
   * @brief Default copy assignment operator, this does a shallow copy.
   * @param src The SortedArray to copy.
   * @return *this.
   */
  inline
  SortedArrayView & operator=( SortedArrayView const & src ) = default;

  /**
   * @brief Default move assignment operator, this does a shallow copy.
   * @param src the SortedArray to be moved from.
   * @return *this.
   */
  LVARRAY_HOST_DEVICE inline
  SortedArrayView & operator=( SortedArrayView && src )
  {
    m_values = std::move( src.m_values );
    m_size = src.m_size;
    src.m_size = 0;
    return *this;
  }

  /**
   * @brief @return A reference to *this.
   */
  LVARRAY_HOST_DEVICE inline
  SortedArrayView const & toView() const LVARRAY_RESTRICT_THIS
  { return *this; }

  /**
   * @brief @return A reference to *this.
   */
  LVARRAY_HOST_DEVICE inline
  SortedArrayView< T const, INDEX_TYPE, BUFFER_TYPE > const & toViewConst() const LVARRAY_RESTRICT_THIS
  { return reinterpret_cast< SortedArrayView< T const, INDEX_TYPE, BUFFER_TYPE > const & >( *this ); }

  /**
   * @brief @return Return a pointer to the values.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  T const * data() const
  { return m_values.data(); }

  /**
   * @brief @return Return the value at position @p i .
   * @param i the index of the value to access.
   */
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  T const & operator[]( INDEX_TYPE const i ) const
  {
    SORTEDARRAY_CHECK_BOUNDS( i );
    return data()[ i ];
  }

  /**
   * @brief @return Return an iterator to the beginning of the array.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  iterator begin() const
  { return data(); }

  /**
   * @brief @return Return an iterator to the end of the array.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  iterator end() const
  { return data() + size(); }

  /**
   * @brief @return Return true if the array holds no values.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  bool empty() const
  { return size() == 0; }

  /**
   * @brief @return Return the number of values in the array.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  INDEX_TYPE size() const
  { return m_size; }

  /**
   * @brief @return Return true if the @p value is in the array.
   * @param value the value to search for.
   */
  LVARRAY_HOST_DEVICE inline
  bool contains( T const & value ) const
  { return sortedArrayManipulation::contains( data(), size(), value ); }

  /**
   * @brief @return Return true if the given value is in the array.
   * @param value the value to search for.
   * @note This is a alias for contains to conform to the std::set interface.
   */
  LVARRAY_HOST_DEVICE inline
  bool count( const T & value ) const
  { return contains( value ); }

  /**
   * @brief Moves the SortedArrayView to the given execution space.
   * @param space the space to move to.
   * @param touch If the values will be modified in the new space.
   * @note Since the SortedArrayView can't be modified on device when moving
   *       to the GPU @p touch is set to false.
   */
  inline
  void move( MemorySpace const space, bool touch=true ) const LVARRAY_RESTRICT_THIS
  {
  #if defined(USE_CUDA)
    if( space == MemorySpace::GPU ) touch = false;
  #endif
    m_values.move( space, touch );
  }

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
 * @tparam INDEX_TYPE The type used to index the SortedArrayView.
 * @brief Specialization of isSortedArrayView for the SortedArrayView class.
 */
template< class T, class INDEX_TYPE >
constexpr bool isSortedArrayView< SortedArrayView< T, INDEX_TYPE > > = true;

} // namespace LvArray

#endif /* SRC_COMMON_SORTEDARRAYVIEW */
