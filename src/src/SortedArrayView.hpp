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

#include "ChaiVector.hpp"
#include "sortedArrayManipulation.hpp"

#ifdef USE_ARRAY_BOUNDS_CHECK

#undef CONSTEXPRFUNC
#define CONSTEXPRFUNC

#define SORTEDARRAY_CHECK_BOUNDS( index ) \
  GEOS_ERROR_IF( index < 0 || index >= size(), \
                 "Array Bounds Check Failed: index=" << index << " size()=" << size())

#else // USE_ARRAY_BOUNDS_CHECK

#define SORTEDARRAY_CHECK_BOUNDS( index )

#endif // USE_ARRAY_BOUNDS_CHECK

namespace LvArray
{

/**
 * @class SortedArrayView
 * @brief This class provides a view into a SortedArray.
 * @tparam T type of data that is contained by the array.
 * @tparam INDEX_TYPE the integer to use for indexing the components of the array.
 *
 * When using a SortedArrayView directly the template parameter T should be const
 * since the View has no way of modifying the values. This also prevents unnecessary
 * memory movement.
 */
template <class T, class INDEX_TYPE=std::ptrdiff_t>
class SortedArrayView
#ifdef USE_CHAI
  : public chai::CHAICopyable
#endif
{
public:

  /**
   * @brief Default copy constructor. Performs a shallow copy and calls the
   *        chai::ManagedArray copy constructor.
   * @param [in] src the SortedArray to copy.
   */
  inline
  SortedArrayView( SortedArrayView const & src ) = default;

  /**
   * @brief Default move constructor, performs a shallow copy.
   * @param [in/out] src the SortedArray to be moved from.
   */
  inline
  SortedArrayView( SortedArrayView && src ) = default;

  /**
   * @brief Default copy assignment operator, this does a shallow copy.
   * @param [in] src the SortedArray to copy.
   */
  inline
  SortedArrayView & operator=( SortedArrayView const & src ) = default;

  /**
   * @brief Default move assignment operator, this does a shallow copy.
   * @param [in/out] src the SortedArray to be moved from.
   */
  inline
  SortedArrayView & operator=( SortedArrayView && src ) = default;

  /**
   * @brief Return a pointer to the values.
   *
   * @note The pointer is of type T const * because it would be unsafe to modify
   *        the values of the set.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  T * values() const
  { return m_values.data(); }

  /**
   * @brief Access the given value of the set.
   * @param [in] i the index of the value to access.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  T & operator[]( INDEX_TYPE const i ) const
  {
    SORTEDARRAY_CHECK_BOUNDS( i );
    return m_values[i];
  }

  /**
   * @brief Return a pointer to the beginning of the array.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  T * begin() const
  { return values(); }

  /**
   * @brief Return a pointer to the end of the array.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  T * end() const
  { return values() + size(); }

  /**
   * @brief Return true if the array holds no values.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  bool empty() const
  { return m_values.empty(); }

  /**
   * @brief Return the number of values in the array.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  INDEX_TYPE size() const
  { return static_cast<INDEX_TYPE>(m_values.size()); }

  /**
   * @brief Return true if the given value is in the array.
   * @param [in] value the value to search for.
   */
  LVARRAY_HOST_DEVICE inline
  bool contains( T const & value ) const
  { return sortedArrayManipulation::contains( values(), size(), value ); }

  /**
   * @brief Return true if the given value is in the array.
   * @param [in] value the value to search for.
   *
   * @note the is a alias for contains to conform to the std::set interface.
   */
  LVARRAY_HOST_DEVICE inline
  bool count( const T& value ) const
  { return contains( value ); }

protected:

  /**
   * @brief Default constructor. Made protected since every SortedArrayView should
   *        either be the base of a SortedArrayView or copied from another SortedArrayView.
   */
  inline
  SortedArrayView() = default;

  // Holds the array of values.
  ChaiVector<T> m_values;
};

} // namespace LvArray

#endif /* SRC_COMMON_SORTEDARRAYVIEW */