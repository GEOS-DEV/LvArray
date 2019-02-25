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
 * @file CSArray2DView.hpp
 */

#ifndef CSARRAY2DVIEW_HPP_
#define CSARRAY2DVIEW_HPP_

#include "CXX_UtilsConfig.hpp"
#include "ChaiVector.hpp"

#ifdef USE_ARRAY_BOUNDS_CHECK

#undef CONSTEXPRFUNC
#define CONSTEXPRFUNC

#define CSARRAY2D_CHECK_BOUNDS( index0 ) \
  GEOS_ERROR_IF( index0 < 0 || index0 >= size(), \
                 "Array Bounds Check Failed: index=" << index0 << " size()=" << size())

#define CSARRAY2D_CHECK_BOUNDS2( index0, index1 ) \
  GEOS_ERROR_IF( index0 < 0 || index0 >= size() || \
                 index1 < 0 || index1 >= size( index0 ), \
                 "Array Bounds Check Failed: index0=" << index0 << " size()=" << size() \
                                                      << " index1=" << index1 << " size(index0)=" << size( index0 ))

#else // USE_ARRAY_BOUNDS_CHECK

#define CSARRAY2D_CHECK_BOUNDS( index )
#define CSARRAY2D_CHECK_BOUNDS2( index0, index1 )

#endif // USE_ARRAY_BOUNDS_CHECK


namespace LvArray
{

/**
 * @class CSArray2DView
 * @brief This class provides an interface similar to an ArrayView<<ArrayView<T, 1, INDEX_TYPE>, 1, INDEX_TYPE>.
 * @tparam T type of data that is contained by the array.
 * @tparam INDEX_TYPE the integer to use for indexing the components of the array.
 *
 * The difference between this class and ArrayView<<ArrayView<T, 1, INDEX_TYPE>, 1, INDEX_TYPE>
 * is that all of the inner arrays are contiguous in memory. Furthermore the capacity of
 * each inner array is always equal to its size.
 *
 * @note When INDEX_TYPE is const m_offsets is not copied back from the device. INDEX_TYPE should always be const
 * since SortedArrayView is not allowed to modify the offsets.
 * @note When T is const and INDEX_TYPE is const you cannot modify the data at all and neither m_values nor m_offsets
 * is copied back from the device.
 */
template <class T, class INDEX_TYPE=std::ptrdiff_t>
class CSArray2DView
#ifdef USE_CHAI
  : public chai::CHAICopyable
#endif
{
public:

  using INDEX_TYPE_NC = std::remove_const_t<INDEX_TYPE>;

  /**
   * @brief Default copy constructor. Performs a shallow copy and calls the
   * chai::ManagedArray copy constructor.
   * @param [in] src the CSArray2DView to be copied.
   */
  inline
  CSArray2DView( CSArray2DView const & src ) = default;

  /**
   * @brief Default move constructor.
   * @param [in/out] src the CSArray2DView to be moved from.
   */
  inline
  CSArray2DView( CSArray2DView && src ) = default;

  /**
   * @brief User defined conversion to move from T to T const.
   */
  template <class U=T>
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  operator typename std::enable_if<!std::is_const<U>::value, CSArray2DView<T const, INDEX_TYPE_NC const> const &>::type
    () const restrict_this
  { return reinterpret_cast<CSArray2DView<T const, INDEX_TYPE_NC const> const &>(*this); }

  /**
   * @brief Method to convert T to T const. Use this method when the above UDC
   * isn't invoked, this usually occurs with template argument deduction.
   */
  LVARRAY_HOST_DEVICE inline
  CSArray2DView<T const, INDEX_TYPE_NC const> const & toViewC() const restrict_this
  { return *this; }

  /**
   * @brief Default copy assignment operator, this does a shallow copy.
   * @param [in] src the CSArray2DView to be copied from.
   */
  inline
  CSArray2DView & operator=( CSArray2DView const & src ) = default;

  /**
   * @brief Default move assignment operator, this does a shallow copy.
   * @param [in/out] src the CSArray2DView to be moved from.
   */
  inline
  CSArray2DView & operator=( CSArray2DView && src ) = default;

  /**
   * @brief Return the number of arrays.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  INDEX_TYPE_NC size() const restrict_this
  { return m_offsets.size() - 1; }

  /**
   * @brief Return the size of the given array.
   * @param [in] i the array to query.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  INDEX_TYPE_NC size( INDEX_TYPE_NC const i ) const restrict_this
  {
    CSARRAY2D_CHECK_BOUNDS( i );
    return m_offsets[i + 1] - m_offsets[i];
  }

  /**
   * @brief Return true iff the number of arrays is zero.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  bool empty() const restrict_this
  { return size() == 0; }

  /**
   * @brief Return true the size of the given array is zero.
   * @param [in] i the array to query.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  bool empty( INDEX_TYPE_NC const i ) const restrict_this
  { return size( i ) == 0; }

  /**
   * @brief Return a reference to the given value of the given array.
   * @param [in] i the array to access.
   * @param [in] j the index of the value to access.
   *
   * @note This method has bounds checking on both indices when
   * USE_ARRAY_BOUNDS_CHECK is defined.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  T & operator()( INDEX_TYPE_NC const i, INDEX_TYPE_NC const j ) const restrict_this
  {
    CSARRAY2D_CHECK_BOUNDS2( i, j );
    return m_values[m_offsets[i] + j];
  }

  /**
   * @brief Return a pointer to the values of the given array.
   * @param [in] i the array to access.
   *
   * @note This method has bounds checking on the first index when
   * USE_ARRAY_BOUNDS_CHECK is defined, however since it returns a bare pointer
   * there is no bounds checking on subsequent dereferences.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  T * operator[]( INDEX_TYPE_NC const i ) const restrict_this
  {
    CSARRAY2D_CHECK_BOUNDS( i );
    return &m_values[m_offsets[i]];
  }

  /**
   * @brief Set the values of the given array.
   * @param [in] i the array to access.
   * @param [in] values the new values of the given array. Must be of length at
   * least size(i).
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  void setArray( INDEX_TYPE_NC const i, T const * const values ) const restrict_this
  {
    CSARRAY2D_CHECK_BOUNDS( i );
    INDEX_TYPE_NC const offset = m_offsets[i];
    INDEX_TYPE_NC const n_values = size( i );
    for( INDEX_TYPE_NC j = 0 ; j < n_values ; ++j )
    {
      m_values[offset + j] = values[j];
    }
  }

  /**
   * @brief Return a pointer to the offsets array, of length size() + 1.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  INDEX_TYPE_NC const * getOffsets() const restrict_this
  { return m_offsets.data(); }

  /**
   * @brief Return a pointer to the values array, of length getOffsets()[size()].
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  T * getValues() const restrict_this
  { return m_values.data(); }

protected:

  /**
   * @brief Default constructor. Made protected since every CSArray2DView should
   * either be the base of a CSArray2D or copied from another CSArray2DView.
   */
  CSArray2DView() = default;

  // Holds the offset of each array, of length size() + 1. Array i begins at
  // m_offsets[i] and has size m_offsets[i+1] - m_offsets[i].
  ChaiVector<INDEX_TYPE> m_offsets;

  // Holds the values of each array, of length m_offsets[size()].
  ChaiVector<T> m_values;
};

} /* namespace LvArray */

#endif /* CSARRAY2DVIEW_HPP_ */
