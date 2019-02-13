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
 * @file CSArray2D.hpp
 */

#ifndef CSARRAY2D_HPP_
#define CSARRAY2D_HPP_

#include "ChaiVector.hpp"
#include "CSArray2DView.hpp"

namespace LvArray
{

/**
 * @class CSArray2D
 * @brief This class provides an interface similar to an Array<<Array<T, 1, INDEX_TYPE>, 1, INDEX_TYPE>.
 * @tparam T type of data that is contained by the array.
 * @tparam INDEX_TYPE the integer to use for indexing the components of the array.
 *
 * The difference between this class and Array<<Array<T, 1, INDEX_TYPE>, 1, INDEX_TYPE>
 * is that all of the inner arrays are contiguous in memory. Furthermore the capacity of
 * each inner array is always equal to its size.
 *
 * @note CSArray2DView is a protected base class of CSArray2D. This is to control
 * the conversion to CSArray2DView so that when using a View the INDEX_TYPE is always const.
 * However the CSArray2DView interface is reproduced here.
 */
template <class T, class INDEX_TYPE=std::ptrdiff_t>
class CSArray2D : protected CSArray2DView<T, INDEX_TYPE>
{
public:

  // Aliasing public methods of CSArray2DView.
  using CSArray2DView<T, INDEX_TYPE>::toViewC;
  using CSArray2DView<T, INDEX_TYPE>::size;
  using CSArray2DView<T, INDEX_TYPE>::empty;
  using CSArray2DView<T, INDEX_TYPE>::operator();
  using CSArray2DView<T, INDEX_TYPE>::operator[];
  using CSArray2DView<T, INDEX_TYPE>::setArray;
  using CSArray2DView<T, INDEX_TYPE>::getOffsets;
  using CSArray2DView<T, INDEX_TYPE>::getValues;

  /**
   * @brief Constructor.
   * @param [in] numArrays initial number of arrays.
   * @param [in] valueCapacity the initial value capacity.
   */
  inline
  CSArray2D(INDEX_TYPE const numArrays=0, INDEX_TYPE const valueCapacity=0) :
    CSArray2DView<T, INDEX_TYPE>()
  {
    resizeNumArrays(numArrays);
    reserveValues(valueCapacity);
    m_offsets[0] = 0;
  }

  /**
   * @brief Copy constructor, performs a deep copy.
   * @param [in] src the CSArray2D to copy.
   */
  inline
  CSArray2D(CSArray2D const & src) :
    CSArray2DView<T, INDEX_TYPE>()
  { *this = src; }

  /**
   * @brief Default move constructor, performs a shallow copy.
   * @param [in/out] src the CSArray2D to be moved from.
   */
  inline
  CSArray2D(CSArray2D && src) = default;

  /**
   * @brief Destructor, frees the offsets and values ChaiVectors.
   */
  inline
  ~CSArray2D() restrict_this
  {
    m_offsets.free();
    m_values.free();
  }

  /**
   * @brief Conversion operator to CSArray2DView<T, INDEX_TYPE const>.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  operator CSArray2DView<T, INDEX_TYPE const> const &
  () const restrict_this
  { return reinterpret_cast<CSArray2DView<T, INDEX_TYPE const> const &>(*this); }

  /**
   * @brief Method to convert to CSArray2DView<T, INDEX_TYPE const>.Use this method when
   * the above UDC isn't invoked, this usually occurs with template argument deduction.
   */
  LVARRAY_HOST_DEVICE inline
  CSArray2DView<T, INDEX_TYPE const> const & toView() const restrict_this
  { return *this; }

  /**
   * @brief Conversion operator to CSArray2DView<T const, INDEX_TYPE const>.
   * @note CSArray2DView defines this operator but for some reason NVCC won't find it
   * if we alias it with a using statement.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  operator CSArray2DView<T const, INDEX_TYPE const> const &
  () const restrict_this
  { return toViewC(); }

  /**
   * @brief Copy assignment operator, performs a deep copy.
   * @param [in] src the CSArray2D to copy.
   */
  inline
  CSArray2D & operator=(CSArray2D const & src) restrict_this
  {
    src.m_offsets.copy_into(m_offsets);
    src.m_values.copy_into(m_values);
    return *this;
  }

  /**
   * @brief Default move assignment operator, performs a shallow copy.
   * @param [in] src the CSArray2D to be moved from.
   */ 
  inline
  CSArray2D & operator=(CSArray2D && src) = default;

#ifdef USE_CHAI
  /**
   * @brief Moves the CSArray2D to the given execution space.
   * @param [in] space the space to move to.
   */ 
  inline
  void move( chai::ExecutionSpace const space ) restrict_this
  {
    m_offsets.move(space);
    m_values.move(space);
  }
#endif

  /**
   * @brief Resize the number of arrays.
   * @param [in] numArrays the new number of Arrays.
   */ 
  inline
  void resizeNumArrays(INDEX_TYPE const numArrays) restrict_this
  {
    GEOS_ERROR_IF(numArrays < 0, "Invalid number of arrays.");
    m_offsets.resize(numArrays + 1, 0);
  }

  /**
   * @brief Reserve space for the given number of arrays.
   * @param [in] numArrays the number of arrays to reserve space for.
   */
  inline
  void reserveNumArrays(INDEX_TYPE const numArrays) restrict_this
  {
    GEOS_ERROR_IF(numArrays < 0, "Invalid arrays capacity.");
    m_offsets.reserve(numArrays + 1);
  }

  /**
   * @brief Reserve space for the given number of values.
   * @param [in] numValues the number of values to reserve space for.
   */
  inline
  void reserveValues(INDEX_TYPE const numValues) restrict_this
  {
    GEOS_ERROR_IF(numValues < 0, "Invalid value capacity.");
    m_values.reserve(numValues);
  }

  /**
   * @brief Resize the given array.
   * @param [in] i the array to resize.
   * @param [in] arraySize the new size.
   * @param [in] defaultValue the value to assign to any new entries.
   */
  inline
  void resizeArray(INDEX_TYPE const i, INDEX_TYPE const arraySize, T const & defaultValue = T()) restrict_this
  {
    CSARRAY2D_CHECK_BOUNDS(i);
    GEOS_ASSERT(arraySize >= 0);
    
    INDEX_TYPE const offset = m_offsets[i];
    INDEX_TYPE const previousSize = size(i);
    INDEX_TYPE const sizeDifference = arraySize - previousSize;
    if (sizeDifference == 0) return;

    // Update the offsets of all subsequent arrays.
    INDEX_TYPE const nArrays = size();
    for (INDEX_TYPE j = i + 1; j < nArrays + 1; ++j)
    {
      m_offsets[j] += sizeDifference;
    }

    // Insert (or remove) the appropriate number of values at the end of the given array.
    if ( sizeDifference > 0 )
    {
      m_values.emplace(offset + previousSize, sizeDifference, defaultValue);
    }
    else
    {
      m_values.erase(offset + previousSize + sizeDifference, -sizeDifference);
    }
  }

  /**
   * @brief Append the given values as a new array.
   * @param [in] values the values of the new array.
   * @param [in] numValues the number of values.
   */
  inline
  void appendArray(T const * const values, INDEX_TYPE const numValues) restrict_this
  {
    GEOS_ASSERT(values != nullptr);
    GEOS_ASSERT(numValues >= 0);

    m_values.push_back(values, numValues);
    m_offsets.push_back(m_values.size());
  }

  /**
   * @brief Insert the given values as a new array.
   * @param [in] i the position to insert the new array.
   * @param [in] values the values of the new array.
   * @param [in] numValues the number of values.
   */
  inline
  void insertArray(INDEX_TYPE const i, T const * const values, INDEX_TYPE const n ) restrict_this
  {
    GEOS_ASSERT(i >= 0 && i <= size());
    GEOS_ASSERT(n == 0 || (n > 0 && values != nullptr));

    // Insert the new offset.
    INDEX_TYPE previous_offset = m_offsets[i];
    m_offsets.insert(i + 1, previous_offset + n);

    // Update the subsequent offsets.
    INDEX_TYPE const nArrays = size();
    for (INDEX_TYPE j = i + 2; j < nArrays + 1; ++j)
    {
      m_offsets[j] += n;
    }

    // Insert the new values.
    m_values.insert(previous_offset, values, n);
  }

  /**
   * @brief Remove the given array.
   * @param [in] i the position of the array to remove.
   */
  inline
  void removeArray(INDEX_TYPE const i) restrict_this
  {
    CSARRAY2D_CHECK_BOUNDS(i);
    INDEX_TYPE const arrayOffset = m_offsets[i];
    INDEX_TYPE const arraySize = size(i);

    // Update the subsequent offsets.
    for (INDEX_TYPE j = i + 1; j < size(); ++j)
    {
      m_offsets[j] = m_offsets[j + 1] - arraySize;
    }

    // Remove the last value.
    m_offsets.pop_back();

    // Erase the values.
    if (arraySize != 0) m_values.erase(arrayOffset, arraySize);
  }

private:

  // Aliasing protected members of CSArray2DView used here.
  using CSArray2DView<T, INDEX_TYPE>::m_offsets;
  using CSArray2DView<T, INDEX_TYPE>::m_values;
};

} /* namespace LvArray */

#endif /* CSARRAY2D_HPP_ */
