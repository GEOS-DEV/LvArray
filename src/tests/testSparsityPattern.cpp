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

#include "SparsityPattern.hpp"
#include "testUtils.hpp"
#include "IntegerConversion.hpp"

#ifdef USE_CUDA

#include "Array.hpp"

#ifdef USE_CHAI
#include "chai/util/forall.hpp"
#endif

#endif

#include "gtest/gtest.h"

#include <vector>
#include <set>
#include <iterator>
#include <random>
#include <climits>

namespace LvArray
{

using uint = unsigned int;
using schar = signed char;

// Alias for SparsityPatternView
template <class COL_TYPE, class INDEX_TYPE=std::ptrdiff_t>
using ViewType = SparsityPatternView<COL_TYPE, INDEX_TYPE const>;

using INDEX_TYPE = std::ptrdiff_t;

// A row is a sorted collection of unique columns, which is a set.
template <class COL_TYPE>
using ROW_REF_TYPE = std::set<COL_TYPE>;

// A sparsity pattern is an array of rows.
template <class COL_TYPE>
using REF_TYPE = std::vector<ROW_REF_TYPE<COL_TYPE>>;

namespace internal
{

// A static random number generator is used so that subsequent calls to the same test method will
// do different things. This initializes the generator with the default seed so the test behavior will
// be uniform across runs and platforms. However the behavior of an individual method will depend on the state
// of the generator and hence on the calls before it.
static std::mt19937_64 gen;

/**
 * @brief Check that the SparsityPatternView is equivalent to the reference type.
 * @param [in] v the SparsityPatternView to check.
 * @param [in] vRef the reference to check against.
 */
template <class COL_TYPE>
void compareToReference(ViewType<COL_TYPE const> const & v, REF_TYPE<COL_TYPE> const & vRef)
{
  INDEX_TYPE const numRows = v.numRows();
  ASSERT_EQ(numRows, integer_conversion<INDEX_TYPE>(vRef.size()));

  INDEX_TYPE ref_nnz = 0;
  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    INDEX_TYPE const rowNNZ = v.numNonZeros(row);
    INDEX_TYPE const refRowNNZ = vRef[row].size();
    ref_nnz += refRowNNZ; 

    ASSERT_EQ(rowNNZ, refRowNNZ);

    if (rowNNZ == 0)
    {
      ASSERT_TRUE(v.empty(row));
    }
    else
    {
      ASSERT_FALSE(v.empty(row));
    }

    typename ROW_REF_TYPE<COL_TYPE>::iterator it = vRef[row].begin();
    COL_TYPE const * const columns = v.getColumns(row);
    for (INDEX_TYPE i = 0; i < v.numNonZeros(row); ++i)
    {
      EXPECT_FALSE(v.empty(row, columns[i]));
      EXPECT_EQ(columns[i], *it);
      it++;
    }
  }

  ASSERT_EQ(v.numNonZeros(), ref_nnz);

  if (v.numNonZeros() == 0)
  {
    ASSERT_TRUE(v.empty());
  }
}

/**
 * @brief Test the insert method of the SparsityPattern.
 * @param [in/out] v the SparsityPattern to test.
 * @param [in/out] vRef the reference to check against.
 * @param [in] MAX_INSERTS the number of times to call insert.
 */
template <class COL_TYPE>
void insertTest(SparsityPattern<COL_TYPE> & v, REF_TYPE<COL_TYPE> & vRef,
                INDEX_TYPE const MAX_INSERTS)
{
  INDEX_TYPE const numRows = v.numRows();
  INDEX_TYPE const numCols = v.numColumns();
  ASSERT_EQ(numRows, integer_conversion<INDEX_TYPE>(vRef.size()));

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    ROW_REF_TYPE<COL_TYPE> & refRow = vRef[row];
    ASSERT_EQ(v.numNonZeros(row), integer_conversion<INDEX_TYPE>(refRow.size()));

    for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
    {
      COL_TYPE const col = columnDist(gen);
      ASSERT_EQ(refRow.insert(col).second, v.insertNonZero(row, col));
    }
  }

  compareToReference(v.toViewC(), vRef);
}

/**
 * @brief Test the insert multiple method of the SparsityPattern.
 * @param [in/out] v the SparsityPattern to test.
 * @param [in/out] vRef the reference to check against.
 * @param [in] MAX_INSERTS the number of values to insert at a time.
 */
template <class COL_TYPE>
void insertMultipleTest(SparsityPattern<COL_TYPE> & v, REF_TYPE<COL_TYPE> & vRef,
                        INDEX_TYPE const MAX_INSERTS)
{
  INDEX_TYPE const numRows = v.numRows();
  INDEX_TYPE const numCols = v.numColumns();
  ASSERT_EQ(numRows, integer_conversion<INDEX_TYPE>(vRef.size()));

  std::vector<COL_TYPE> columns(MAX_INSERTS);

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    ROW_REF_TYPE<COL_TYPE> & refRow = vRef[row];
    ASSERT_EQ(v.numNonZeros(row), integer_conversion<COL_TYPE>(refRow.size()));

    for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
    {
      columns[i] = columnDist(gen);
    }

    v.insertNonZeros(row, columns.data(), MAX_INSERTS);
    refRow.insert(columns.begin(), columns.end());

    ASSERT_EQ(v.numNonZeros(row), integer_conversion<COL_TYPE>(refRow.size()));
  }

  compareToReference(v.toViewC(), vRef);
}

/**
 * @brief Test the insertSorted method of the SparsityPattern.
 * @param [in/out] v the SparsityPattern to test.
 * @param [in/out] vRef the reference to check against.
 * @param [in] MAX_INSERTS the number of values to insert at a time.
 */
template <class COL_TYPE>
void insertSortedTest(SparsityPattern<COL_TYPE> & v, REF_TYPE<COL_TYPE> & vRef, INDEX_TYPE const MAX_INSERTS)
{
  INDEX_TYPE const numRows = v.numRows();
  INDEX_TYPE const numCols = v.numColumns();
  ASSERT_EQ(numRows, integer_conversion<INDEX_TYPE>(vRef.size()));

  std::vector<COL_TYPE> columns(MAX_INSERTS);

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    ROW_REF_TYPE<COL_TYPE> & refRow = vRef[row];
    ASSERT_EQ(v.numNonZeros(row), integer_conversion<COL_TYPE>(refRow.size()));

    for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
    {
      columns[i] = columnDist(gen);
    }

    std::sort(columns.begin(), columns.end());
    v.insertNonZerosSorted(row, columns.data(), MAX_INSERTS);
    refRow.insert(columns.begin(), columns.end());

    ASSERT_EQ(v.numNonZeros(row), integer_conversion<COL_TYPE>(refRow.size()));
  }

  compareToReference(v.toViewC(), vRef);
}

/**
 * @brief Test the remove method of the SparsityPatternView.
 * @param [in/out] v the SparsityPatternView to test.
 * @param [in/out] vRef the reference to check against.
 * @param [in] MAX_REMOVES the number of times to call remove.
 */
template <class COL_TYPE>
void removeTest(ViewType<COL_TYPE> const & v, REF_TYPE<COL_TYPE> & vRef,
                INDEX_TYPE const MAX_REMOVES)
{
  INDEX_TYPE const numRows = v.numRows();
  INDEX_TYPE const numCols = v.numColumns();
  ASSERT_EQ(numRows, integer_conversion<INDEX_TYPE>(vRef.size()));

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    ROW_REF_TYPE<COL_TYPE> & refRow = vRef[row];
    ASSERT_EQ(v.numNonZeros(row), integer_conversion<COL_TYPE>(refRow.size()));
    
    for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
    {
      COL_TYPE const col = columnDist(gen);
      ASSERT_EQ(v.removeNonZero(row, col), refRow.erase(col));
    }
  }

  compareToReference(v.toViewC(), vRef);
}
  
/**
 * @brief Test the remove multiple method of the SparsityPatternView.
 * @param [in/out] v the SparsityPatternView to test.
 * @param [in/out] vRef the reference to check against.
 * @param [in] MAX_REMOVES the number of values to remove at a time.
 */
template <class COL_TYPE>
void removeMultipleTest(ViewType<COL_TYPE> const & v, REF_TYPE<COL_TYPE> & vRef,
                        INDEX_TYPE const MAX_REMOVES)
{
  INDEX_TYPE const numRows = v.numRows();
  INDEX_TYPE const numCols = v.numColumns();
  ASSERT_EQ(numRows, integer_conversion<INDEX_TYPE>(vRef.size()));

  std::vector<COL_TYPE> columns(MAX_REMOVES);

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    ROW_REF_TYPE<COL_TYPE> & refRow = vRef[row];
    ASSERT_EQ(v.numNonZeros(row), integer_conversion<COL_TYPE>(integer_conversion<COL_TYPE>(refRow.size())));

    for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
    {
      columns[i] = columnDist(gen);
    }

    v.removeNonZeros(row, columns.data(), MAX_REMOVES);
    for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
    {
      refRow.erase(columns[i]);
    }

    ASSERT_EQ(v.numNonZeros(row), integer_conversion<COL_TYPE>(integer_conversion<COL_TYPE>(refRow.size())));
  }

  compareToReference(v.toViewC(), vRef);
}

/**
 * @brief Test the removeSorted method of the SparsityPatternView.
 * @param [in/out] v the SparsityPatternView to test.
 * @param [in/out] vRef the reference to check against.
 * @param [in] MAX_REMOVES the number of values to remove at a time.
 */
template <class COL_TYPE>
void removeSortedTest(ViewType<COL_TYPE> const & v, REF_TYPE<COL_TYPE> & vRef, INDEX_TYPE const MAX_REMOVES)
{
  INDEX_TYPE const numRows = v.numRows();
  INDEX_TYPE const numCols = v.numColumns();
  ASSERT_EQ(numRows, integer_conversion<INDEX_TYPE>(vRef.size()));

  std::vector<COL_TYPE> columns(MAX_REMOVES);

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    ROW_REF_TYPE<COL_TYPE> & refRow = vRef[row];
    ASSERT_EQ(v.numNonZeros(row), integer_conversion<COL_TYPE>(integer_conversion<COL_TYPE>(refRow.size())));

    for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
    {
      columns[i] = columnDist(gen);
    }

    std::sort(columns.begin(), columns.end());
    v.removeNonZeros(row, columns.data(), MAX_REMOVES);
    for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
    {
      refRow.erase(columns[i]);
    }

    ASSERT_EQ(v.numNonZeros(row), integer_conversion<COL_TYPE>(integer_conversion<COL_TYPE>(refRow.size())));
  }

  compareToReference(v.toViewC(), vRef);
}

/**
 * @brief Test the empty method of the SparsityPatternView.
 * @param [in] v the SparsityPatternView to test.
 * @param [in] vRef the reference to check against.
 */
template <class COL_TYPE>
void emptyTest(ViewType<COL_TYPE const> const & v, REF_TYPE<COL_TYPE> const & vRef)
{
  INDEX_TYPE const numRows = v.numRows();
  ASSERT_EQ(numRows, vRef.size());

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    ROW_REF_TYPE<COL_TYPE> const & refRow = vRef[row];
    INDEX_TYPE const nnz = v.numNonZeros(row);
    ASSERT_EQ(nnz, refRow.size());

    typename ROW_REF_TYPE<COL_TYPE>::iterator it = refRow.begin();
    for (INDEX_TYPE i = 0; i < nnz; ++i)
    {
      COL_TYPE const col = *it;
      EXPECT_FALSE(v.empty(row, col));
      EXPECT_NE(v.empty(row, COL_TYPE(i)), refRow.count(COL_TYPE(i)));
      ++it;
    }
  }
}

/**
 * @brief Fill a row of the SparsityPatternView up to its capacity.
 * @param [in/out] v the SparsityPatternView.
 * @param [in/out] refRow the reference row to compare against.
 * @param [in] row the row of the SparsityPatternView to fill.
 */
template <class COL_TYPE>
void fillRow(ViewType<COL_TYPE> const & v, ROW_REF_TYPE<COL_TYPE> & refRow, INDEX_TYPE const row)
{
  INDEX_TYPE nToInsert = v.nonZeroCapacity(row) - v.numNonZeros(row);
  COL_TYPE testColum = COL_TYPE(v.numColumns() - 1);
  COL_TYPE const * const columns = v.getColumns(row);

  ASSERT_GE(v.numColumns(), v.nonZeroCapacity(row));

  while (nToInsert > 0)
  {
    bool const success = refRow.insert(testColum).second;
    ASSERT_EQ(success, v.insertNonZero(row, testColum));
    nToInsert -= success;
    testColum -= 1;
  }

  COL_TYPE const * const newColumns = v.getColumns(row);
  ASSERT_EQ(columns, newColumns);
}

/**
 * @brief Test the setRowCapacity method of SparsityPattern.
 * @param [in/out] v the SparsityPattern to test.
 * @param [in/out] vRef the reference to compare against.
 */
template <class COL_TYPE>
void rowCapacityTest(SparsityPattern<COL_TYPE> & v, REF_TYPE<COL_TYPE> & vRef)
{
  INDEX_TYPE const numRows = v.numRows();
  ASSERT_EQ(numRows, integer_conversion<INDEX_TYPE>(vRef.size()));

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    ROW_REF_TYPE<COL_TYPE> & refRow = vRef[row];
    INDEX_TYPE const rowNNZ = v.numNonZeros(row);
    ASSERT_EQ(rowNNZ, integer_conversion<COL_TYPE>(refRow.size()));

    // The new capacity will be the index of the row.
    INDEX_TYPE const new_capacity = row;
    if (new_capacity < rowNNZ)
    {
      COL_TYPE const * const columns = v.getColumns(row);
      v.setRowCapacity(row, new_capacity);

      // Erase the last values from the reference.
      INDEX_TYPE const difference = rowNNZ - new_capacity;
      typename ROW_REF_TYPE<COL_TYPE>::iterator erase_pos = refRow.end();
      std::advance(erase_pos, -difference);
      refRow.erase(erase_pos, refRow.end());
      
      // Check that the pointer to the row hasn't changed.
      COL_TYPE const * const newColumns = v.getColumns(row);
      EXPECT_EQ(columns, newColumns);
    }
    else
    {
      v.setRowCapacity(row, new_capacity);

      if (new_capacity > v.numColumns())
      {
        EXPECT_EQ(v.nonZeroCapacity(row), v.numColumns());
      }
      else
      {
        EXPECT_EQ(v.nonZeroCapacity(row), new_capacity);
      }
      
      fillRow(v.toView(), refRow, row);
    }
  }

  compareToReference(v.toViewC(), vRef);
}

/**
 * @brief Test the compress method of SparsityPattern.
 * @param [in/out] v the SparsityPattern to test.
 * @param [in] vRef the reference to compare against.
 */
template <class COL_TYPE>
void compressTest(SparsityPattern<COL_TYPE> & v, REF_TYPE<COL_TYPE> const & vRef)
{
  v.compress();

  COL_TYPE const * const columns = v.getColumns( 0 );
  INDEX_TYPE const * const offsets = v.getOffsets();

  INDEX_TYPE curOffset = 0;
  for (INDEX_TYPE row = 0; row < v.numRows(); ++row)
  {
    // The last row will have all the extra capacity.
    if (row != v.numRows() - 1)
    {
      ASSERT_EQ(v.numNonZeros(row), v.nonZeroCapacity(row));
    }
    
    COL_TYPE const * const rowColumns = v.getColumns(row);
    ASSERT_EQ(rowColumns, columns + curOffset);
    ASSERT_EQ(offsets[row], curOffset);

    curOffset += v.numNonZeros(row);
  }

  compareToReference(v.toViewC(), vRef);
}

/**
 * @brief Test the copy constructor of the SparsityPattern.
 * @param [in] v the SparsityPattern to check.
 * @param [in] vRef the reference to check against.
 */
template <class COL_TYPE>
void deepCopyTest(SparsityPattern<COL_TYPE> const & v, REF_TYPE<COL_TYPE> const & vRef)
{
  SparsityPattern<COL_TYPE> copy(v);

  ASSERT_EQ(v.numRows(), copy.numRows());
  ASSERT_EQ(v.numColumns(), copy.numColumns());
  ASSERT_EQ(v.numNonZeros(), copy.numNonZeros());

  INDEX_TYPE const totalNNZ = v.numNonZeros();

  for (INDEX_TYPE row = 0; row < v.numRows(); ++row)
  {
    INDEX_TYPE const nnz = v.numNonZeros(row);
    ASSERT_EQ(nnz, copy.numNonZeros(row));

    COL_TYPE const * const cols = v.getColumns(row);
    COL_TYPE const * const cols_cpy = copy.getColumns(row);
    ASSERT_NE(cols, cols_cpy);

    // Iterate backwards and remove entries from copy.
    for (INDEX_TYPE i = nnz - 1; i >= 0; --i)
    {
      EXPECT_EQ(cols[i], cols_cpy[i]);
      copy.removeNonZero(row, cols_cpy[i]);
    }

    EXPECT_EQ(copy.numNonZeros(row), 0);
    EXPECT_EQ(v.numNonZeros(row), nnz);
  }

  EXPECT_EQ(copy.numNonZeros(), 0);
  EXPECT_EQ(v.numNonZeros(), totalNNZ);

  compareToReference(v.toViewC(), vRef);
}

/**
 * @brief Test the copy constructor of the SparsityPatternView.
 * @param [in/out] v the SparsityPattern to check.
 */
template <class COL_TYPE>
void shallowCopyTest(SparsityPattern<COL_TYPE> const & v)
{
  ViewType<COL_TYPE> const & view = v;
  ViewType<COL_TYPE> copy(view);

  ASSERT_EQ(v.numRows(), copy.numRows());
  ASSERT_EQ(v.numColumns(), copy.numColumns());
  ASSERT_EQ(v.numNonZeros(), copy.numNonZeros());

  for (INDEX_TYPE row = 0; row < v.numRows(); ++row)
  {
    INDEX_TYPE const nnz = v.numNonZeros(row);
    ASSERT_EQ(nnz, copy.numNonZeros(row));

    COL_TYPE const * const cols = v.getColumns(row);
    COL_TYPE const * const cols_cpy = copy.getColumns(row);
    ASSERT_EQ(cols, cols_cpy);

    // Iterate backwards and remove entries from copy.
    for (INDEX_TYPE i = nnz - 1; i >= 0; --i)
    {
      EXPECT_EQ(cols[i], cols_cpy[i]);
      copy.removeNonZero(row, cols_cpy[i]);
    }

    EXPECT_EQ(copy.numNonZeros(row), 0);
    EXPECT_EQ(v.numNonZeros(row), 0);
  }

  EXPECT_EQ(copy.numNonZeros(), 0);
  EXPECT_EQ(v.numNonZeros(), 0);
}

#ifdef USE_CUDA

/**
 * @brief Test the SparsityPatternView copy constructor in regards to memory motion.
 * @param [in/out] v the SparsityPatternView to test.
 */
template <class COL_TYPE>
void memoryMotionTest(ViewType<COL_TYPE> const & v)
{
  INDEX_TYPE const numRows = v.numRows();

  // Set the columns. The const cast here isn't necessary, we could remove and insert
  // into the SparsityPattern instead, but this is quicker.
  INDEX_TYPE curIndex = 0;
  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    COL_TYPE const * const columns = v.getColumns(row);
    COL_TYPE * const columnsNC = const_cast<COL_TYPE *>(columns);
    for (INDEX_TYPE i = 0; i < v.numNonZeros(row); ++i)
    {
      columnsNC[i] = COL_TYPE(curIndex++);
    }
  }

  // Capture the view on device and set the values. Here the const cast is necessary
  // because we don't want to test the insert/remove methods on device yet.
  forall(gpu(), 0, numRows,
    [=] __device__ (INDEX_TYPE row)
    {
      COL_TYPE const * const columns = v.getColumns(row);
      COL_TYPE * const columnsNC = const_cast<COL_TYPE *>(columns);
      for (INDEX_TYPE i = 0; i < v.numNonZeros(row); ++i)
      {
        columnsNC[i] *= columns[i];
      }
    }
  );

  // Check that the columns have been updated.
  curIndex = 0;
  forall(sequential(), 0, numRows,
    [=, &curIndex](INDEX_TYPE row)
    {
      for (INDEX_TYPE i = 0; i < v.numNonZeros(row); ++i)
      {
        COL_TYPE val = COL_TYPE(curIndex) * COL_TYPE(curIndex);
        EXPECT_EQ(val, v.getColumns(row)[i]);
        ++curIndex;
      }
    }
  );
}

/**
 * @brief Test the SparsityPattern move method.
 * @param [in/out] v the SparsityPattern to test.
 */
template <class COL_TYPE>
void memoryMotionMove(SparsityPattern<COL_TYPE> & v)
{
  INDEX_TYPE const numRows = v.numRows();

  ViewType<COL_TYPE> const & view = v;

  INDEX_TYPE curIndex = 0;
  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    COL_TYPE const * const columns = view.getColumns(row);
    COL_TYPE * const columnsNC = const_cast<COL_TYPE *>(columns);
    for (INDEX_TYPE i = 0; i < view.numNonZeros(row); ++i)
    {
      columnsNC[i] = COL_TYPE(curIndex++);
    }
  }

  forall(gpu(), 0, numRows,
    [=] __device__ (INDEX_TYPE row)
    {
      COL_TYPE const * const columns = view.getColumns(row);
      COL_TYPE * const columnsNC = const_cast<COL_TYPE *>(columns);
      for (INDEX_TYPE i = 0; i < view.numNonZeros(row); ++i)
      {
        columnsNC[i] *= columns[i];
      }
    }
  );

  v.move(chai::CPU);
  curIndex = 0;
  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    for (INDEX_TYPE i = 0; i < view.numNonZeros(row); ++i)
    {
      COL_TYPE val = COL_TYPE(curIndex) * COL_TYPE(curIndex);
      EXPECT_EQ(val, view.getColumns(row)[i]);
      curIndex++;
    }
  }
}

/**
 * @brief Test the const capture semantics.
 * @param [in/out] v the SparsityPattern to test.
 * @param [in/out] vRef the reference to compare to.
 */
template <class COL_TYPE>
void memoryMotionConstTest(SparsityPattern<COL_TYPE> v, REF_TYPE<COL_TYPE> & vRef)
{
  INDEX_TYPE const numRows = v.numRows();
  INDEX_TYPE const numCols = v.numColumns();

  // Create a view const and capture it on device.
  ViewType<COL_TYPE const> const & vConst = v;
  forall(gpu(), 0, numRows,
    [=] __device__ (INDEX_TYPE row)
    {
      COL_TYPE const * const columns = vConst.getColumns(row);
      for (INDEX_TYPE i = 0; i < vConst.numNonZeros(row); ++i)
      {
        GEOS_ERROR_IF(!arrayManipulation::isPositive(columns[i]) || columns[i] >= numCols, "Invalid column.");
      }
    }
  );

  // Fill in the rows on host.
  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    fillRow(v.toView(), vRef[row], row);
  }

  // Move the SparsityPattern back from device, this should be a no-op.
  v.move(chai::CPU);
  compareToReference(v.toViewC(), vRef);
}

/**
 * @brief Test the SparsityPatternView insert method on device.
 * @param [in/out] v the SparsityPattern to test.
 * @param [in/out] vRef the reference to compare to.
 * @param [in] MAX_INSERTS the maximum number of inserts.
 */
template <class COL_TYPE>
void insertDeviceTest(SparsityPattern<COL_TYPE> & v, REF_TYPE<COL_TYPE> & vRef,
                      INDEX_TYPE const MAX_INSERTS)
{
  INDEX_TYPE const numRows = v.numRows();
  INDEX_TYPE const numCols = v.numColumns();

  // Create an Array of the columns to insert into each row.
  Array<COL_TYPE, 2> columnsToInsert(numRows, MAX_INSERTS);

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  // Reserve space in each row for the number of insertions and populate the columnsToInsert Array.
  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    v.reserveNonZeros(row, v.numNonZeros(row) + MAX_INSERTS);

    for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
    {
      columnsToInsert(row, i) = columnDist(gen);
    }
  }

  compareToReference(v.toViewC(), vRef);

  // Create views and insert the columns on the device.
  ViewType<COL_TYPE> const & vView = v;
  ArrayView<COL_TYPE const, 2> const & insertView = columnsToInsert;
  forall(gpu(), 0, numRows,
    [=] __device__ (INDEX_TYPE row)
    {
      for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
      {
        vView.insertNonZero(row, insertView(row, i));
      }
    }
  );

  // Insert the values into the reference.
  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
    {
      vRef[row].insert(columnsToInsert(row, i));
    }
  }

  // Move the SparsityPattern back to the host and compare against the reference.
  v.move(chai::CPU);
  compareToReference(v.toViewC(), vRef);
}

/**
 * @brief Test the SparsityPatternView insert multiple method on device.
 * @param [in/out] v the SparsityPattern to test.
 * @param [in/out] vRef the reference to compare to.
 * @param [in] MAX_INSERTS the maximum number of inserts.
 */
template <class COL_TYPE>
void insertMultipleDeviceTest(SparsityPattern<COL_TYPE> & v, REF_TYPE<COL_TYPE> & vRef,
                              INDEX_TYPE const MAX_INSERTS)
{
  INDEX_TYPE const numRows = v.numRows();
  INDEX_TYPE const numCols = v.numColumns();

  Array<COL_TYPE, 2> columnsToInsert(numRows, MAX_INSERTS);

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    v.reserveNonZeros(row, v.numNonZeros(row) + MAX_INSERTS);

    for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
    {
      columnsToInsert(row, i) = columnDist(gen);
    }
  }

  compareToReference(v.toViewC(), vRef);

  ViewType<COL_TYPE> const & vView = v;
  ArrayView<COL_TYPE const, 2> const & insertView = columnsToInsert;
  forall(gpu(), 0, numRows,
    [=] __device__ (INDEX_TYPE row)
    {
      vView.insertNonZeros(row, insertView[row], MAX_INSERTS);
    }
  );

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    COL_TYPE const * const colPtr = columnsToInsert[row];
    vRef[row].insert(colPtr, colPtr + MAX_INSERTS);
  }

  v.move(chai::CPU);
  compareToReference(v.toViewC(), vRef);
}

/**
 * @brief Test the SparsityPatternView insertSorted method on device.
 * @param [in/out] v the SparsityPattern to test.
 * @param [in/out] vRef the reference to compare to.
 * @param [in] MAX_INSERTS the maximum number of inserts.
 */
template <class COL_TYPE>
void insertSortedDeviceTest(SparsityPattern<COL_TYPE> & v, REF_TYPE<COL_TYPE> & vRef,
                            INDEX_TYPE const MAX_INSERTS)
{
  INDEX_TYPE const numRows = v.numRows();
  INDEX_TYPE const numCols = v.numColumns();

  Array<COL_TYPE, 2> columnsToInsert(numRows, MAX_INSERTS);

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    v.reserveNonZeros(row, v.numNonZeros(row) + MAX_INSERTS);

    for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
    {
      columnsToInsert(row, i) = columnDist(gen);
    }

    // Sort the columns so they're ready to be inserted.
    COL_TYPE * const colPtr = columnsToInsert[row];
    std::sort(colPtr, colPtr + MAX_INSERTS);
  }

  compareToReference(v.toViewC(), vRef);

  ViewType<COL_TYPE> const & vView = v;
  ArrayView<COL_TYPE const, 2, 1, INDEX_TYPE> const & insertView = columnsToInsert;
  forall(gpu(), 0, numRows,
    [=] __device__ (INDEX_TYPE row)
    {
      vView.insertNonZerosSorted(row, insertView[row], MAX_INSERTS);
    }
  );

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    COL_TYPE const * const colPtr = columnsToInsert[row];
    vRef[row].insert(colPtr, colPtr + MAX_INSERTS);
  }

  v.move(chai::CPU);
  compareToReference(v.toViewC(), vRef);
}

/**
 * @brief Test the SparsityPatternView remove method on device.
 * @param [in/out] v the SparsityPattern to test.
 * @param [in/out] vRef the reference to compare to.
 * @param [in] MAX_REMOVES the maximum number of removes.
 */
template <class COL_TYPE>
void removeDeviceTest(SparsityPattern<COL_TYPE> & v, REF_TYPE<COL_TYPE> & vRef,
                      INDEX_TYPE const MAX_REMOVES)
{
  INDEX_TYPE const numRows = v.numRows();
  INDEX_TYPE const numCols = v.numColumns();

  // Create an array of the columns to remove from each row.
  Array<COL_TYPE, 2> columnsToRemove(numRows, MAX_REMOVES);

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  // Initialize the columns to remove.
  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
    {
      columnsToRemove(row, i) = columnDist(gen);
    }    
  }

  // Create views and remove the columns on device.
  ViewType<COL_TYPE> const & vView = v;
  ArrayView<COL_TYPE const, 2> const & insertView = columnsToRemove;
  forall(gpu(), 0, numRows,
    [=] __device__ (INDEX_TYPE row)
    {
      for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
      {
        vView.removeNonZero(row, insertView(row, i));
      }
    }
  );

  // Remove the columns from vRef
  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
    {
      vRef[row].erase(columnsToRemove(row, i));
    }
  }

  // Move v back to the host and compare.
  v.move(chai::CPU);
  compareToReference(v.toViewC(), vRef);
}

/**
 * @brief Test the SparsityPatternView remove multiple method on device.
 * @param [in/out] v the SparsityPattern to test.
 * @param [in/out] vRef the reference to compare to.
 * @param [in] MAX_REMOVES the maximum number of removes.
 */
template <class COL_TYPE>
void removeMultipleDeviceTest(SparsityPattern<COL_TYPE> & v, REF_TYPE<COL_TYPE> & vRef,
                              INDEX_TYPE MAX_REMOVES)
{
  INDEX_TYPE const numRows = v.numRows();
  INDEX_TYPE const numCols = v.numColumns();

  Array<COL_TYPE, 2> columnsToRemove(numRows, MAX_REMOVES);

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
    {
      columnsToRemove(row, i) = columnDist(gen);
    }
  }

  ViewType<COL_TYPE> const & vView = v;
  ArrayView<COL_TYPE const, 2> const & removeView = columnsToRemove;

  forall(gpu(), 0, numRows,
    [=] __device__ (INDEX_TYPE row)
    {
      vView.removeNonZeros(row, removeView[row], MAX_REMOVES);
    }
  );

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
    {
      vRef[row].erase(columnsToRemove(row, i));
    }
  }

  v.move(chai::CPU);
  compareToReference(v.toViewC(), vRef);
}

/**
 * @brief Test the SparsityPatternView removeSorted method on device.
 * @param [in/out] v the SparsityPattern to test.
 * @param [in/out] vRef the reference to compare to.
 * @param [in] MAX_REMOVES the maximum number of removes.
 */
template <class COL_TYPE>
void removeSortedDeviceTest(SparsityPattern<COL_TYPE> & v, REF_TYPE<COL_TYPE> & vRef,
                            INDEX_TYPE MAX_REMOVES)
{
  INDEX_TYPE const numRows = v.numRows();
  INDEX_TYPE const numCols = v.numColumns();

  Array<COL_TYPE, 2> columnsToRemove(numRows, MAX_REMOVES);

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
    {
      columnsToRemove(row, i) = columnDist(gen);
    }

    // Sort the columns so they're ready to be removed.
    COL_TYPE * const colPtr = columnsToRemove[row];
    std::sort(colPtr, colPtr + MAX_REMOVES);
  }

  ViewType<COL_TYPE> const & vView = v;
  ArrayView<COL_TYPE const, 2> const & removeView = columnsToRemove;

  forall(gpu(), 0, numRows,
    [=] __device__ (INDEX_TYPE row)
    {
      vView.removeNonZerosSorted(row, removeView[row], MAX_REMOVES);
    }
  );

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
    {
      vRef[row].erase(columnsToRemove(row, i));
    }
  }

  v.move(chai::CPU);
  compareToReference(v.toViewC(), vRef);
}

/**
 * @brief Test the empty method of the SparsityPatternView on device.
 * @param [in] v the SparsityPatternView to test.
 */
template <class COL_TYPE>
void emptyDeviceTest(ViewType<COL_TYPE const> const & v)
{
  INDEX_TYPE const numRows = v.numRows();

  // Initialize each row to only contain even numbered columns.
  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    COL_TYPE const * const columns = v.getColumns(row);
    COL_TYPE * const columnsNC = const_cast<COL_TYPE *>(columns);
    for (INDEX_TYPE i = 0; i < v.numNonZeros(row); ++i)
    {
      columnsNC[i] = COL_TYPE(2 * i);
    }
  }

  // Check that each row contains the even columns and no odd columns on device.
  forall(gpu(), 0, numRows,
    [=] __device__ (INDEX_TYPE row)
    {
      for (INDEX_TYPE i = 0; i < v.numNonZeros(row); ++i)
      {
        GEOS_ERROR_IF(v.empty(row, COL_TYPE(2 * i)), "Row should contain even columns.");
        GEOS_ERROR_IF(!v.empty(row, COL_TYPE(2 * i + 1)), "Row should not contain odd columns.");                
      }
    }
  );
}

#endif

} /* namespace internal */

TEST(SparsityPattern, construction)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr int NCOLS = 150;
  constexpr int SIZE_HINT = 5;

  // Test the construction with no hint
  {
    SparsityPattern<uint> sp(NROWS, NCOLS);
    EXPECT_EQ(sp.numRows(), NROWS);
    EXPECT_EQ(sp.numColumns(), NCOLS);
    EXPECT_EQ(sp.numNonZeros(), 0);
    EXPECT_TRUE(sp.empty());

    for (INDEX_TYPE row = 0; row < NROWS; ++row)
    {
      EXPECT_EQ(sp.numNonZeros(row), 0);
      EXPECT_EQ(sp.nonZeroCapacity(row), 0);
      EXPECT_EQ(sp.empty(row), true);
      uint const * const columns = sp.getColumns(row);
      EXPECT_EQ(columns, nullptr);
    }
  }

  // Test the construction with a hint
  {
    SparsityPattern<uint> sp(NROWS, NCOLS, SIZE_HINT);
    EXPECT_EQ(sp.numRows(), NROWS);
    EXPECT_EQ(sp.numColumns(), NCOLS);
    EXPECT_EQ(sp.numNonZeros(), 0);
    EXPECT_EQ(sp.nonZeroCapacity(), NROWS * SIZE_HINT);
    EXPECT_TRUE(sp.empty());

    for (INDEX_TYPE row = 0; row < NROWS; ++row)
    {
      EXPECT_EQ(sp.numNonZeros(row), 0);
      EXPECT_EQ(sp.nonZeroCapacity(row), SIZE_HINT);
      EXPECT_EQ(sp.empty(row), true);

      uint const * const columns = sp.getColumns(row);
      ASSERT_NE(columns, nullptr);
    }
  }
}

TEST(SparsityPattern, insert)
{
  {
    constexpr INDEX_TYPE NROWS = 100;
    constexpr int NCOLS = 150;
    constexpr int MAX_INSERTS = 75;

    SparsityPattern<uint> sp(NROWS, NCOLS);
    REF_TYPE<uint> ref(NROWS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::insertTest(sp, ref, MAX_INSERTS);
  }

  {
    constexpr INDEX_TYPE NROWS = SCHAR_MAX + 10;
    constexpr schar NCOLS = SCHAR_MAX;
    constexpr schar MAX_INSERTS = SCHAR_MAX;

    SparsityPattern<schar> sp(NROWS, NCOLS);
    REF_TYPE<schar> ref(NROWS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::insertTest(sp, ref, MAX_INSERTS);
  }
}

TEST(SparsityPattern, insertMultiple)
{
  {
    constexpr INDEX_TYPE NROWS = 100;
    constexpr int NCOLS = 150;
    constexpr int MAX_INSERTS = 75;

    SparsityPattern<uint> sp(NROWS, NCOLS);
    REF_TYPE<uint> ref(NROWS);
    internal::insertMultipleTest(sp, ref, MAX_INSERTS);
    internal::insertMultipleTest(sp, ref, MAX_INSERTS);
    internal::insertMultipleTest(sp, ref, MAX_INSERTS);
    internal::insertMultipleTest(sp, ref, MAX_INSERTS);
  }

  {
    constexpr INDEX_TYPE NROWS = SCHAR_MAX + 10;
    constexpr schar NCOLS = SCHAR_MAX;
    constexpr schar MAX_INSERTS = SCHAR_MAX;

    SparsityPattern<schar> sp(NROWS, NCOLS);
    REF_TYPE<schar> ref(NROWS);
    internal::insertMultipleTest(sp, ref, MAX_INSERTS);
    internal::insertMultipleTest(sp, ref, MAX_INSERTS);
    internal::insertMultipleTest(sp, ref, MAX_INSERTS);
    internal::insertMultipleTest(sp, ref, MAX_INSERTS);
  }
}

TEST(SparsityPattern, insertSorted)
{
  {
    constexpr INDEX_TYPE NROWS = 100;
    constexpr int NCOLS = 150;
    constexpr int MAX_INSERTS = 75;

    SparsityPattern<uint> sp(NROWS, NCOLS);
    REF_TYPE<uint> ref(NROWS);
    internal::insertSortedTest(sp, ref, MAX_INSERTS);
    internal::insertSortedTest(sp, ref, MAX_INSERTS);
    internal::insertSortedTest(sp, ref, MAX_INSERTS);
    internal::insertSortedTest(sp, ref, MAX_INSERTS);
  }

  {
    constexpr INDEX_TYPE NROWS = SCHAR_MAX + 10;
    constexpr schar NCOLS = SCHAR_MAX;
    constexpr schar MAX_INSERTS = SCHAR_MAX;

    SparsityPattern<schar> sp(NROWS, NCOLS);
    REF_TYPE<schar> ref(NROWS);
    internal::insertSortedTest(sp, ref, MAX_INSERTS);
    internal::insertSortedTest(sp, ref, MAX_INSERTS);
    internal::insertSortedTest(sp, ref, MAX_INSERTS);
    internal::insertSortedTest(sp, ref, MAX_INSERTS);
  }
}

TEST(SparsityPattern, remove)
{
  {
    constexpr INDEX_TYPE NROWS = 100;
    constexpr int NCOLS = 150;
    constexpr int MAX_INSERTS = 75;
    constexpr int MAX_REMOVES = 75;

    SparsityPattern<uint> sp(NROWS, NCOLS);
    REF_TYPE<uint> ref(NROWS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::removeTest(sp.toView(), ref, MAX_REMOVES);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::removeTest(sp.toView(), ref, MAX_REMOVES);
  }

  {
    constexpr INDEX_TYPE NROWS = SCHAR_MAX + 10;
    constexpr schar NCOLS = SCHAR_MAX;
    constexpr schar MAX_INSERTS = SCHAR_MAX;
    constexpr schar MAX_REMOVES = SCHAR_MAX;

    SparsityPattern<schar> sp(NROWS, NCOLS);
    REF_TYPE<schar> ref(NROWS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::removeTest(sp.toView(), ref, MAX_REMOVES);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::removeTest(sp.toView(), ref, MAX_REMOVES);
  }
}

TEST(SparsityPattern, removeMultiple)
{
  {
    constexpr INDEX_TYPE NROWS = 100;
    constexpr int NCOLS = 150;
    constexpr int MAX_INSERTS = 75;
    constexpr int MAX_REMOVES = 75;

    SparsityPattern<uint> sp(NROWS, NCOLS);
    REF_TYPE<uint> ref(NROWS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::removeMultipleTest(sp.toView(), ref, MAX_REMOVES);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::removeMultipleTest(sp.toView(), ref, MAX_REMOVES);
  }

  {
    constexpr INDEX_TYPE NROWS = SCHAR_MAX + 10;
    constexpr schar NCOLS = SCHAR_MAX;
    constexpr schar MAX_INSERTS = SCHAR_MAX;
    constexpr schar MAX_REMOVES = SCHAR_MAX;

    SparsityPattern<schar> sp(NROWS, NCOLS);
    REF_TYPE<schar> ref(NROWS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::removeMultipleTest(sp.toView(), ref, MAX_REMOVES);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::removeMultipleTest(sp.toView(), ref, MAX_REMOVES);
  }
}

TEST(SparsityPattern, removeSorted)
{
  {
    constexpr INDEX_TYPE NROWS = 100;
    constexpr int NCOLS = 150;
    constexpr int MAX_INSERTS = 75;
    constexpr int MAX_REMOVES = 75;

    SparsityPattern<uint> sp(NROWS, NCOLS);
    REF_TYPE<uint> ref(NROWS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::removeSortedTest(sp.toView(), ref, MAX_REMOVES);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::removeSortedTest(sp.toView(), ref, MAX_REMOVES);
  }

  {
    constexpr INDEX_TYPE NROWS = SCHAR_MAX + 10;
    constexpr schar NCOLS = SCHAR_MAX;
    constexpr schar MAX_INSERTS = SCHAR_MAX;
    constexpr schar MAX_REMOVES = SCHAR_MAX;

    SparsityPattern<schar> sp(NROWS, NCOLS);
    REF_TYPE<schar> ref(NROWS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::removeSortedTest(sp.toView(), ref, MAX_REMOVES);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::removeSortedTest(sp.toView(), ref, MAX_REMOVES);
  }
}

TEST(SparsityPattern, empty)
{
  {
    constexpr INDEX_TYPE NROWS = 100;
    constexpr int NCOLS = 150;
    constexpr int MAX_INSERTS = 75;

    SparsityPattern<uint> sp(NROWS, NCOLS);
    REF_TYPE<uint> ref(NROWS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::emptyTest(sp.toViewC(), ref);
  }

  {
    constexpr INDEX_TYPE NROWS = SCHAR_MAX + 10;
    constexpr schar NCOLS = SCHAR_MAX;
    constexpr schar MAX_INSERTS = SCHAR_MAX;

    SparsityPattern<schar> sp(NROWS, NCOLS);
    REF_TYPE<schar> ref(NROWS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::emptyTest(sp.toViewC(), ref);
  }
}


TEST(SparsityPattern, capacity)
{
  {
    constexpr INDEX_TYPE NROWS = 100;
    constexpr int NCOLS = 150;
    constexpr int MAX_INSERTS = 75;

    SparsityPattern<uint> sp(NROWS, NCOLS);
    REF_TYPE<uint> ref(NROWS);

    ASSERT_EQ(sp.nonZeroCapacity(), 0);

    // We reserve 2 * MAX_INSERTS per row because each row may grow to have 2 * MAX_INSERTS capacity.
    sp.reserveNonZeros(2 * NROWS * MAX_INSERTS);
    ASSERT_EQ(sp.nonZeroCapacity(), 2 * NROWS * MAX_INSERTS);
    ASSERT_EQ(sp.numNonZeros(), 0);

    for (INDEX_TYPE row = 0; row < NROWS; ++row)
    {
      ASSERT_EQ(sp.nonZeroCapacity(row), 0);
      ASSERT_EQ(sp.numNonZeros(row), 0);
    }

    uint const * const columns = sp.getColumns(0);
    internal::insertTest(sp, ref, MAX_INSERTS);
    uint const * const newColumns = sp.getColumns(0);
    ASSERT_EQ(columns, newColumns);
  }

  {
    constexpr INDEX_TYPE NROWS = SCHAR_MAX + 10;
    constexpr schar NCOLS = SCHAR_MAX;
    constexpr schar MAX_INSERTS = SCHAR_MAX;

    SparsityPattern<schar> sp(NROWS, NCOLS);
    REF_TYPE<schar> ref(NROWS);

    ASSERT_EQ(sp.nonZeroCapacity(), 0);

    // We reserve 2 * MAX_INSERTS per row because each row may grow to have 2 * MAX_INSERTS capacity.
    sp.reserveNonZeros(2 * NROWS * MAX_INSERTS);
    ASSERT_EQ(sp.nonZeroCapacity(), 2 * NROWS * MAX_INSERTS);
    ASSERT_EQ(sp.numNonZeros(), 0);

    for (INDEX_TYPE row = 0; row < NROWS; ++row)
    {
      ASSERT_EQ(sp.nonZeroCapacity(row), 0);
      ASSERT_EQ(sp.numNonZeros(row), 0);
    }

    schar const * const columns = sp.getColumns(0);
    internal::insertTest(sp, ref, MAX_INSERTS);
    schar const * const newColumns = sp.getColumns(0);
    ASSERT_EQ(columns, newColumns);
  }
}

TEST(SparsityPattern, rowCapacity)
{
  {
    constexpr INDEX_TYPE NROWS = 100;
    constexpr int NCOLS = 150;
    constexpr int MAX_INSERTS = 75;

    SparsityPattern<uint> sp(NROWS, NCOLS, MAX_INSERTS);
    REF_TYPE<uint> ref(NROWS);

    uint const * const columns = sp.getColumns(0);
    internal::insertTest(sp, ref, MAX_INSERTS);
    uint const * const newColumns = sp.getColumns(0);
    ASSERT_EQ(columns, newColumns);

    internal::rowCapacityTest(sp, ref);
  }

  {
    constexpr INDEX_TYPE NROWS = SCHAR_MAX + 10;
    constexpr schar NCOLS = SCHAR_MAX;
    constexpr schar MAX_INSERTS = SCHAR_MAX;

    SparsityPattern<schar> sp(NROWS, NCOLS, MAX_INSERTS);
    REF_TYPE<schar> ref(NROWS);

    schar const * const columns = sp.getColumns(0);
    internal::insertTest(sp, ref, MAX_INSERTS);
    schar const * const newColumns = sp.getColumns(0);
    ASSERT_EQ(columns, newColumns);

    internal::rowCapacityTest(sp, ref);
  }
}

TEST(SparsityPattern, compress)
{
  {
    constexpr INDEX_TYPE NROWS = 100;
    constexpr int NCOLS = 150;
    constexpr int MAX_INSERTS = 75;

    SparsityPattern<uint> sp(NROWS, NCOLS, MAX_INSERTS);
    REF_TYPE<uint> ref(NROWS);

    uint const * const columns = sp.getColumns(0);
    internal::insertTest(sp, ref, MAX_INSERTS);
    uint const * const newColumns = sp.getColumns(0);
    ASSERT_EQ(columns, newColumns);

    internal::compressTest(sp, ref);
  }

  {
    constexpr INDEX_TYPE NROWS = SCHAR_MAX + 10;
    constexpr schar NCOLS = SCHAR_MAX;
    constexpr schar MAX_INSERTS = SCHAR_MAX;

    SparsityPattern<schar> sp(NROWS, NCOLS, MAX_INSERTS);
    REF_TYPE<schar> ref(NROWS);

    schar const * const columns = sp.getColumns(0);
    internal::insertTest(sp, ref, MAX_INSERTS);
    schar const * const newColumns = sp.getColumns(0);
    ASSERT_EQ(columns, newColumns);

    internal::compressTest(sp, ref);
  }
}

TEST(SparsityPattern, deepCopy)
{
  {
    constexpr INDEX_TYPE NROWS = 100;
    constexpr int NCOLS = 150;
    constexpr int MAX_INSERTS = 75;
  
    SparsityPattern<uint> sp(NROWS, NCOLS);
    REF_TYPE<uint> ref(NROWS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::deepCopyTest(sp, ref);
  }

  {
    constexpr INDEX_TYPE NROWS = SCHAR_MAX + 10;
    constexpr schar NCOLS = SCHAR_MAX;
    constexpr schar MAX_INSERTS = SCHAR_MAX;
  
    SparsityPattern<schar> sp(NROWS, NCOLS);
    REF_TYPE<schar> ref(NROWS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::deepCopyTest(sp, ref);
  }
}

TEST(SparsityPattern, shallowCopy)
{
  {
    constexpr INDEX_TYPE NROWS = 100;
    constexpr int NCOLS = 150;
    constexpr int MAX_INSERTS = 75;
  
    SparsityPattern<uint> sp(NROWS, NCOLS);
    REF_TYPE<uint> ref(NROWS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::shallowCopyTest(sp);
  }

  {
    constexpr INDEX_TYPE NROWS = SCHAR_MAX + 10;
    constexpr schar NCOLS = SCHAR_MAX;
    constexpr schar MAX_INSERTS = SCHAR_MAX;
  
    SparsityPattern<schar> sp(NROWS, NCOLS);
    REF_TYPE<schar> ref(NROWS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::shallowCopyTest(sp);
  }
}

#ifdef USE_CUDA

CUDA_TEST(SparsityPattern, memoryMotion)
{
  {
    constexpr INDEX_TYPE NROWS = 100;
    constexpr int NCOLS = 150;
    constexpr int MAX_INSERTS = 75;
  
    SparsityPattern<uint> sp(NROWS, NCOLS);
    REF_TYPE<uint> ref(NROWS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::memoryMotionTest(sp.toView());
  }

  {
    constexpr INDEX_TYPE NROWS = SCHAR_MAX + 10;
    constexpr schar NCOLS = SCHAR_MAX;
    constexpr schar MAX_INSERTS = SCHAR_MAX;
  
    SparsityPattern<schar> sp(NROWS, NCOLS);
    REF_TYPE<schar> ref(NROWS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::memoryMotionTest(sp.toView());
  }
}

CUDA_TEST(SparsityPattern, memoryMotionMove)
{
  {
    constexpr INDEX_TYPE NROWS = 100;
    constexpr int NCOLS = 150;
    constexpr int MAX_INSERTS = 75;
  
    SparsityPattern<uint> sp(NROWS, NCOLS);
    REF_TYPE<uint> ref(NROWS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::memoryMotionMove(sp);
  }

  {
    constexpr INDEX_TYPE NROWS = SCHAR_MAX + 10;
    constexpr schar NCOLS = SCHAR_MAX;
    constexpr schar MAX_INSERTS = SCHAR_MAX;
  
    SparsityPattern<schar> sp(NROWS, NCOLS);
    REF_TYPE<schar> ref(NROWS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::memoryMotionMove(sp);
  }
}

CUDA_TEST(SparsityPattern, memoryMotionConst)
{
  {
    constexpr INDEX_TYPE NROWS = 100;
    constexpr int NCOLS = 150;
    constexpr int MAX_INSERTS = 75;
  
    SparsityPattern<uint> sp(NROWS, NCOLS);
    REF_TYPE<uint> ref(NROWS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::memoryMotionConstTest(sp, ref);
  }

  {
    constexpr INDEX_TYPE NROWS = SCHAR_MAX + 10;
    constexpr schar NCOLS = SCHAR_MAX;
    constexpr schar MAX_INSERTS = SCHAR_MAX;
  
    SparsityPattern<schar> sp(NROWS, NCOLS);
    REF_TYPE<schar> ref(NROWS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::memoryMotionConstTest(sp, ref);
  }
}

TEST(SparsityPattern, insertDevice)
{
  {
    constexpr INDEX_TYPE NROWS = 100;
    constexpr int NCOLS = 150;
    constexpr int MAX_INSERTS = 75;
  
    SparsityPattern<uint> sp(NROWS, NCOLS);
    REF_TYPE<uint> ref(NROWS);
    internal::insertDeviceTest(sp, ref, MAX_INSERTS);
    internal::insertDeviceTest(sp, ref, MAX_INSERTS);
    internal::insertDeviceTest(sp, ref, MAX_INSERTS);
    internal::insertDeviceTest(sp, ref, MAX_INSERTS);
  }

  {
    constexpr INDEX_TYPE NROWS = SCHAR_MAX + 10;
    constexpr schar NCOLS = SCHAR_MAX;
    constexpr schar MAX_INSERTS = SCHAR_MAX;
  
    SparsityPattern<schar> sp(NROWS, NCOLS);
    REF_TYPE<schar> ref(NROWS);
    internal::insertDeviceTest(sp, ref, MAX_INSERTS);
    internal::insertDeviceTest(sp, ref, MAX_INSERTS);
    internal::insertDeviceTest(sp, ref, MAX_INSERTS);
    internal::insertDeviceTest(sp, ref, MAX_INSERTS);
  }
}

TEST(SparsityPattern, insertMultipleDevice)
{
  {
    constexpr INDEX_TYPE NROWS = 100;
    constexpr int NCOLS = 150;
    constexpr int MAX_INSERTS = 75;
  
    SparsityPattern<uint> sp(NROWS, NCOLS);
    REF_TYPE<uint> ref(NROWS);
    internal::insertMultipleDeviceTest(sp, ref, MAX_INSERTS);
    internal::insertMultipleDeviceTest(sp, ref, MAX_INSERTS);
    internal::insertMultipleDeviceTest(sp, ref, MAX_INSERTS);
    internal::insertMultipleDeviceTest(sp, ref, MAX_INSERTS);
  }

  {
    constexpr INDEX_TYPE NROWS = SCHAR_MAX + 10;
    constexpr schar NCOLS = SCHAR_MAX;
    constexpr schar MAX_INSERTS = SCHAR_MAX;
  
    SparsityPattern<schar> sp(NROWS, NCOLS);
    REF_TYPE<schar> ref(NROWS);
    internal::insertMultipleDeviceTest(sp, ref, MAX_INSERTS);
    internal::insertMultipleDeviceTest(sp, ref, MAX_INSERTS);
    internal::insertMultipleDeviceTest(sp, ref, MAX_INSERTS);
    internal::insertMultipleDeviceTest(sp, ref, MAX_INSERTS);
  }
}

TEST(SparsityPattern, insertSortedDevice)
{
  {
    constexpr INDEX_TYPE NROWS = 100;
    constexpr int NCOLS = 150;
    constexpr int MAX_INSERTS = 75;
  
    SparsityPattern<uint> sp(NROWS, NCOLS);
    REF_TYPE<uint> ref(NROWS);
    internal::insertSortedDeviceTest(sp, ref, MAX_INSERTS);
    internal::insertSortedDeviceTest(sp, ref, MAX_INSERTS);
    internal::insertSortedDeviceTest(sp, ref, MAX_INSERTS);
    internal::insertSortedDeviceTest(sp, ref, MAX_INSERTS);
  }

  {
    constexpr INDEX_TYPE NROWS = SCHAR_MAX + 10;
    constexpr schar NCOLS = SCHAR_MAX;
    constexpr schar MAX_INSERTS = SCHAR_MAX;
  
    SparsityPattern<schar> sp(NROWS, NCOLS);
    REF_TYPE<schar> ref(NROWS);
    internal::insertSortedDeviceTest(sp, ref, MAX_INSERTS);
    internal::insertSortedDeviceTest(sp, ref, MAX_INSERTS);
    internal::insertSortedDeviceTest(sp, ref, MAX_INSERTS);
    internal::insertSortedDeviceTest(sp, ref, MAX_INSERTS);
  }
}

TEST(SparsityPattern, removeDevice)
{
  {
    constexpr INDEX_TYPE NROWS = 100;
    constexpr int NCOLS = 150;
    constexpr int MAX_INSERTS = 75;
    constexpr int MAX_REMOVES = 75;
  
    SparsityPattern<uint> sp(NROWS, NCOLS);
    REF_TYPE<uint> ref(NROWS);
    internal::insertDeviceTest(sp, ref, MAX_INSERTS);
    internal::removeDeviceTest(sp, ref, MAX_REMOVES);
    internal::insertDeviceTest(sp, ref, MAX_INSERTS);
    internal::removeDeviceTest(sp, ref, MAX_REMOVES);
  }

  {
    constexpr INDEX_TYPE NROWS = SCHAR_MAX + 10;
    constexpr schar NCOLS = SCHAR_MAX;
    constexpr schar MAX_INSERTS = SCHAR_MAX;
    constexpr schar MAX_REMOVES = SCHAR_MAX;
  
    SparsityPattern<schar> sp(NROWS, NCOLS);
    REF_TYPE<schar> ref(NROWS);
    internal::insertDeviceTest(sp, ref, MAX_INSERTS);
    internal::removeDeviceTest(sp, ref, MAX_REMOVES);
    internal::insertDeviceTest(sp, ref, MAX_INSERTS);
    internal::removeDeviceTest(sp, ref, MAX_REMOVES);
  }
}

TEST(SparsityPattern, removeMultipleDevice)
{
  {
    constexpr INDEX_TYPE NROWS = 100;
    constexpr int NCOLS = 150;
    constexpr int MAX_INSERTS = 75;
    constexpr int MAX_REMOVES = 75;
  
    SparsityPattern<uint> sp(NROWS, NCOLS);
    REF_TYPE<uint> ref(NROWS);
    internal::insertDeviceTest(sp, ref, MAX_INSERTS);
    internal::removeMultipleDeviceTest(sp, ref, MAX_REMOVES);
    internal::insertDeviceTest(sp, ref, MAX_INSERTS);
    internal::removeMultipleDeviceTest(sp, ref, MAX_REMOVES);
  }

  {
    constexpr INDEX_TYPE NROWS = SCHAR_MAX + 10;
    constexpr schar NCOLS = SCHAR_MAX;
    constexpr schar MAX_INSERTS = SCHAR_MAX;
    constexpr schar MAX_REMOVES = SCHAR_MAX;
  
    SparsityPattern<schar> sp(NROWS, NCOLS);
    REF_TYPE<schar> ref(NROWS);
    internal::insertDeviceTest(sp, ref, MAX_INSERTS);
    internal::removeMultipleDeviceTest(sp, ref, MAX_REMOVES);
    internal::insertDeviceTest(sp, ref, MAX_INSERTS);
    internal::removeMultipleDeviceTest(sp, ref, MAX_REMOVES);
  }
}

TEST(SparsityPattern, removeSortedDevice)
{
  {
    constexpr INDEX_TYPE NROWS = 100;
    constexpr int NCOLS = 150;
    constexpr int MAX_INSERTS = 75;
    constexpr int MAX_REMOVES = 75;
  
    SparsityPattern<uint> sp(NROWS, NCOLS);
    REF_TYPE<uint> ref(NROWS);
    internal::insertDeviceTest(sp, ref, MAX_INSERTS);
    internal::removeSortedDeviceTest(sp, ref, MAX_REMOVES);
    internal::insertDeviceTest(sp, ref, MAX_INSERTS);
    internal::removeSortedDeviceTest(sp, ref, MAX_REMOVES);
  }

  {
    constexpr INDEX_TYPE NROWS = SCHAR_MAX + 10;
    constexpr schar NCOLS = SCHAR_MAX;
    constexpr schar MAX_INSERTS = SCHAR_MAX;
    constexpr schar MAX_REMOVES = SCHAR_MAX;
  
    SparsityPattern<schar> sp(NROWS, NCOLS);
    REF_TYPE<schar> ref(NROWS);
    internal::insertDeviceTest(sp, ref, MAX_INSERTS);
    internal::removeSortedDeviceTest(sp, ref, MAX_REMOVES);
    internal::insertDeviceTest(sp, ref, MAX_INSERTS);
    internal::removeSortedDeviceTest(sp, ref, MAX_REMOVES);
  }
}

TEST(SparsityPattern, emptyDevice)
{
  {
    constexpr INDEX_TYPE NROWS = 100;
    constexpr int NCOLS = 150;
    constexpr int MAX_INSERTS = 75;

    SparsityPattern<uint> sp(NROWS, NCOLS);
    REF_TYPE<uint> ref(NROWS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::emptyDeviceTest(sp.toViewC());
  }

  {
    constexpr INDEX_TYPE NROWS = SCHAR_MAX + 10;
    constexpr schar NCOLS = SCHAR_MAX;
    constexpr schar MAX_INSERTS = SCHAR_MAX / 3;

    SparsityPattern<schar> sp(NROWS, NCOLS);
    REF_TYPE<schar> ref(NROWS);
    internal::insertTest(sp, ref, MAX_INSERTS);
    internal::emptyDeviceTest(sp.toViewC());
  }
}

#endif

} // namespace LvArray

int main( int argc, char* argv[] )
{
#if defined(USE_MPI)
  MPI_Init( &argc, &argv );
  logger::InitializeLogger( MPI_COMM_WORLD );
#else
  logger::InitializeLogger( );
#endif

  int result = 0;
  testing::InitGoogleTest( &argc, argv );
  result = RUN_ALL_TESTS();

  logger::FinalizeLogger();

#if defined(USE_MPI)
  MPI_Finalize();
#endif
  return result;
}
