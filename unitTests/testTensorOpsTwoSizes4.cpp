/*
 * Copyright (c) 2020, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "testTensorOpsTwoSizes.hpp"

namespace LvArray
{
namespace testing
{

TYPED_TEST( TwoSizesTest, scaledCopy )
{
  this->testScaledCopy();
}

TYPED_TEST( TwoSizesTest, add )
{
  this->testAdd();
}

TYPED_TEST( TwoSizesTest, plusAikAjk )
{
  this->testPlusAikAjk();
}

TYPED_TEST( TwoSizesTest, transpose )
{
  this->testTranspose();
}

} // namespace testing
} // namespace LvArray
