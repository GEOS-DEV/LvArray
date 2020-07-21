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

TYPED_TEST( TwoSizesTest, AjiBj )
{
  this->testAjiBj();
}

TYPED_TEST( TwoSizesTest, plusAjiBj )
{
  this->testPlusAjiBj();
}

TYPED_TEST( TwoSizesTest, copy )
{
  this->testCopy();
}

} // namespace testing
} // namespace LvArray
