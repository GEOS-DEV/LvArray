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

TYPED_TEST( TwoSizesTest, scale )
{
  this->testScale();
}

TYPED_TEST( TwoSizesTest, fill )
{
  this->testFill();
}

TYPED_TEST( TwoSizesTest, AiBj )
{
  this->testAiBj();
}

} // namespace testing
} // namespace LvArray
