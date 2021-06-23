/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
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

TYPED_TEST( TwoSizesTest, Rij_add_AiBj )
{
  this->testPlusAiBj();
}

TYPED_TEST( TwoSizesTest, AijBj )
{
  this->testAijBj();
}

TYPED_TEST( TwoSizesTest, Ri_add_AijBj )
{
  this->testPlusAijBj();
}

} // namespace testing
} // namespace LvArray
