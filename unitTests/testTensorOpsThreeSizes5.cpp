/*
 * Copyright (c) 2020, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "testTensorOpsThreeSizes.hpp"

namespace LvArray
{
namespace testing
{

TYPED_TEST( ThreeSizesTest, AkiBkj )
{
  this->testAkiBkj();
}

} // namespace testing
} // namespace LvArray
