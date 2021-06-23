/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "limits.hpp"

// TPL includes
#include <gtest/gtest.h>

typedef  int32_t int32;
typedef uint32_t uint32;
typedef  int64_t int64;
typedef uint64_t uint64;


TEST( IntegerConversion, unsignedToSigned )
{
  uint32 source0 = std::numeric_limits< int32 >::max();
  uint32 source1 = std::numeric_limits< uint32 >::max();
  uint64 source2 = std::numeric_limits< int64 >::max();
  uint64 source3 = std::numeric_limits< uint64 >::max();

  int32 compare0 = std::numeric_limits< int32 >::max();
  int64 compare2 = std::numeric_limits< int64 >::max();

  ASSERT_TRUE( compare0 == LvArray::integerConversion< int32 >( source0 ) );
  ASSERT_DEATH_IF_SUPPORTED( LvArray::integerConversion< int32 >( source1 ), "" );

  ASSERT_TRUE( compare2 == LvArray::integerConversion< int64 >( source2 ) );
  ASSERT_DEATH_IF_SUPPORTED( LvArray::integerConversion< int64 >( source3 ), "" );
}


TEST( IntegerConversion, signedToUnsigned )
{
  int32 source0 = -1;
  int64 source1 = -1;
  int64 source3 = std::numeric_limits< int64 >::max();

  ASSERT_DEATH_IF_SUPPORTED( LvArray::integerConversion< uint32 >( source0 ), "" );
  ASSERT_DEATH_IF_SUPPORTED( LvArray::integerConversion< uint64 >( source1 ), "" );
  ASSERT_DEATH_IF_SUPPORTED( LvArray::integerConversion< uint32 >( source3 ), "" );
}


TEST( IntegerConversion, sameSign )
{
  int64 source0 = std::numeric_limits< int32 >::lowest();
  int64 source1 = std::numeric_limits< int64 >::lowest();

  ASSERT_EQ( std::numeric_limits< int32 >::lowest(), LvArray::integerConversion< int32 >( source0 ) );
  ASSERT_DEATH_IF_SUPPORTED( LvArray::integerConversion< int32 >( source1 ), "" );


  int64 source2 = std::numeric_limits< int32 >::max();
  int64 source3 = std::numeric_limits< int64 >::max();

  ASSERT_EQ( std::numeric_limits< int32 >::max(), LvArray::integerConversion< int32 >( source2 ) );
  ASSERT_DEATH_IF_SUPPORTED( LvArray::integerConversion< int32 >( source3 ), "" );


  uint64 source4 = std::numeric_limits< uint32 >::max();
  uint64 source5 = std::numeric_limits< uint64 >::max();

  ASSERT_EQ( std::numeric_limits< uint32 >::max(), LvArray::integerConversion< uint32 >( source4 ) );
  ASSERT_DEATH_IF_SUPPORTED( LvArray::integerConversion< uint32 >( source5 ), "" );
}
