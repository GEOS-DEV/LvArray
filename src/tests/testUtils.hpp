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

#ifndef TEST_UTILS_HPP_
#define TEST_UTILS_HPP_

#include "CXX_UtilsConfig.hpp"
#include <string>
#include <ostream>

#ifdef USE_CUDA

#define CUDA_TEST(X, Y)              \
  static void cuda_test_##X##Y();    \
  TEST(X, Y) { cuda_test_##X##Y(); } \
  static void cuda_test_##X##Y()

#endif

// Comparator that compares a std::pair by it's first object.
template <class A, class B, class COMP=std::less<B>>
struct PairComp {
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline
  constexpr bool operator()(const std::pair<A, B>& lhs, const std::pair<A, B>& rhs) const
  {
    return COMP()(lhs.second, rhs.second); 
  }
};

/**
 * @class TestString
 * @brief A wrapper around std::string that adds a constructor that takes a number
 * and converts it to a string. Used for testing purposes.
 */
class TestString : public std::string
{
public:
  template <class T=int>
  TestString(T val=0) :
    std::string(std::to_string(val))
  {}
};

/**
 * @class Tensor
 * @brief A simple struct containing three doubles used for testing purposes.
 * Can be used on the device.
 */
struct Tensor
{
  double x, y, z;

  LVARRAY_HOST_DEVICE Tensor():
    x(), y(), z()
  {}

  LVARRAY_HOST_DEVICE explicit Tensor( double val ):
    x( 3 * val ), y( 3 * val + 1 ), z( 3 * val + 2 )
  {}

  LVARRAY_HOST_DEVICE Tensor( const Tensor & from) :
    x(from.x),
    y(from.y),
    z(from.z)
  {}

  LVARRAY_HOST_DEVICE Tensor& operator=( const Tensor& other )
  {
    x = other.x;
    y = other.y;
    z = other.z;
    return *this;
  }

  LVARRAY_HOST_DEVICE Tensor& operator*=( const Tensor& other )
  {
    x *= other.x;
    y *= other.y;
    z *= other.z;
    return *this;
  }

  LVARRAY_HOST_DEVICE Tensor operator*( const Tensor& other ) const
  {
    Tensor result = *this;
    result *= other;
    return result;
  }

  LVARRAY_HOST_DEVICE bool operator==( const Tensor& other ) const
  {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
    return x == other.x && y == other.y && z == other.z;
#pragma GCC diagnostic pop
  }

  LVARRAY_HOST_DEVICE bool operator<( const Tensor& other ) const
  { return x < other.x; }

  LVARRAY_HOST_DEVICE bool operator>( const Tensor& other ) const
  { return x > other.x; }

  LVARRAY_HOST_DEVICE bool operator !=( const Tensor & other ) const
  { return !(*this == other); }
};

/**
  * @brief Helper method used to print out the values of a Tensor.
  */
std::ostream& operator<<(std::ostream& stream, const Tensor & t )
{
  stream << "(" << t.x << ", " << t.y << ", " << t.z << ")";
  return stream;
}

#endif // TEST_UTILS_HPP_
