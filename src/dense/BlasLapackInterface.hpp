#pragma once

#include "common.hpp"

namespace LvArray
{
namespace dense
{

template< typename T >
struct BlasLapackInterface
{
  static constexpr MemorySpace MEMORY_SPACE = MemorySpace::host;

  static void gemm(
    Operation opA,
    Operation opB,
    T const alpha,
    Matrix< T const > const & A,
    Matrix< T const > const & B,
    T const beta, 
    Matrix< T > const & C );
  
  static void gesv(
    Matrix< T > const & A,
    Matrix< T > const & B,
    Vector< int > const & pivots );
};

} // namespace dense
} // namespace LvArray