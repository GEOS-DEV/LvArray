#pragma once

#include "common.hpp"
#include "../Array.hpp"
#include "../ChaiBuffer.hpp"

namespace LvArray
{
namespace dense
{

/**
 *
 */
enum class SymmetricMatrixStorageType
{
  UPPER_TRIANGULAR,
  LOWER_TRIANGULAR,
};

/**
 * TODO: move to internal namespace
 */
char const * getOption( SymmetricMatrixStorageType const option );

/**
 *
 */
template< typename T >
struct Workspace
{
  Array< std::complex< T >, 1, RAJA::PERM_I, std::ptrdiff_t, ChaiBuffer > workComplex;
  Array< T, 1, RAJA::PERM_I, std::ptrdiff_t, ChaiBuffer > rWork;
};

} // namespace dense
} // namespace LvArray