#pragma once

#include "common.hpp"

namespace LvArray
{
namespace dense
{

/**
 *
 */
struct EigenDecompositionOptions
{
  enum Type
  {
    EIGENVALUES,
    EIGENVALUES_AND_VECTORS,
  };

  enum Range
  {
    ALL,
    IN_RANGE,
    BY_INDEX,
  };

  Type const m_type;
  Range const m_range;
  double const rangeMin;
  double const rangeMax;
  int const indexMin;
  int const indexMax;
};

/**
 *
 */
template< typename T, INDEX_TYPE >
void heev(
  MemorySpace const space,
  EigenDecompositionOption const options,
  ArraySlice< std::complex< T >, 2, 1, INDEX_TYPE > const & A,
  ArraySlice< T, 1, 0, INDEX_TYPE > const & eigenValues,
  Workspace< T > & workspace,
  SymmetricMatrixStorageType const storageType = SymmetricMatrixStorageType::UPPER_TRIANGULAR
);

} // namespace dense
} // namespace LvArray