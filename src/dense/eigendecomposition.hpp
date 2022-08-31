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
    IN_INTERVAL,
    BY_INDEX,
  };

  char const * typeArg() const
  {
    static constexpr char const * const eigenvalueString = "N";
    static constexpr char const * const eigenvectorString = "V";

    return type == Type::EIGENVALUES ? eigenvalueString : eigenvectorString;
  }

  char const * rangeArg() const
  {
    static constexpr char const * const allString = "A";
    static constexpr char const * const intervalString = "V";
    static constexpr char const * const indexString = "I";

    if( range == Range::ALL )
    { return allString; }

    return range == Range::IN_INTERVAL ? intervalString : indexString;
  }

  Type const type;
  Range const range;
  double const rangeMin;
  double const rangeMax;
  int const indexMin;
  int const indexMax;
  double const abstol;
};

} // namespace dense
} // namespace LvArray