#pragma once

#include "common.hpp"

namespace LvArray
{
namespace dense
{

/**
 */
template< typename T >
void geqrf(
    BuiltInBackends const backend,
    Matrix< T > const & A,
    Vector< T > const & tau,
    Workspace< T > & workspace );

/**
 *
 */
template< typename BACK_END, typename T, int USD_A, int USD_V, typename INDEX_TYPE >
void geqrf(
  BACK_END && backend,
  ArraySlice< T, 2, USD_A, INDEX_TYPE > const & A,
  ArraySlice< T, 1, 0, INDEX_TYPE > const & tau,
  Workspace< T > & workspace )
{
  Matrix< T > AMatrix( A );
  Vector< T > tauVector( tau );

  return geqrf(
    std::forward< BACK_END >( backend ),
    AMatrix,
    tauVector,
    workspace );
}

/**
 *
 */
template< typename BACK_END, typename T, int USD_A, int USD_V, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
DenseInt geqrf(
  BACK_END && backend,
  ArrayView< T, 2, USD_A, INDEX_TYPE, BUFFER_TYPE > const & A,
  ArrayView< T, 1, 0, INDEX_TYPE, BUFFER_TYPE > const & eigenvalues,
  Workspace< T > & workspace )
{
  return geqrf(
    std::forward< BACK_END >( backend ),
    A.toSlice(),
    tau.toSlice(),
    workspace );
}

} // namespace dense
} // namespace LvArray