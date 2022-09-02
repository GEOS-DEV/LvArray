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
  /**
   *
   */
  enum Type
  {
    EIGENVALUES,
    EIGENVALUES_AND_VECTORS,
  };

  /**
   *
   */
  enum Range
  {
    ALL,
    IN_INTERVAL,
    BY_INDEX,
  };

  /**
   *
   */
  EigenDecompositionOptions( Type const typeP, double const abstolP=0 ):
    type{ typeP },
    abstol{ abstolP }
  {
    LVARRAY_ERROR_IF( type != EIGENVALUES && type != EIGENVALUES_AND_VECTORS, "Wrong type provided: type = " << type );
  }

  /**
   *
   */
  EigenDecompositionOptions(
    Type const typeP,
    double const rangeMinP,
    double const rangeMaxP,
    double const abstolP ):
    type{ typeP },
    range{ IN_INTERVAL },
    rangeMin{ rangeMinP },
    rangeMax{ rangeMaxP },
    abstol{ abstolP }
  {
    LVARRAY_ERROR_IF( type != EIGENVALUES && type != EIGENVALUES_AND_VECTORS, "Wrong type provided: type = " << type );
    LVARRAY_ERROR_IF_GE( rangeMin, rangeMax );
  }

  /**
   * TODO: Not sure how I feel about the one based indexing for eigenvalues by index.
   */
  EigenDecompositionOptions(
    Type const typeP,
    DenseInt const indexMinP,
    DenseInt const indexMaxP,
    double const abstolP ):
    type{ typeP },
    range{ IN_INTERVAL },
    indexMin{ indexMinP },
    indexMax{ indexMaxP },
    abstol{ abstolP }
  {
    LVARRAY_ERROR_IF( type != EIGENVALUES && type != EIGENVALUES_AND_VECTORS, "Wrong type provided: type = " << type );
    LVARRAY_ERROR_IF_LT( indexMin, 1 );
    LVARRAY_ERROR_IF_GT( indexMin, indexMax );
  }

  /**
   *
   */
  char const * typeArg() const
  {
    static constexpr char const * const eigenvalueString = "N";
    static constexpr char const * const eigenvectorString = "V";

    return type == EIGENVALUES ? eigenvalueString : eigenvectorString;
  }

  /**
   *
   */
  char const * rangeArg() const
  {
    static constexpr char const * const allString = "A";
    static constexpr char const * const intervalString = "V";
    static constexpr char const * const indexString = "I";

    if( range == ALL )
    { return allString; }

    return range == IN_INTERVAL ? intervalString : indexString;
  }

  ///
  Type const type;

  ///
  Range const range = ALL;
  
  ///
  double const rangeMin = std::numeric_limits< double >::max();
  
  ///
  double const rangeMax = std::numeric_limits< double >::lowest();
  
  ///
  DenseInt const indexMin = std::numeric_limits< DenseInt >::max();
  
  ///
  DenseInt const indexMax = std::numeric_limits< DenseInt >::lowest();
  
  ///
  double const abstol = 0;
};


/**
 *
 */
template< typename T >
DenseInt heevr(
  BuiltInBackends const backend,
  EigenDecompositionOptions const decompositionOptions,
  Matrix< std::complex< T > > const & A,
  Vector< T > const & eigenValues,
  Matrix< std::complex< T > > const & eigenVectors,
  Vector< DenseInt > const & support,
  Workspace< std::complex< T > > & workspace,
  SymmetricMatrixStorageType const storageType );

/**
 *
 */
template< typename BACK_END, typename T, int USD, typename INDEX_TYPE >
DenseInt heevr(
  BACK_END && backend,
  EigenDecompositionOptions const decompositionOptions,
  ArraySlice< std::complex< T >, 2, USD, INDEX_TYPE > const & A,
  ArraySlice< T, 1, 0, INDEX_TYPE > const & eigenValues,
  ArraySlice< std::complex< T >, 2, USD, INDEX_TYPE > const & eigenVectors,
  ArraySlice< DenseInt, 1, 0, INDEX_TYPE > const & support,
  Workspace< std::complex< T > > & workspace,
  SymmetricMatrixStorageType const storageType )
{
  Matrix< std::complex< T > > AMatrix( A );
  Vector< T > eigenValuesVector( eigenValues );
  Matrix< std::complex< T > > eigenVectorsMatrix( eigenVectors );
  Vector< DenseInt > supportVector( support );

  return heevr(
    std::forward< BACK_END >( backend ),
    decompositionOptions,
    AMatrix,
    eigenValuesVector,
    eigenVectorsMatrix,
    supportVector,
    workspace,
    storageType );
}

/**
 *
 */
template< typename BACK_END, typename T, int USD, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
DenseInt heevr(
  BACK_END && backend,
  EigenDecompositionOptions const decompositionOptions,
  ArrayView< std::complex< T >, 2, USD, INDEX_TYPE, BUFFER_TYPE > const & A,
  ArrayView< T, 1, 0, INDEX_TYPE, BUFFER_TYPE > const & eigenValues,
  ArrayView< std::complex< T >, 2, USD, INDEX_TYPE, BUFFER_TYPE > const & eigenVectors,
  ArrayView< DenseInt, 1, 0, INDEX_TYPE, BUFFER_TYPE > const & support,
  Workspace< std::complex< T > > & workspace,
  SymmetricMatrixStorageType const storageType )
{
  // Unclear about the touch here since half of A is destroyed, maybe it's not necessary.
  MemorySpace const space = getSpaceForBackend( backend );
  A.move( space, true );
  eigenValues.move( space, true );
  eigenVectors.move( space, true );
  support.move( space, true );

  return heevr(
    std::forward< BACK_END >( backend ),
    decompositionOptions,
    A.toSlice(),
    eigenValues.toSlice(),
    eigenVectors.toSlice(),
    support.toSlice(),
    workspace,
    storageType );
}

} // namespace dense
} // namespace LvArray