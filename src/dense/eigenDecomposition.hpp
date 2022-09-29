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
  // TODO(corbett5): Move these enums out into their own enum classes.
  enum Type
  {
    EIGENVALUES,
    EIGENVALUES_AND_LEFT_VECTORS,
    EIGENVALUES_AND_RIGHT_VECTORS,
    EIGENVALUES_AND_LEFT_AND_RIGHT_VECTORS,
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
    LVARRAY_ERROR_IF(
      type != EIGENVALUES &&
      type != EIGENVALUES_AND_LEFT_VECTORS &&
      type != EIGENVALUES_AND_RIGHT_VECTORS &&
      type != EIGENVALUES_AND_LEFT_AND_RIGHT_VECTORS,
      "Wrong type provided: type = " << type );
  }

  /**
   *
   */
  EigenDecompositionOptions(
    Type const typeP,
    double const rangeMinP,
    double const rangeMaxP,
    double const abstolP=0 ):
    type{ typeP },
    range{ IN_INTERVAL },
    rangeMin{ rangeMinP },
    rangeMax{ rangeMaxP },
    abstol{ abstolP }
  {
    LVARRAY_ERROR_IF(
      type != EIGENVALUES &&
      type != EIGENVALUES_AND_LEFT_VECTORS &&
      type != EIGENVALUES_AND_RIGHT_VECTORS &&
      type != EIGENVALUES_AND_LEFT_AND_RIGHT_VECTORS,
      "Wrong type provided: type = " << type );

    LVARRAY_ERROR_IF_GE( rangeMin, rangeMax );
  }

  /**
   * TODO: Not sure how I feel about the one based indexing for eigenvalues by index.
   */
  EigenDecompositionOptions(
    Type const typeP,
    DenseInt const indexMinP,
    DenseInt const indexMaxP,
    double const abstolP=0 ):
    type{ typeP },
    range{ BY_INDEX },
    indexMin{ indexMinP },
    indexMax{ indexMaxP },
    abstol{ abstolP }
  {
    LVARRAY_ERROR_IF(
      type != EIGENVALUES &&
      type != EIGENVALUES_AND_LEFT_VECTORS &&
      type != EIGENVALUES_AND_RIGHT_VECTORS &&
      type != EIGENVALUES_AND_LEFT_AND_RIGHT_VECTORS,
      "Wrong type provided: type = " << type );

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

  /**
   *
   */
  bool computeLeftEigenvectors() const
  {
    return type == EIGENVALUES_AND_LEFT_VECTORS ||
           type == EIGENVALUES_AND_LEFT_AND_RIGHT_VECTORS;
  }

  /**
   *
   */
  bool computeRightEigenvectors() const
  {
    return type == EIGENVALUES_AND_RIGHT_VECTORS ||
           type == EIGENVALUES_AND_LEFT_AND_RIGHT_VECTORS;
  }

  /**
   *
   */
  char const * leftArg() const
  {
    static constexpr char const * const eigenvalueString = "N";
    static constexpr char const * const eigenvectorString = "V";

    return computeLeftEigenvectors() ? eigenvalueString : eigenvectorString;
  }

  /**
   *
   */
  char const * rightArg() const
  {
    static constexpr char const * const eigenvalueString = "N";
    static constexpr char const * const eigenvectorString = "V";

    return computeRightEigenvectors() ? eigenvalueString : eigenvectorString;
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
template< typename BACK_END, typename T, int USD_A, int USD_V, typename INDEX_TYPE >
DenseInt heevr(
  BACK_END && backend,
  EigenDecompositionOptions const decompositionOptions,
  ArraySlice< std::complex< T >, 2, USD_A, INDEX_TYPE > const & A,
  ArraySlice< T, 1, 0, INDEX_TYPE > const & eigenValues,
  ArraySlice< std::complex< T >, 2, USD_V, INDEX_TYPE > const & eigenVectors,
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
template< typename BACK_END, typename T, int USD_A, int USD_V, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
DenseInt heevr(
  BACK_END && backend,
  EigenDecompositionOptions const decompositionOptions,
  ArrayView< std::complex< T >, 2, USD_A, INDEX_TYPE, BUFFER_TYPE > const & A,
  ArrayView< T, 1, 0, INDEX_TYPE, BUFFER_TYPE > const & eigenValues,
  ArrayView< std::complex< T >, 2, USD_V, INDEX_TYPE, BUFFER_TYPE > const & eigenVectors,
  ArrayView< DenseInt, 1, 0, INDEX_TYPE, BUFFER_TYPE > const & support,
  Workspace< std::complex< T > > & workspace,
  SymmetricMatrixStorageType const storageType )
{
  MemorySpace const space = getSpaceForBackend( backend );
  
  // The A matrix isn't touched because it is destroyed.
  A.move( space, false );
  eigenVectors.move( space, true );

#if defined( LVARRAY_USE_MAGMA )
  // MAGMA wants the eigenvalues and support on the CPU.
  if( backend == BuiltInBackends::MAGMA_GPU )
  {
    eigenValues.move( MemorySpace::host, true );
    support.move( MemorySpace::host, true );
  }
  else
#endif
  {
    eigenValues.move( space, true );
    support.move( space, true );
  }

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

/**
 * Real valued A.
 */
template< typename T >
void geev(
  BuiltInBackends const backend,
  EigenDecompositionOptions const decompositionOptions,
  Matrix< T > const & A,
  Vector< T > const & eigenValuesReal,
  Vector< T > const & eigenValuesImag,
  Matrix< T > const & leftEigenvectors,
  Matrix< T > const & rightEigenvectors,
  Workspace< T > & workspace );

/**
 * Complex valued A.
 */
template< typename T >
void geev(
  BuiltInBackends const backend,
  EigenDecompositionOptions const decompositionOptions,
  Matrix< std::complex< T > > const & A,
  Vector< std::complex< T > > const & eigenValues,
  Matrix< std::complex< T > > const & leftEigenvectors,
  Matrix< std::complex< T > > const & rightEigenvectors,
  Workspace< std::complex< T > > & workspace );

} // namespace dense
} // namespace LvArray