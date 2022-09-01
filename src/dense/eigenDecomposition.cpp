#include "eigenDecomposition.hpp"

/// This macro provide a flexible interface for Fortran naming convention for compiled objects
// #ifdef FORTRAN_MANGLE_NO_UNDERSCORE
#define FORTRAN_MANGLE( name ) name
// #else
// #define FORTRAN_MANGLE( name ) name ## _
// #endif

extern "C"
{

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_CHEEVR FORTRAN_MANGLE( cheevr )
void LVARRAY_CHEEVR( 
  char const * JOBZ,
  char const * RANGE,
  char const * UPLO,
  LvArray::dense::DenseInt const * N,
  std::complex< float > * A,
  LvArray::dense::DenseInt const * LDA,
  float const * VL,
  float const * VU,
  LvArray::dense::DenseInt const * IL,
  LvArray::dense::DenseInt const * IU,
  float const * ABSTOL,
  LvArray::dense::DenseInt * M,
  float * W,
  std::complex< float > * Z,
  LvArray::dense::DenseInt const * LDZ,
  LvArray::dense::DenseInt * ISUPPZ,
  std::complex< float > * WORK,
  LvArray::dense::DenseInt const * LWORK,
  float * RWORK,
  LvArray::dense::DenseInt const * LRWORK,
  LvArray::dense::DenseInt * IWORK,
  LvArray::dense::DenseInt const * LIWORK,
  LvArray::dense::DenseInt * INFO );

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_ZHEEVR FORTRAN_MANGLE( zheevr )
void LVARRAY_ZHEEVR( 
  char const * JOBZ,
  char const * RANGE,
  char const * UPLO,
  LvArray::dense::DenseInt const * N,
  std::complex< double > * A,
  LvArray::dense::DenseInt const * LDA,
  double const * VL,
  double const * VU,
  LvArray::dense::DenseInt const * IL,
  LvArray::dense::DenseInt const * IU,
  double const * ABSTOL,
  LvArray::dense::DenseInt * M,
  double * W,
  std::complex< double > * Z,
  LvArray::dense::DenseInt const * LDZ,
  LvArray::dense::DenseInt * ISUPPZ,
  std::complex< double > * WORK,
  LvArray::dense::DenseInt const * LWORK,
  double * RWORK,
  LvArray::dense::DenseInt const * LRWORK,
  LvArray::dense::DenseInt * IWORK,
  LvArray::dense::DenseInt const * LIWORK,
  LvArray::dense::DenseInt * INFO );


} // extern "C"

namespace LvArray
{
namespace dense
{
namespace internal
{

/**
 * 
 */
template< typename T >
DenseInt heevr(
  MemorySpace const space,
  EigenDecompositionOptions const decompositionOptions,
  Matrix< std::complex< T > > const & A,
  Vector< T > const & eigenValues,
  Matrix< std::complex< T > > const & eigenVectors,
  Vector< DenseInt > const & support,
  Workspace< std::complex< T > > & workspace,
  SymmetricMatrixStorageType const storageType,
  bool const compute )
{
  LVARRAY_ERROR_IF_NE_MSG( space, MemorySpace::host, "Device not yet supported." );

  LVARRAY_ERROR_IF( !A.isSquare(), "The matrix A must be square." );

  char const * const JOBZ = decompositionOptions.typeArg();
  char const * const RANGE = decompositionOptions.rangeArg();
  char const * const UPLO = getOption( storageType );
  DenseInt const N = A.nCols;
  DenseInt const LDA = A.stride;

  T const VL = decompositionOptions.rangeMin;
  T const VU = decompositionOptions.rangeMax;

  DenseInt maxEigenvaluesToFind = N;
  DenseInt const IL = decompositionOptions.indexMin;
  DenseInt const IU = decompositionOptions.indexMax;
  if( decompositionOptions.range == EigenDecompositionOptions::Range::BY_INDEX )
  {
    LVARRAY_ERROR_IF_GT( IU, N );
    maxEigenvaluesToFind = IU - IL + 1;
  }

  LVARRAY_ERROR_IF_LT( eigenValues.size, maxEigenvaluesToFind );

  DenseInt const ABSTOL = decompositionOptions.abstol;
  DenseInt M = 0;

  if( decompositionOptions.type == EigenDecompositionOptions::Type::EIGENVALUES_AND_VECTORS )
  {
    LVARRAY_ERROR_IF_NE( eigenVectors.nRows, N );
    LVARRAY_ERROR_IF_LT( eigenVectors.nCols, maxEigenvaluesToFind );
  }

  DenseInt const LDZ = std::max( 1, eigenVectors.stride );

  if( decompositionOptions.range == EigenDecompositionOptions::Range::ALL ||
      ( decompositionOptions.range == EigenDecompositionOptions::Range::BY_INDEX &&
        maxEigenvaluesToFind == N ) )
  {
    LVARRAY_ERROR_IF_LT( support.size, 2 * maxEigenvaluesToFind );
  }

  DenseInt const LWORK = compute ? workspace.work().size : -1;
  DenseInt const LRWORK = compute ? workspace.rwork().size : -1;
  DenseInt const LIWORK = compute ? workspace.iwork().size : -1;

  DenseInt INFO = 0;

  // With C++ 17 we can remove the reinterpret_cast with constexpr if.
  if( std::is_same< T, float >::value )
  {
    LVARRAY_CHEEVR(
      JOBZ,
      RANGE,
      UPLO,
      &N,
      reinterpret_cast< std::complex< float > * >( A.data ),
      &LDA,
      reinterpret_cast< float const * >( &VL ),
      reinterpret_cast< float const * >( &VU ),
      &IL,
      &IU,
      reinterpret_cast< float const * >( &ABSTOL ),
      &M,
      reinterpret_cast< float * >( eigenValues.data ),
      reinterpret_cast< std::complex< float > * >( eigenVectors.data ),
      &LDZ,
      support.data,
      reinterpret_cast< std::complex< float > * >( workspace.work().data ),
      &LWORK,
      reinterpret_cast< float * >( workspace.rwork().data ),
      &LRWORK,
      workspace.iwork().data,
      &LIWORK,
      &INFO );
  }
  else
  {
    LVARRAY_ZHEEVR(
      JOBZ,
      RANGE,
      UPLO,
      &N,
      reinterpret_cast< std::complex< double > * >( A.data ),
      &LDA,
      reinterpret_cast< double const * >( &VL ),
      reinterpret_cast< double const * >( &VU ),
      &IL,
      &IU,
      reinterpret_cast< double const * >( &ABSTOL ),
      &M,
      reinterpret_cast< double * >( eigenValues.data ),
      reinterpret_cast< std::complex< double > * >( eigenVectors.data ),
      &LDZ,
      support.data,
      reinterpret_cast< std::complex< double > * >( workspace.work().data ),
      &LWORK,
      reinterpret_cast< double * >( workspace.rwork().data ),
      &LRWORK,
      workspace.iwork().data,
      &LIWORK,
      &INFO );
  }

  LVARRAY_ERROR_IF_NE( INFO, 0 );

  return M;
}

} // namespace internal

////////////////////////////////////////////////////////////////////////////////////////////////////
template< typename T >
DenseInt heevr(
  MemorySpace const space,
  EigenDecompositionOptions const decompositionOptions,
  Matrix< std::complex< T > > const & A,
  Vector< T > const & eigenValues,
  Matrix< std::complex< T > > const & eigenVectors,
  Vector< DenseInt > const & support,
  Workspace< std::complex< T > > & workspace,
  SymmetricMatrixStorageType const storageType )
{
  bool const reallocateWork = workspace.work().size < 2 * A.nRows;
  bool const reallocateRWork = workspace.rwork().size < 24 * A.nRows;
  bool const reallocateIWork = workspace.iwork().size < 10 * A.nRows;

  if( reallocateWork || reallocateRWork || reallocateIWork )
  {
    OptimalSizeCalculation< std::complex< T > > optimalSizes;
    internal::heevr( MemorySpace::host, decompositionOptions, A, eigenValues, eigenVectors, support, optimalSizes, storageType, false );
    
    if( reallocateWork )
    {
      workspace.resizeWork( space, optimalSizes.optimalWorkSize() );
    }

    if( reallocateRWork )
    {
      workspace.resizeRWork( space, optimalSizes.optimalRWorkSize() );
    }

    if( reallocateIWork )
    {
      workspace.resizeIWork( space, optimalSizes.optimalIWorkSize() );
    }
  }

  return internal::heevr( space, decompositionOptions, A, eigenValues, eigenVectors, support, workspace, storageType, true );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// explicit instantiations.
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
template DenseInt heevr< float >(
  MemorySpace const space,
  EigenDecompositionOptions const decompositionOptions,
  Matrix< std::complex< float > > const & A,
  Vector< float > const & eigenValues,
  Matrix< std::complex< float > > const & eigenVectors,
  Vector< DenseInt > const & support,
  Workspace< std::complex< float > > & workspace,
  SymmetricMatrixStorageType const storageType );

////////////////////////////////////////////////////////////////////////////////////////////////////
template DenseInt heevr< double >(
  MemorySpace const space,
  EigenDecompositionOptions const decompositionOptions,
  Matrix< std::complex< double > > const & A,
  Vector< double > const & eigenValues,
  Matrix< std::complex< double > > const & eigenVectors,
  Vector< DenseInt > const & support,
  Workspace< std::complex< double > > & workspace,
  SymmetricMatrixStorageType const storageType );

} // namespace dense
} // namespace LvArray
