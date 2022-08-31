#include "eigendecomposition.hpp"

/// This macro provide a flexible interface for Fortran naming convention for compiled objects
// #ifdef FORTRAN_MANGLE_NO_UNDERSCORE
#define FORTRAN_MANGLE( name ) name
// #else
// #define FORTRAN_MANGLE( name ) name ## _
// #endif

extern "C"
{

#define LVARRAY_CHEEV FORTRAN_MANGLE( cheev )
void LVARRAY_CHEEV(
  char const * JOBZ,
  char const * UPLO,
  int const * N,
  std::complex< float > * A,
  int const * LDA,
  float * W,
  std::complex< float > * WORK,
  int const * LWORK,
  float const * RWORK,
  int * INFO
);

#define LVARRAY_ZHEEV FORTRAN_MANGLE( zheev )
void LVARRAY_ZHEEV(
  char const * JOBZ,
  char const * UPLO,
  int const * N,
  std::complex< double > * A,
  int const * LDA,
  double * W,
  std::complex< double > * WORK,
  int const * LWORK,
  double const * RWORK,
  int * INFO );

#define LVARRAY_ZHEEVR FORTRAN_MANGLE( zheevr )
void LVARRAY_ZHEEVR( 
  char const * JOBZ,
  char const * RANGE,
  char const * UPLO,
  int const * N,
  std::complex< double > * A,
  int const * LDA,
  double const * VL,
  double const * VU,
  int const * IL,
  int const * IU,
  double const * ABSTOL,
  int * M,
  double * W,
  std::complex< double > * Z,
  int const * LDZ,
  int * ISUPPZ,
  std::complex< double > * WORK,
  int const * LWORK,
  double * RWORK,
  int const * LRWORK,
  int * IWORK,
  int const * LIWORK,
  int * INFO );


} // extern "C"

namespace LvArray
{
namespace dense
{
namespace internal
{

template< typename T >
int heevr(
  MemorySpace const space,
  EigenDecompositionOptions const decompositionOptions,
  Matrix< std::complex< T > > const & A,
  Vector< T > const & eigenValues,
  Matrix< std::complex< T > > const & eigenVectors,
  Vector< int > const & support,
  Workspace< std::complex< T > > & workspace,
  SymmetricMatrixStorageType const storageType,
  bool const compute )
{
  LVARRAY_ERROR_IF_NE_MSG( space, MemorySpace::host, "Device not yet supported." );

  LVARRAY_ERROR_IF( !A.isSquare(), "The matrix A must be square." );

  char const * const JOBZ = decompositionOptions.typeArg();
  char const * const RANGE = decompositionOptions.rangeArg();
  char const * const UPLO = getOption( storageType );
  int const N = integerConversion< int >( A.nCols );
  int const LDA = A.stride;

  T const VL = decompositionOptions.rangeMin;
  T const VU = decompositionOptions.rangeMax;

  if( decompositionOptions.range == EigenDecompositionOptions::Range::IN_INTERVAL )
  {
    LVARRAY_ERROR_IF_GE( VL, VU );
  }

  int maxEigenvaluesToFind = N;
  int const IL = decompositionOptions.indexMin;
  int const IU = decompositionOptions.indexMax;
  if( decompositionOptions.range == EigenDecompositionOptions::Range::BY_INDEX )
  {
    LVARRAY_ERROR_IF_LT( IL, 1 );
    LVARRAY_ERROR_IF_GT( IU, N );
    LVARRAY_ERROR_IF_GT( IL, IU );

    maxEigenvaluesToFind = IU - IL + 1;
  }

  LVARRAY_ERROR_IF_LT( eigenValues.n, maxEigenvaluesToFind );

  int const ABSTOL = decompositionOptions.abstol;
  int M = 0;

  if( decompositionOptions.type == EigenDecompositionOptions::Type::EIGENVALUES_AND_VECTORS )
  {
    LVARRAY_ERROR_IF_NE( eigenVectors.nRows, N );
    LVARRAY_ERROR_IF_LT( eigenVectors.nCols, maxEigenvaluesToFind );
  }

  int const LDZ = eigenVectors.stride;

  // TODO: check ISUPPZ

  int const LWORK = compute ? integerConversion< int >( workspace.work.size() ) : -1;
  int const LRWORK = integerConversion< int >( workspace.rwork.size() );
  int const LIWORK = integerConversion< int >( workspace.iwork.size() );

  int INFO = 0;

  // With C++ 17 we can remove the reinterpret_cast with constexpr if.
  if( std::is_same< T, float >::value )
  {
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
      reinterpret_cast< std::complex< double > * >( workspace.work.data() ),
      &LWORK,
      reinterpret_cast< double * >( workspace.rwork.data() ),
      &LRWORK,
      workspace.iwork.data(),
      &LIWORK,
      &INFO );
  }

  LVARRAY_ERROR_IF_NE( INFO, 0 );

  return M;
}

} // namespace internal

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T >
int heevr(
  MemorySpace const space,
  EigenDecompositionOptions const decompositionOptions,
  Matrix< std::complex< T > > const & A,
  Vector< T > const & eigenValues,
  Matrix< std::complex< T > > const & eigenVectors,
  Vector< int > const & support,
  Workspace< std::complex< T > > & workspace,
  SymmetricMatrixStorageType const storageType )
{
  bool const reallocateWork = workspace.work.size() < 2 * A.nRows;
  bool const reallocateRWork = workspace.rwork.size() < 24 * A.nRows;
  bool const reallocateIWork = workspace.iwork.size() < 10 * A.nRows;

  if( reallocateWork || reallocateRWork || reallocateIWork )
  {
    Workspace< std::complex< T > > optimalSizes( 1 );
    internal::heevr( MemorySpace::host, decompositionOptions, A, eigenValues, eigenVectors, support, optimalSizes, storageType, false );
    
    if( reallocateWork )
    {
      workspace.work.resizeWithoutInitializationOrDestruction( space, static_cast< std::ptrdiff_t >( optimalSizes.work[ 0 ].real() ) );
    }

    if( reallocateRWork )
    {
      workspace.rwork.resizeWithoutInitializationOrDestruction( space, static_cast< std::ptrdiff_t >( optimalSizes.rwork[ 0 ] ) );
    }

    if( reallocateIWork )
    {
      workspace.rwork.resizeWithoutInitializationOrDestruction( space, optimalSizes.iwork[ 0 ] );
    }
  }

  return internal::heevr( space, decompositionOptions, A, eigenValues, eigenVectors, support, workspace, storageType, true );
}

// explicit instantiations.
template int heevr< float >(
  MemorySpace const space,
  EigenDecompositionOptions const decompositionOptions,
  Matrix< std::complex< float > > const & A,
  Vector< float > const & eigenValues,
  Matrix< std::complex< float > > const & eigenVectors,
  Vector< int > const & support,
  Workspace< std::complex< float > > & workspace,
  SymmetricMatrixStorageType const storageType );

// template int heevr< double >(
//   MemorySpace const space,
//   EigenDecompositionOptions const decompositionOptions,
//   Matrix< std::complex< double > > const & A,
//   Vector< double > const & eigenValues,
//   Matrix< std::complex< double > > const & eigenVectors,
//   Vector< int > const & support,
//   Workspace< std::complex< double > > & workspace,
//   SymmetricMatrixStorageType const storageType );

} // namespace dense
} // namespace LvArray
