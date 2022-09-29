#include "eigenDecomposition.hpp"
#include "backendHelpers.hpp"

extern "C"
{

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_CHEEVR LVARRAY_LAPACK_FORTRAN_MANGLE( cheevr )
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
#define LVARRAY_ZHEEVR LVARRAY_LAPACK_FORTRAN_MANGLE( zheevr )
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

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_SGEEV LVARRAY_LAPACK_FORTRAN_MANGLE( sgeev )
void LVARRAY_SGEEV(
  char const * JOBVL,
  char const * JOBVR,
  LvArray::dense::DenseInt const * N,
  float * A,
  LvArray::dense::DenseInt const * LDA,
  float * WR,
  float * WI,
  float * VL,
  LvArray::dense::DenseInt const * LDVL,
  float * VR,
  LvArray::dense::DenseInt const * LDVR,
  float * WORK,
  LvArray::dense::DenseInt const * LWORK,
  LvArray::dense::DenseInt * INFO
);

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_DGEEV LVARRAY_LAPACK_FORTRAN_MANGLE( dgeev )
void LVARRAY_DGEEV(
  char const * JOBVL,
  char const * JOBVR,
  LvArray::dense::DenseInt const * N,
  double * A,
  LvArray::dense::DenseInt const * LDA,
  double * WR,
  double * WI,
  double * VL,
  LvArray::dense::DenseInt const * LDVL,
  double * VR,
  LvArray::dense::DenseInt const * LDVR,
  double * WORK,
  LvArray::dense::DenseInt const * LWORK,
  LvArray::dense::DenseInt * INFO
);

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_CGEEV LVARRAY_LAPACK_FORTRAN_MANGLE( cgeev )
void LVARRAY_CGEEV(
  char const * JOBVL,
  char const * JOBVR,
  LvArray::dense::DenseInt const * N,
  std::complex< float > * A,
  LvArray::dense::DenseInt const * LDA,
  std::complex< float > * W,
  std::complex< float > * VL,
  LvArray::dense::DenseInt const * LDVL,
  std::complex< float > * VR,
  LvArray::dense::DenseInt const * LDVR,
  std::complex< float > * WORK,
  LvArray::dense::DenseInt const * LWORK,
  float * RWORK,
  LvArray::dense::DenseInt * INFO
);

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_ZGEEV LVARRAY_LAPACK_FORTRAN_MANGLE( zgeev )
void LVARRAY_ZGEEV(
  char const * JOBVL,
  char const * JOBVR,
  LvArray::dense::DenseInt const * N,
  std::complex< double > * A,
  LvArray::dense::DenseInt const * LDA,
  std::complex< double > * W,
  std::complex< double > * VL,
  LvArray::dense::DenseInt const * LDVL,
  std::complex< double > * VR,
  LvArray::dense::DenseInt const * LDVR,
  std::complex< double > * WORK,
  LvArray::dense::DenseInt const * LWORK,
  double * RWORK,
  LvArray::dense::DenseInt * INFO
);


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
  BuiltInBackends const backend,
  EigenDecompositionOptions const decompositionOptions,
  Matrix< std::complex< T > > const & A,
  Vector< T > const & eigenvalues,
  Matrix< std::complex< T > > const & eigenvectors,
  Vector< DenseInt > const & support,
  Workspace< std::complex< T > > & workspace,
  SymmetricMatrixStorageType const storageType,
  bool const compute )
{
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
  if( decompositionOptions.range == EigenDecompositionOptions::BY_INDEX )
  {
    LVARRAY_ERROR_IF_GT( IU, N );
    maxEigenvaluesToFind = IU - IL + 1;
  }

  LVARRAY_ERROR_IF_LT( eigenvalues.size, maxEigenvaluesToFind );

  DenseInt const ABSTOL = decompositionOptions.abstol;
  DenseInt M = 0;

  if( decompositionOptions.type == EigenDecompositionOptions::EIGENVALUES_AND_RIGHT_VECTORS )
  {
    LVARRAY_ERROR_IF_NE( eigenvectors.nRows, N );
    LVARRAY_ERROR_IF_LT( eigenvectors.nCols, maxEigenvaluesToFind );
  }

  DenseInt const LDZ = std::max( 1, eigenvectors.stride );

  if( decompositionOptions.range == EigenDecompositionOptions::ALL ||
      ( decompositionOptions.range == EigenDecompositionOptions::BY_INDEX &&
        maxEigenvaluesToFind == N ) )
  {
    LVARRAY_ERROR_IF_LT( support.size, 2 * maxEigenvaluesToFind );
  }

  DenseInt const LWORK = compute ? workspace.work().size : -1;
  DenseInt const LRWORK = compute ? workspace.rwork().size : -1;
  DenseInt const LIWORK = compute ? workspace.iwork().size : -1;

  DenseInt INFO = 0;

  // With C++ 17 we can remove the reinterpret_cast with constexpr if.
  if( backend == BuiltInBackends::LAPACK )
  {
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
        reinterpret_cast< float * >( eigenvalues.data ),
        reinterpret_cast< std::complex< float > * >( eigenvectors.data ),
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
        reinterpret_cast< double * >( eigenvalues.data ),
        reinterpret_cast< std::complex< double > * >( eigenvectors.data ),
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
  }
#if defined( LVARRAY_USE_MAGMA )
  else if( backend == BuiltInBackends::MAGMA )
  {
    if( std::is_same< T, float >::value )
    {
      magma_cheevr(
        magma_vec_const( *JOBZ ),
        magma_range_const( *RANGE ),
        magma_uplo_const( *UPLO ),
        N,
        reinterpret_cast< magmaFloatComplex * >( A.data ),
        LDA,
        VL,
        VU,
        IL,
        IU,
        ABSTOL,
        &M,
        reinterpret_cast< float * >( eigenvalues.data ),
        reinterpret_cast< magmaFloatComplex * >( eigenvectors.data ),
        LDZ,
        support.data,
        reinterpret_cast< magmaFloatComplex * >( workspace.work().data ),
        LWORK,
        reinterpret_cast< float * >( workspace.rwork().data ),
        LRWORK,
        workspace.iwork().data,
        LIWORK,
        &INFO );
    }
    else
    {
      magma_zheevr(
        magma_vec_const( *JOBZ ),
        magma_range_const( *RANGE ),
        magma_uplo_const( *UPLO ),
        N,
        reinterpret_cast< magmaDoubleComplex * >( A.data ),
        LDA,
        VL,
        VU,
        IL,
        IU,
        ABSTOL,
        &M,
        reinterpret_cast< double * >( eigenvalues.data ),
        reinterpret_cast< magmaDoubleComplex * >( eigenvectors.data ),
        LDZ,
        support.data,
        reinterpret_cast< magmaDoubleComplex * >( workspace.work().data ),
        LWORK,
        reinterpret_cast< double * >( workspace.rwork().data ),
        LRWORK,
        workspace.iwork().data,
        LIWORK,
        &INFO );
    }
  }
  else if( backend == BuiltInBackends::MAGMA_GPU )
  {
    int LDWA = N;
    int LDWZ = N;

    if( compute )
    {
      workspace.resizeWork2( MemorySpace::host, LDWA * N );
      workspace.resizeWork3( MemorySpace::host, LDWZ * maxEigenvaluesToFind );
    }

    if( std::is_same< T, float >::value )
    {
      magma_cheevr_gpu(
        magma_vec_const( *JOBZ ),
        magma_range_const( *RANGE ),
        magma_uplo_const( *UPLO ),
        N,
        reinterpret_cast< magmaFloatComplex * >( A.data ),
        LDA,
        VL,
        VU,
        IL,
        IU,
        ABSTOL,
        &M,
        reinterpret_cast< float * >( eigenvalues.data ),
        reinterpret_cast< magmaFloatComplex * >( eigenvectors.data ),
        LDZ,
        support.data,
        reinterpret_cast< magmaFloatComplex * >( workspace.work2().data ),
        LDWA,
        reinterpret_cast< magmaFloatComplex * >( workspace.work3().data ),
        LDWZ,
        reinterpret_cast< magmaFloatComplex * >( workspace.work().data ),
        LWORK,
        reinterpret_cast< float * >( workspace.rwork().data ),
        LRWORK,
        workspace.iwork().data,
        LIWORK,
        &INFO );
    }
    else
    {
      magma_zheevr_gpu(
        magma_vec_const( *JOBZ ),
        magma_range_const( *RANGE ),
        magma_uplo_const( *UPLO ),
        N,
        reinterpret_cast< magmaDoubleComplex * >( A.data ),
        LDA,
        VL,
        VU,
        IL,
        IU,
        ABSTOL,
        &M,
        reinterpret_cast< double * >( eigenvalues.data ),
        reinterpret_cast< magmaDoubleComplex * >( eigenvectors.data ),
        LDZ,
        support.data,
        reinterpret_cast< magmaDoubleComplex * >( workspace.work2().data ),
        LDWA,
        reinterpret_cast< magmaDoubleComplex * >( workspace.work3().data ),
        LDWZ,
        reinterpret_cast< magmaDoubleComplex * >( workspace.work().data ),
        LWORK,
        reinterpret_cast< double * >( workspace.rwork().data ),
        LRWORK,
        workspace.iwork().data,
        LIWORK,
        &INFO );
    }
  }
#endif
  else
  {
    LVARRAY_ERROR( "Unknown built in backend: " << static_cast< int >( backend ) );
  }

  LVARRAY_ERROR_IF_NE( INFO, 0 );

  return M;
}

/**
 *
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
  Workspace< T > & workspace,
  bool const compute )
{
  char const * const JOBVL = decompositionOptions.leftArg();
  char const * const JOBVR = decompositionOptions.rightArg();
  DenseInt const N = A.nCols;
  DenseInt const LDA = A.stride;
  DenseInt LDVL = std::max( 1, leftEigenvectors.stride );
  DenseInt LDVR = std::max( 1, rightEigenvectors.stride );

  DenseInt const LWORK = compute ? workspace.work().size : -1;
  DenseInt INFO = 0;

  if( backend == BuiltInBackends::LAPACK )
  {
    if( std::is_same< T, float >::value )
    {
      LVARRAY_SGEEV(
        JOBVL,
        JOBVR,
        &N,
        reinterpret_cast< float * >( A.data ),
        &LDA,
        reinterpret_cast< float * >( eigenValuesReal.data ),
        reinterpret_cast< float * >( eigenValuesImag.data ),
        reinterpret_cast< float * >( leftEigenvectors.data ),
        &LDVL,
        reinterpret_cast< float * >( rightEigenvectors.data ),
        &LDVR,
        reinterpret_cast< float * >( workspace.work().data ),
        &LWORK,
        &INFO );
    }
    if( std::is_same< T, double >::value )
    {
      LVARRAY_DGEEV(
        JOBVL,
        JOBVR,
        &N,
        reinterpret_cast< double * >( A.data ),
        &LDA,
        reinterpret_cast< double * >( eigenValuesReal.data ),
        reinterpret_cast< double * >( eigenValuesImag.data ),
        reinterpret_cast< double * >( leftEigenvectors.data ),
        &LDVL,
        reinterpret_cast< double * >( rightEigenvectors.data ),
        &LDVR,
        reinterpret_cast< double * >( workspace.work().data ),
        &LWORK,
        &INFO );
    }
    if( std::is_same< T, std::complex< float > >::value )
    {
      LVARRAY_CGEEV(
        JOBVL,
        JOBVR,
        &N,
        reinterpret_cast< std::complex< float > * >( A.data ),
        &LDA,
        reinterpret_cast< std::complex< float > * >( eigenValuesReal.data ),
        reinterpret_cast< std::complex< float > * >( leftEigenvectors.data ),
        &LDVL,
        reinterpret_cast< std::complex< float > * >( rightEigenvectors.data ),
        &LDVR,
        reinterpret_cast< std::complex< float > * >( workspace.work().data ),
        &LWORK,
        reinterpret_cast< float * >( workspace.rwork().data ),
        &INFO );
    }
    if( std::is_same< T, std::complex< double > >::value )
    {
      LVARRAY_ZGEEV(
        JOBVL,
        JOBVR,
        &N,
        reinterpret_cast< std::complex< double > * >( A.data ),
        &LDA,
        reinterpret_cast< std::complex< double > * >( eigenValuesReal.data ),
        reinterpret_cast< std::complex< double > * >( leftEigenvectors.data ),
        &LDVL,
        reinterpret_cast< std::complex< double > * >( rightEigenvectors.data ),
        &LDVR,
        reinterpret_cast< std::complex< double > * >( workspace.work().data ),
        &LWORK,
        reinterpret_cast< double * >( workspace.rwork().data ),
        &INFO );
    }
  }
  else
  {
    LVARRAY_ERROR( "Unknown built in backend: " << static_cast< int >( backend ) );
  }

  LVARRAY_ERROR_IF_NE( INFO, 0 );
}

} // namespace internal

// TODO(corbett5): Add support for symmetric matrices.
////////////////////////////////////////////////////////////////////////////////////////////////////
template< typename T >
DenseInt heevr(
  BuiltInBackends const backend,
  EigenDecompositionOptions const decompositionOptions,
  Matrix< std::complex< T > > const & A,
  Vector< T > const & eigenvalues,
  Matrix< std::complex< T > > const & eigenvectors,
  Vector< DenseInt > const & support,
  Workspace< std::complex< T > > & workspace,
  SymmetricMatrixStorageType const storageType )
{
  LVARRAY_ERROR_IF_EQ( decompositionOptions.type, EigenDecompositionOptions::EIGENVALUES_AND_LEFT_VECTORS );
  LVARRAY_ERROR_IF_EQ( decompositionOptions.type, EigenDecompositionOptions::EIGENVALUES_AND_LEFT_AND_RIGHT_VECTORS );

  LVARRAY_ERROR_IF( !A.isSquare(), "The matrix A must be square." );

  // TODO(corbett5): I think we can support row major by simply complex-conjugating all entries.
  // I'm not sure exactly how this would work for the eigenvectors though.
  LVARRAY_ERROR_IF( !A.isColumnMajor, "Row major is not yet supported." );
  LVARRAY_ERROR_IF( !eigenvectors.isColumnMajor, "Row major is not yet supported." );

  bool const reallocateWork = workspace.work().size < 2 * A.nRows;
  bool const reallocateRWork = workspace.rwork().size < 24 * A.nRows;
  bool const reallocateIWork = workspace.iwork().size < 10 * A.nRows;

  if( reallocateWork || reallocateRWork || reallocateIWork )
  {
    OptimalSizeCalculation< std::complex< T > > optimalSizes;
    internal::heevr(
      backend,
      decompositionOptions,
      A,
      eigenvalues,
      eigenvectors,
      support,
      optimalSizes,
      storageType,
      false );
    
    if( reallocateWork )
    {
      workspace.resizeWork( MemorySpace::host, optimalSizes.optimalWorkSize() );
    }

    if( reallocateRWork )
    {
      workspace.resizeRWork( MemorySpace::host, optimalSizes.optimalRWorkSize() );
    }

    if( reallocateIWork )
    {
      workspace.resizeIWork( MemorySpace::host, optimalSizes.optimalIWorkSize() );
    }
  }

  return internal::heevr(
    backend,
    decompositionOptions,
    A,
    eigenvalues,
    eigenvectors,
    support,
    workspace,
    storageType,
    true );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template< typename T >
void geev(
  BuiltInBackends const backend,
  EigenDecompositionOptions const decompositionOptions,
  Matrix< T > const & A,
  Vector< T > const & eigenValuesReal,
  Vector< T > const & eigenValuesImag,
  Matrix< T > const & leftEigenvectors,
  Matrix< T > const & rightEigenvectors,
  Workspace< T > & workspace )
{
  // TODO(corbett5): I think we can support row major by switching between left and right eigenvectors.
  LVARRAY_ERROR_IF( !A.isColumnMajor, "Row major is not yet supported." );

  LVARRAY_ERROR_IF( !A.isSquare(), "The matrix A must be square." );

  LVARRAY_ERROR_IF_NE( decompositionOptions.range, EigenDecompositionOptions::ALL );

  LVARRAY_ERROR_IF_NE( eigenValuesReal.size, A.nCols );
  if( !IsComplex< T > )
  {
    LVARRAY_ERROR_IF_NE( eigenValuesImag.size, A.nCols );
  }

  if( decompositionOptions.computeLeftEigenvectors() )
  {
    LVARRAY_ERROR_IF( !leftEigenvectors.isColumnMajor, "Row major is not yet supported." );
    LVARRAY_ERROR_IF_NE( leftEigenvectors.nRows, A.nCols );
    LVARRAY_ERROR_IF_LT( leftEigenvectors.nCols, A.nCols );
  }
  if( decompositionOptions.computeRightEigenvectors() )
  {
    LVARRAY_ERROR_IF( !rightEigenvectors.isColumnMajor, "Row major is not yet supported." );
    LVARRAY_ERROR_IF_NE( rightEigenvectors.nRows, A.nCols );
    LVARRAY_ERROR_IF_LT( rightEigenvectors.nCols, A.nCols );
  }
  
  bool reallocateWork;
  if( !IsComplex< T > )
  {
    bool const computeVectors = decompositionOptions.type != EigenDecompositionOptions::EIGENVALUES;
    reallocateWork = workspace.work().size < (3 + computeVectors) * A.nRows;
  }
  else
  {
    reallocateWork = workspace.work().size < 2 * A.nRows;
    workspace.resizeRWork( MemorySpace::host, 2 * A.nCols );
  }

  if( reallocateWork )
  {
    OptimalSizeCalculation< T > optimalSizes;
    internal::geev(
      backend,
      decompositionOptions,
      A,
      eigenValuesReal,
      eigenValuesImag,
      leftEigenvectors,
      rightEigenvectors,
      optimalSizes,
      false );
    
    if( reallocateWork )
    {
      workspace.resizeWork( MemorySpace::host, optimalSizes.optimalWorkSize() );
    }
  }

  return internal::geev(
    backend,
    decompositionOptions,
    A,
    eigenValuesReal,
    eigenValuesImag,
    leftEigenvectors,
    rightEigenvectors,
    workspace,
    true );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template< typename T >
void geev(
  BuiltInBackends const backend,
  EigenDecompositionOptions const decompositionOptions,
  Matrix< std::complex< T > > const & A,
  Vector< std::complex< T > > const & eigenValues,
  Matrix< std::complex< T > > const & leftEigenvectors,
  Matrix< std::complex< T > > const & rightEigenvectors,
  Workspace< std::complex< T > > & workspace )
{
  return geev(
    backend,
    decompositionOptions,
    A,
    eigenValues,
    Vector< std::complex< T > > {},
    leftEigenvectors,
    rightEigenvectors,
    workspace );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// explicit instantiations.
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
template DenseInt heevr< float >(
  BuiltInBackends const backend,
  EigenDecompositionOptions const decompositionOptions,
  Matrix< std::complex< float > > const & A,
  Vector< float > const & eigenvalues,
  Matrix< std::complex< float > > const & eigenvectors,
  Vector< DenseInt > const & support,
  Workspace< std::complex< float > > & workspace,
  SymmetricMatrixStorageType const storageType );

////////////////////////////////////////////////////////////////////////////////////////////////////
template DenseInt heevr< double >(
  BuiltInBackends const backend,
  EigenDecompositionOptions const decompositionOptions,
  Matrix< std::complex< double > > const & A,
  Vector< double > const & eigenvalues,
  Matrix< std::complex< double > > const & eigenvectors,
  Vector< DenseInt > const & support,
  Workspace< std::complex< double > > & workspace,
  SymmetricMatrixStorageType const storageType );

////////////////////////////////////////////////////////////////////////////////////////////////////
template void geev< float >(
  BuiltInBackends const backend,
  EigenDecompositionOptions const decompositionOptions,
  Matrix< float > const & A,
  Vector< float > const & eigenValuesReal,
  Vector< float > const & eigenValuesImag,
  Matrix< float > const & leftEigenvectors,
  Matrix< float > const & rightEigenvectors,
  Workspace< float > & workspace );

////////////////////////////////////////////////////////////////////////////////////////////////////
template void geev< double >(
  BuiltInBackends const backend,
  EigenDecompositionOptions const decompositionOptions,
  Matrix< double > const & A,
  Vector< double > const & eigenValuesReal,
  Vector< double > const & eigenValuesImag,
  Matrix< double > const & leftEigenvectors,
  Matrix< double > const & rightEigenvectors,
  Workspace< double > & workspace );

////////////////////////////////////////////////////////////////////////////////////////////////////
template void geev< float >(
  BuiltInBackends const backend,
  EigenDecompositionOptions const decompositionOptions,
  Matrix< std::complex< float > > const & A,
  Vector< ComplexVersion< std::complex< float > > > const & eigenValues,
  Matrix< ComplexVersion< std::complex< float > > > const & leftEigenvectors,
  Matrix< ComplexVersion< std::complex< float > > > const & rightEigenvectors,
  Workspace< std::complex< float > > & workspace );

////////////////////////////////////////////////////////////////////////////////////////////////////
template void geev< double >(
  BuiltInBackends const backend,
  EigenDecompositionOptions const decompositionOptions,
  Matrix< std::complex< double > > const & A,
  Vector< ComplexVersion< std::complex< double > > > const & eigenValues,
  Matrix< ComplexVersion< std::complex< double > > > const & leftEigenvectors,
  Matrix< ComplexVersion< std::complex< double > > > const & rightEigenvectors,
  Workspace< std::complex< double > > & workspace );

} // namespace dense
} // namespace LvArray
