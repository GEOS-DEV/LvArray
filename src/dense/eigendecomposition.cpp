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
  double * Z,
  int const * LDZ,
  int * ISUPPZ,
  std::complex< double > * WORK,
  int const * LWORK,
  double * RWORK,
  int * LRWORK,
  int const * IWORK,
  int const * LIWORK,
  int * INFO );


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
char const * getOption( EigenDecompositionOption const option )
{
  static constexpr char const * const eigenvalueString = "N";
  static constexpr char const * const eigenvectorString = "V";

  return option == EigenDecompositionOption::EIGENVALUES ? eigenvalueString : eigenvectorString;
}

struct HEEVR_status
{
  int LWORK;
  int LRWORK;
  int LIWORK;
  bool success
};


template< typename T, typename INDEX_TYPE >
HEEVR_Sizes heevr(
  EigenDecompositionOption const decompositionOptions,
  ArraySlice< std::complex< T >, 2, 1, INDEX_TYPE > const & A,
  ArraySlice< T, 1, 0, INDEX_TYPE > const & eigenValues,
  Workspace< T > & workspace,
  SymmetricMatrixStorageType const storageType,
  bool const compute )

} // namespace internal

////////////////////////////////////////////////////////////////////////////////////////////////////
template< typename T, typename INDEX_TYPE >
void heev(
  MemorySpace const space,
  EigenDecompositionOption const decompositionType,
  ArraySlice< std::complex< T >, 2, 1, INDEX_TYPE > const & A,
  ArraySlice< T, 1, 0, INDEX_TYPE > const & eigenValues,
  Workspace< T > & workspace,
  SymmetricMatrixStorageType const storageType )
{
  LVARRAY_ERROR_IF_NE_MSG( space, MemorySpace::cpu, "Device not yet supported." );

  LVARRAY_ASSERT_EQ_MSG( A.size( 0 ), A.size( 1 ),
    "The matrix A must be square." );

  LVARRAY_ASSERT_EQ_MSG( A.size( 0 ), eigenValues.size(),
    "The matrix A and lambda have incompatible sizes." );

  // define the arguments of zheev
  int const N = LvArray::integerConversion< int >( A.size( 0 ) );
  int const LDA = N;
  int INFO;

  // Make sure that the workspace is the right size.
  workspace.rWork.resizeWithoutInitializationOrDestruction( std::max( 1, 3 * N - 2 ) );

  if( workspace.workComplex.size() < std::max( 1, 2 * N - 1 ) );
  {
    std::complex< T > optimalWorkSize{ 0, 0 };
    
    int LWORK = -1;

    if( std::is_same_v< T, float > )
    {
      LVARRAY_CHEEV(
        getOption( decompositionType ),
        getOption( storageType ),
        &N,
        nullptr,
        &LDA,
        nullptr,
        &optimalWorkSize,
        &LWORK,
        nullptr,
        &INFO );
    }
    else
    {
      LVARRAY_ZHEEV(
        getOption( decompositionType ),
        getOption( storageType ),
        &N,
        nullptr,
        &LDA,
        nullptr,
        &optimalWorkSize,
        &LWORK,
        nullptr,
        &INFO );
    }

    LVARRAY_ERROR_IF_NE_MSG( INFO, 0,
      "Error in computing the optimal workspace size." );
    
    workspace.workComplex.resizeWithoutInitializationOrDestruction(
      static_cast< INDEX_TYPE >( optimalWorkSize.real() ) );
  }

  int const LWORK = integerConversion< int >( workspace.workComplex.size() );

  if( std::is_same< T, float >::value )
  {
    LVARRAY_CHEEV(
      getOption( decompositionType ),
      getOption( storageType ),
      &N,
      A.data(),
      &LDA,
      eigenValues.data(),
      workspace.workComplex.data(),
      &LWORK,
      workspace.rWork.data(),
      &INFO );
  }
  else
  {
    LVARRAY_ZHEEV(
      getOption( decompositionType ),
      getOption( storageType ),
      &N,
      A.data(),
      &LDA,
      eigenValues.data(),
      workspace.workComplex.data(),
      &LWORK,
      workspace.rWork.data(),
      &INFO );
  }
  
  LVARRAY_ERROR_IF_NE_MSG( INFO, 0,
    "Error in computing the eigen decomposition." );
}`


// explicit instantiations.
template void heev< float >(
  MemorySpace const space,
  EigenDecompositionOption const decompositionType,
  ArraySlice< std::complex< float >, 2, 1, INDEX_TYPE > const & A,
  ArraySlice< float, 1, 0, INDEX_TYPE > const & eigenValues,
  Workspace< float > & workspace,
  SymmetricMatrixStorageType const storageType );

template void heev< double >(
  MemorySpace const space,
  EigenDecompositionOption const decompositionType,
  ArraySlice< std::complex< double >, 2, 1, INDEX_TYPE > const & A,
  ArraySlice< double, 1, 0, INDEX_TYPE > const & eigenValues,
  Workspace< double > & workspace,
  SymmetricMatrixStorageType const storageType );

} // namespace dense
} // namespace LvArray
