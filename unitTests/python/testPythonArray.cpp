#include "Array.hpp"
#include "MallocBuffer.hpp"
#include "python/numpyArrayView.hpp"
#include "testPythonArray.h"


#ifdef __cplusplus
extern "C" {
#endif

// global Array of ints
static LvArray::Array< long long, 1, RAJA::PERM_I, std::ptrdiff_t, LvArray::MallocBuffer > array1dInt( 30 );
// global 4D Array of doubles
static LvArray::Array< double, 4, RAJA::PERM_LKIJ, std::ptrdiff_t, LvArray::MallocBuffer > array4dDouble( 5, 6, 7, 8 );
// global 2D Array of chars
static LvArray::Array< char, 2, RAJA::PERM_JI, std::ptrdiff_t, LvArray::MallocBuffer > array2dChar( 3, 4 );

/**
 * Initialize and return a Numpy view of the global 1D-Array of ints.
 */
PyObject * initArray1dInt( int offset){
    forValuesInSlice( array1dInt.toSlice(), [&offset]( long long & value )
    {
        value = offset++;
    } );
    return getArray1dInt();
}

/**
 * Fetch a Numpy view of the global 1D-Array of ints.
 * Note this view may be invalidated by subsequent calls to `initArray1dInt`
 */
PyObject * getArray1dInt( void ){
    return LvArray::python::create( array1dInt.toView() );
}

/**
 * Multiply the global 1D Array of ints
 * @param factor the multiplication factor
 */
void multiplyArray1d( int factor ){
    for (long long& i : array1dInt)
    {
        i = factor * i;
    }
}

/**
 * Initialize and return a Numpy view of the global 4D-Array of doubles.
 */
PyObject * initArray4dDouble( int offset ){
    forValuesInSlice( array4dDouble.toSlice(), [&offset]( double & value )
    {
        value = offset++;
    } );
    return getArray4dDouble();
}

/**
 * Fetch a Numpy view of the global 4D-Array of doubles.
 * Note this view may be invalidated by subsequent calls to `initArray4dLongLong`
 */
PyObject * getArray4dDouble( void ){
    return LvArray::python::create( array4dDouble.toView() );
}

/**
 * Multiply the global 4D Array of doubles
 * @param factor the multiplication factor
 */
void multiplyArray4dDouble( int factor ){
    for (double& i : array4dDouble)
    {
        i = factor * i;
    }
}

/**
 * Initialize and return a Numpy view of the global 2D-Array of chars.
 */
PyObject * initArray2dChar( int offset ){
    forValuesInSlice( array2dChar.toSlice(), [&offset]( char & value )
    {
        value = offset++;
    } );
    return getArray2dChar();
}

/**
 * Fetch a Numpy view of the global 2D-Array of chars.
 * Note this view may be invalidated by subsequent calls to `initArray4dLongLong`
 */
PyObject * getArray2dChar( void ){
    return LvArray::python::create( array2dChar.toView() );
}

/**
 * Multiply the global 2D-Array of chars
 * @param factor the multiplication factor
 */
void multiplyArray2dChar( int factor ){
    for (char & i : array2dChar)
    {
        i = factor * i;
    }
}

#ifdef __cplusplus
}
#endif


