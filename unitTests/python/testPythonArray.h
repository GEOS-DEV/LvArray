#pragma once
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize and return a Numpy view of the global 1D-Array of ints.
 */
PyObject * initArray1dInt( int offset );

/**
 * Fetch a Numpy view of the global 1D-Array of ints.
 * Note this view may be invalidated by subsequent calls to `initArray1dInt`
 */
PyObject * getArray1dInt( void );

/**
 * Multiply the global SortedArray of ints
 * @param factor the multiplication factor
 */
void multiplyArray1d( int factor );

/**
 * Initialize and return a Numpy view of the global 4D-Array of doubles.
 */
PyObject * initArray4dDouble( int offset );

/**
 * Fetch a Numpy view of the global 4D-Array of doubles.
 * Note this view may be invalidated by subsequent calls to `initArray4dLongLong`
 */
PyObject * getArray4dDouble( void );

/**
 * Multiply the global 4D Array of doubles
 * @param factor the multiplication factor
 */
void multiplyArray4dDouble( int factor );

/**
 * Initialize and return a Numpy view of the global 2D-Array of chars.
 */
PyObject * initArray2dChar( int offset );

/**
 * Fetch a Numpy view of the global 2D-Array of chars.
 * Note this view may be invalidated by subsequent calls to `initArray4dLongLong`
 */
PyObject * getArray2dChar( void );

/**
 * Multiply the global 2D-Array of chars
 * @param factor the multiplication factor
 */
void multiplyArray2dChar( int factor );

#ifdef __cplusplus
}
#endif
