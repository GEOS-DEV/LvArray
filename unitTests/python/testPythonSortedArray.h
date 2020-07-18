#pragma once
#define PY_SSIZE_T_CLEAN
#include <Python.h>


#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize and return a Numpy view of the global SortedArray of ints.
 */
PyObject * initSortedArrayInt( int start, int stop );

/**
 * Fetch a Numpy view of the global SortedArray of ints.
 * Note this view may be invalidated by subsequent calls to `initSortedArrayInt`
 */
PyObject * getSortedArrayInt( void );

/**
 * Multiply the global SortedArray of ints
 * @param factor the multiplication factor
 */
void multiplySortedArrayInt( int factor );

#ifdef __cplusplus
}
#endif
