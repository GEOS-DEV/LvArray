#include "SortedArray.hpp"
#include "MallocBuffer.hpp"
#include "python/numpySortedArrayView.hpp"
#include "testPythonSortedArray.h"

#ifdef __cplusplus
extern "C" {
#endif

// global SortedArray of ints
LvArray::SortedArray< int, std::ptrdiff_t, LvArray::MallocBuffer > sortedArrayInt;

/**
 * Initialize and return a Numpy view of the global SortedArray of ints.
 */
PyObject * initSortedArrayInt( int start, int stop ){
    sortedArrayInt.clear();
    for ( int i = start; i < stop; ++i )
    {
        sortedArrayInt.insert( i );
    }
    return getSortedArrayInt();
}

/**
 * Fetch a Numpy view of the global SortedArray of ints.
 * Note this view may be invalidated by subsequent calls to `initSortedArrayInt`
 */
PyObject * getSortedArrayInt( void ){
    return LvArray::python::create( sortedArrayInt.toView() );
}

/**
 * Multiply the global SortedArray of ints by a factor
 */
void multiplySortedArrayInt( int factor ){
    LvArray::SortedArray< int, std::ptrdiff_t, LvArray::MallocBuffer > newSortedArray;
    for (int i : sortedArrayInt)
    {
        newSortedArray.insert( factor * i );
    }
    sortedArrayInt = newSortedArray;
}

#ifdef __cplusplus
}
#endif
