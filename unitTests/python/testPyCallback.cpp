#define PY_SSIZE_T_CLEAN

#include <Python.h>

#include "python/PyFunc.hpp"
#include "python/PySortedArray.hpp"
#include "python/python.hpp"
#include "MallocBuffer.hpp"

static LvArray::SortedArray< long, std::ptrdiff_t, LvArray::MallocBuffer > sortedArrayOfLongs;

/**
 * Clear and initialize the global 1D Array with values beginning at
 * an integer offset.
 * NOTE: this may invalidate previous views of the array!
 */
static PyObject *
call( PyObject *self, PyObject *args )
{
  LVARRAY_UNUSED_VARIABLE( args );
  LvArray::python::PyObjectRef<> callback { PyObject_GetAttrString( self, "callback" ) };
  if( callback == nullptr )
  {
    return nullptr;
  }
  if( !PyCallable_Check( callback ) )
  {
    PyErr_SetString( PyExc_TypeError, "callback attribute must be callable" ); return nullptr;
  }
  //std::function< void( LvArray::SortedArray< long, std::ptrdiff_t, LvArray::MallocBuffer > & ) > func =
  LvArray::python::PythonFunction< LvArray::SortedArray< long, std::ptrdiff_t, LvArray::MallocBuffer > & > func{ callback };
  try
  {
    func( sortedArrayOfLongs );
  } catch( const LvArray::python::PythonError & )
  {
    return nullptr;
  }
  Py_RETURN_NONE;
}

/**
 * Array of functions and docstrings to export to Python
 */
static PyMethodDef ModuleFuncs[] = {
  {"call", call, METH_NOARGS, ""},
  {NULL, NULL, 0, NULL}          /* Sentinel */
};

/**
 * Initialize the module object for Python with the exported functions
 */
static struct PyModuleDef testPyCallbackModule = {
  PyModuleDef_HEAD_INIT,
  "testPyCallback",     /* name of module */
  "Module for testing python callbacks",   /* module documentation, may be NULL */
  -1,         /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
  ModuleFuncs,
  NULL,
  NULL,
  NULL,
  NULL,
};

PyMODINIT_FUNC
PyInit_testPyCallback( void )
{
  LvArray::python::PyObjectRef<> module = PyModule_Create( &testPyCallbackModule );

  if( !LvArray::python::addPyLvArrayModule( module ) )
  {
    return nullptr;
  }
  Py_INCREF( Py_None );
  if( PyModule_AddObject( module, "callback", Py_None ) < 0 )
  {
    Py_DECREF( Py_None );
    return nullptr;
  }
  return module.release();
}
