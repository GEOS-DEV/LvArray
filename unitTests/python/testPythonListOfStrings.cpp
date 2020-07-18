#define PY_SSIZE_T_CLEAN
#define ARR_SIZE 10

#include "Array.hpp"
#include "MallocBuffer.hpp"
#include "python/pythonHelpers.hpp"
#include "python/PyArray.hpp"
#include "python/python.hpp"

#include <Python.h>

static LvArray::Array< std::string, 1, RAJA::PERM_I, long long, LvArray::MallocBuffer > arr( ARR_SIZE );

static std::vector< std::string > vec( ARR_SIZE );

/**
 * Fetch the global 1D array of ints and return a numpy view of it.
 */
static PyObject *
getarray( PyObject *self, PyObject *args )
{
  LVARRAY_UNUSED_VARIABLE( self );
  LVARRAY_UNUSED_VARIABLE( args );
  return LvArray::python::create( arr );
}

/**
 * Clear and initialize the global 1D Array with values beginning at
 * an integer offset.
 * NOTE: this may invalidate previous views of the array!
 */
static PyObject *
setarray( PyObject *self, PyObject *args )
{
  LVARRAY_UNUSED_VARIABLE( self );
  char const * strptr;
  if( !PyArg_ParseTuple( args, "s", &strptr ) )
  {
    return NULL;
  }
  arr.clear();
  for( long long i = 0; i < ARR_SIZE; ++i )
  {
    arr.emplace_back( strptr );
  }
  return getarray( nullptr, nullptr );
}

/**
 * Fetch the global 1D array of ints and return a numpy view of it.
 */
static PyObject *
getvector( PyObject *self, PyObject *args )
{
  LVARRAY_UNUSED_VARIABLE( self );
  LVARRAY_UNUSED_VARIABLE( args );
  return LvArray::python::create( vec );
}

/**
 * Clear and initialize the global 1D Array with values beginning at
 * an integer offset.
 * NOTE: this may invalidate previous views of the array!
 */
static PyObject *
setvector( PyObject *self, PyObject *args )
{
  LVARRAY_UNUSED_VARIABLE( self );
  char const * strptr;
  if( !PyArg_ParseTuple( args, "s", &strptr ) )
  {
    return NULL;
  }
  vec.clear();
  for( long long i = 0; i < ARR_SIZE; ++i )
  {
    vec.push_back( std::string( strptr ) );
  }
  return getvector( nullptr, nullptr );
}

/**
 * Array of functions and docstrings to export to Python
 */
static PyMethodDef ListOfStringsFunc[] = {
  {"getarray", getarray, METH_NOARGS,
   ""},
  {"setarray", setarray, METH_VARARGS,
   ""},
  {"getvector", getvector, METH_NOARGS,
   ""},
  {"setvector", setvector, METH_VARARGS,
   ""},
  {NULL, NULL, 0, NULL}          /* Sentinel */
};


/**
 * Initialize the module object for Python with the exported functions
 */
static struct PyModuleDef testPythonListOfStringsModule = {
  PyModuleDef_HEAD_INIT,
  "testPythonListOfStrings",     /* name of module */
  "Module for testing copies of arrays of strings",   /* module documentation, may be NULL */
  -1,         /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
  ListOfStringsFunc,
  NULL,
  NULL,
  NULL,
  NULL,
};


PyMODINIT_FUNC
PyInit_testPythonListOfStrings( void )
{
  PyObject * module = PyModule_Create( &testPythonListOfStringsModule );
  PyModule_AddIntMacro( module, ARR_SIZE );
  if( !LvArray::python::addPyLvArrayModule( module ) )
  {
    return nullptr;
  }
  return module;
}
