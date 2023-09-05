/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2019, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * LLNL-CODE-746361
 *
 * All rights reserved. See COPYRIGHT for details.
 *
 * This file is part of the GEOSX Simulation Framework.
 *
 * GEOSX is a free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License (as published by the
 * Free Software Foundation) version 2.1 dated February 1999.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

#pragma once

/**
 * @file pythonHelpers.hpp
 */

// Source includes
#include "pythonForwardDeclarations.hpp"
#include "../system.hpp"
#include "../Macros.hpp"
#include "../limits.hpp"

// system includes
#include <vector>
#include <stdexcept>

/**
 * @brief Raise a Python exception if @c CONDITION is met.
 * @param CONDITION The condition to check.
 * @param TYPE The Type of Python exception to raise.
 * @param MSG The message to give the exception.
 * @param RET The return value if there was an error.
 */
#if defined(PyObject_HEAD)
#define PYTHON_ERROR_IF( CONDITION, TYPE, MSG, RET ) \
  do \
  { \
    if( CONDITION ) \
    { \
      std::ostringstream __oss; \
      __oss << "***** ERROR\n"; \
      __oss << "***** LOCATION: " LOCATION "\n"; \
      __oss << "***** Controlling expression (should be false): " STRINGIZE( EXP ) "\n"; \
      __oss << MSG << "\n"; \
      __oss << LvArray::system::stackTrace( true ); \
      PyErr_SetString( TYPE, __oss.str().c_str() ); \
      return RET; \
    } \
  } while( false )
#else
#define PYTHON_ERROR_IF( CONDITION, TYPE, MSG, RET ) \
  static_assert( false, "You are attempting to use PYTHON_ERROR_IF but haven't yet included Python.hpp" )
#endif

/// @cond DO_NOT_DOCUMENT
// UNCRUSTIFY-OFF
/// Doxygen and uncrustify don't like this macro/pragma nest.

/**
 * @brief Begin mixing designated and non-designated initializers in the same initializer list.
 */
#if defined( __clang_version__ )
  #if defined( __CUDACC__ )
    #define BEGIN_ALLOW_DESIGNATED_INITIALIZERS \
      _Pragma( "GCC diagnostic push" ) \
      _Pragma( "GCC diagnostic ignored \"-Wc99-extensions\"") \
      _Pragma( "GCC diagnostic ignored \"-Wgnu-designator\"")
    #if __has_warning( "-Wc99-designator" )
      _Pragma( "GCC diagnostic ignored \"-Wc99-designator\"")
    #endif
  #else
    #define BEGIN_ALLOW_DESIGNATED_INITIALIZERS \
      _Pragma( "GCC diagnostic push" ) \
      _Pragma( "GCC diagnostic ignored \"-Wc99-designator\"")
  #endif
#elif defined( __GNUC__ )
  #define BEGIN_ALLOW_DESIGNATED_INITIALIZERS \
    _Pragma( "GCC diagnostic push" ) \
    _Pragma( "GCC diagnostic ignored \"-Wpedantic\"" ) \
    _Pragma( "GCC diagnostic ignored \"-Wmissing-field-initializers\"" )
    #if __GNUC__ > 8 
      _Pragma( "GCC diagnostic ignored \"-Wc++20-extensions\"")
    #endif
#endif

/**
 * @brief End mixing designated and non-designated initializers in the same initializer list.
 */
#define END_ALLOW_DESIGNATED_INITIALIZERS \
  _Pragma( "GCC diagnostic pop" )

// UNCRUSTIFY-ON
/// @endcond DO_NOT_DOCUMENT

namespace LvArray
{
namespace python
{
namespace internal
{

/**
 * @brief A wrapper around Py_XINCREF.
 * @param obj The object to increase the reference count of, may be a nullptr.
 */
void xincref( PyObject * const obj );

/**
 * @brief A wrapper around Py_XDECREF.
 * @param obj The object to decrease the reference count of, may be a nullptr.
 */
void xdecref( PyObject * const obj );

/**
 * @brief Return @c true if @p obj is an instance of @p type.
 * @param obj The object.
 * @param type The type.
 * @return @c true if @p obj is an instance of @p type.
 */
bool isInstanceOf( PyObject * const obj, PyTypeObject * type );

} // namespace internal

/**
 * @enum PyModify
 * @brief An enumeration of the various access policies for Python objects.
 */
enum class PyModify
{
  READ_ONLY = 0,
  MODIFIABLE = 1,
  RESIZEABLE = 2,
};

/**
 * @class PyObjectRef
 * @brief A class that manages an owned Python reference with RAII semantics.
 * @tparam T The type of the managed object, must be castable to a PyObject.
 */
template< typename T = PyObject >
class PyObjectRef
{
public:

  /**
   * @brief Create an uninitialized (nullptr) reference.
   */
  PyObjectRef() = default;

  /**
   * @brief Take ownership of a reference to @p src.
   * @param src The object to be referenced.
   * @note Does not increase the reference count.
   */
  PyObjectRef( T * const src ):
    m_object( src )
  {}

  /**
   * @brief Create a new reference to @p src.
   * @param src The object to create a new reference to.
   * @note Increases the reference count.
   */
  PyObjectRef( PyObjectRef const & src )
  { *this = src; }

  /**
   * @brief Steal a reference from @p src.
   * @param src The object to steal a reference to.
   * @note Does not increase the reference count.
   */
  PyObjectRef( PyObjectRef && src )
  { *this = std::move( src ); }

  /**
   * @brief Destructor, decreases the reference count.
   */
  ~PyObjectRef()
  { internal::xdecref( reinterpret_cast< PyObject * >( m_object ) ); }

  /**
   * @brief Create a new reference to @p src.
   * @param src The object to create a new reference to.
   * @return *this.
   * @note Increases the reference count.
   */
  PyObjectRef & operator=( PyObjectRef const & src )
  {
    m_object = src.m_object;
    internal::xincref( reinterpret_cast< PyObject * >( src.m_object ) );
    return *this;
  }

  /**
   * @brief Steal a reference from @p src.
   * @param src The object to steal a reference to.
   * @return *this.
   * @note Does not increase the reference count.
   */
  PyObjectRef & operator=( PyObjectRef && src )
  {
    m_object = src.m_object;
    src.m_object = nullptr;
    return *this;
  }

  /**
   * @brief Decrease the reference count to the current object and take ownership
   *   of a new reference.
   * @param src The new object to be referenced.
   * @return *this.
   */
  PyObjectRef & operator=( PyObject * src )
  {
    internal::xdecref( reinterpret_cast< PyObject * >( m_object ) );

    m_object = src;
    return *this;
  }

  /**
   * @brief Conversion operator to a T *.
   * @return A pointer to the managed object.
   */
  operator T *()
  { return m_object; }

  /**
   * @brief Return a pointer to the managed object.
   * @return A pointer to the managed object.
   */
  T * get() const
  { return m_object; }

  /**
   * @brief Return the address of the pointer to the manged object.
   * @return The address of the pointer to the manged object.
   * @details Useful for functions which take a PyObject** as an output parameter
   *   and fill it with a new reference.
   */
  T * * getAddress()
  { return &m_object; }

  /**
   * @brief Return the address of the managed object and release ownership.
   * @return The address of the managed object and release ownership.
   * @details Useful for functions which return a PyObject *.
   */
  T * release()
  {
    T * const ret = m_object;
    m_object = nullptr;
    return ret;
  }

private:
  /// A pointer to the manged object.
  T * m_object = nullptr;
};

/**
 * @brief Return @p obj casted to T if @p obj is an instance of @p type or @c nullptr if it is not.
 * @tparam T The type to cast @p obj to.
 * @param obj The object.
 * @param type The type.
 * @return @p obj casted to T if @p obj is an instance of @p type or @c nullptr if it is not.
 */
template< typename T >
T * convert( PyObject * const obj, PyTypeObject * const type )
{
  if( internal::isInstanceOf( obj, type ) )
  {
    return reinterpret_cast< T * >( obj );
  }

  return nullptr;
}

/**
 * @brief Add the Python type @p type to the module @p module.
 * @param module The Python module to add @p type to.
 * @param type The Python type to add to @p module.
 * @param typeName The name to give the type.
 * @return @c true iff the operation was successful.
 */
bool addTypeToModule( PyObject * const module,
                      PyTypeObject * const type,
                      char const * const typeName );

/**
 * @brief Return a Python string copy of @c value.
 * @param value The string to copy into Python.
 * @return A Python string copy of @c value.
 */
PyObject * create( std::string const & value );

/**
 * @brief Create and return a Python list of strings from an array of std::strings.
 *   The Python strings will be copies.
 * @param strptr A pointer to the strings to convert, must be of length @c size.
 * @param size The number of strings in @c strptr.
 * @return A Python list of strings, or @c nullptr if there was an error.
 */
PyObject * createPyListOfStrings( std::string const * const strptr, long long const size );

/**
 * @brief Create and return a Python list of strings from a std::vector of std::strings.
 *   The Python strings will be copies.
 * @param vec the vector to convert.
 * @return A Python list of strings, or @c nullptr if there was an error.
 */
inline PyObject * create( std::vector< std::string > const & vec )
{ return createPyListOfStrings( vec.data(), integerConversion< long long >( vec.size() ) ); }

/**
 * @class PythonError
 * @brief Base class for all C++ exceptions related to Python.
 * @details Intended to be thrown when an actual Python exception is raised,
 *   then caught before returning control to Python. When used this
 *   way, it provides an easy way to return control to Python
 *   when a Python exception occurs (instead of e.g. propagating nullptr
 *   all the way back to Python, which is usually how error handling is done).
 */
class PythonError : public std::runtime_error
{
public:
  PythonError():
    std::runtime_error( LvArray::system::stackTrace( true ) )
  {}
};

} // namespace python
} // namespace LvArray
