#ifndef STACKARRAYWRAPPER_HPP_
#define STACKARRAYWRAPPER_HPP_

#include <stddef.h>
#include "Logger.hpp"

namespace LvArray
{

template< typename T, int LENGTH >
struct StackArrayWrapper
{
  typedef T * iterator;
  typedef T const * const_iterator;

  void free() {}

  void resize( ptrdiff_t length )
  {
    GEOS_ERROR_IF( length > LENGTH, "C_Array::resize("<<length<<") is larger than template argument LENGTH=" << LENGTH );
  }

  T * data()             { return m_data; }
  T const * data() const { return m_data; }

  T m_data[LENGTH];
};
}

#endif
