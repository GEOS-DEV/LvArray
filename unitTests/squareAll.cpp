// Source includes
#include "squareAll.hpp"

// POLICY_DEF should be defined on the command line.
void squareAll( LvArray::ArraySlice< int, 1, 0, std::ptrdiff_t > const dst,
                LvArray::ArraySlice< int, 1, 0, std::ptrdiff_t > const src )
{
  internal::squareAll< POLICY_DEF >( dst, src );
}

int x = 0;

std::string getStringX()
{ return std::to_string( x ); }

extern "C"
{

// This contains a list of symbols to be exported. exporting c++ code as well as type checking.
SymbolTable * getExportedSymbols()
{
  static SymbolTable symbols {
    createEntry( "squareAll", &squareAll ),
    createEntry( "x", &x ),
    createEntry( "getStringX", &getStringX )
  };

  return &symbols;
}

} // extern "C"

