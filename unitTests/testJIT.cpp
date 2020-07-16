// Source includes
#include "StringUtilities.hpp"
#include "Array.hpp"
#include "MallocBuffer.hpp"

// TPL includes
#include <gtest/gtest.h>

// System includes
#include <iostream>
#include <dlfcn.h>
#include <unordered_map>
#include <typeindex>
#include <string>
#include <cstdlib>

class DynamicLibrary
{

using SymbolTable = std::unordered_map< std::string, std::pair< void *, std::type_index > >;

public:
  DynamicLibrary( std::string const & path ):
    m_handle( dlopen( path.data(), RTLD_LAZY ) ),
    m_symbols( getSymbolTable( m_handle ) )
  {}

  DynamicLibrary( DynamicLibrary const & ) = delete;

  DynamicLibrary( DynamicLibrary && src ) = default;

  ~DynamicLibrary()
  {
    LVARRAY_ERROR_IF( dlclose( m_handle ), dlerror() );
  }

  template< typename T >
  T getSymbol( std::string const & name )
  {
    auto const iter = m_symbols.find( name );
    LVARRAY_ERROR_IF( iter == m_symbols.end(), "Symbol \"" << name << "\" not found in the exported symbol table.\n" );

    std::pair< void *, std::type_index > const value = iter->second;

    LVARRAY_ERROR_IF( value.second != std::type_index( typeid( T ) ),
                      "Symbol \"" << name << "\" found but it has type " <<
                      LvArray::demangle( value.second.name() ) << " not " << LvArray::demangleType< T >() );

    return reinterpret_cast< T >( value.first );
  }

private:
  SymbolTable getSymbolTable( void * const handle )
  {
    LVARRAY_ERROR_IF( handle == nullptr, dlerror() );

    SymbolTable * (* getExportedSymbols )() = reinterpret_cast< SymbolTable * (*)() >( dlsym( handle, "getExportedSymbols" ) );

    char const * const error = dlerror();

    LVARRAY_ERROR_IF( error != nullptr, "Could not find the symbols table!\n" << error );

    return *getExportedSymbols();
  }

  void * const m_handle;
  SymbolTable const m_symbols;
};


class Compiler
{
public:
  Compiler( std::string const & compilerPath, std::string const & compilerArgs ):
    m_compilerPath( compilerPath ),
    m_compilerArgs( compilerArgs )
  {}

  DynamicLibrary createDynamicLibrary( std::string const & filePath,
                                       std::string const & outputPath,
                                       std::vector< std::string > const & includeDirs,
                                       std::vector< std::string > const & systemIncludeDirs,
                                       std::vector< std::string > const & libs,
                                       std::vector< std::string > const & defines ) const
  {
    std::string command = m_compilerPath + " " + m_compilerArgs + " -fPIC -shared -o " + outputPath + " " + filePath;

    for ( std::string const & include : includeDirs )
    { command += " -I " + include; }

    for ( std::string const & include : systemIncludeDirs )
    { command += " -isystem " + include; }

    for ( std::string const & lib : libs )
    { command += " " + lib; }

    for ( std::string const & define : defines )
    { command += " -D " + define; }

    LVARRAY_LOG( "Compiling " << filePath );
    LVARRAY_ERROR_IF( std::system( command.data() ) != 0, command );

    return DynamicLibrary( outputPath );
  }

private:
  std::string const m_compilerPath;
  std::string const m_compilerArgs;
};


TEST( JIT, SquareAll )
{
  // The compiler to use and the standard compilation flags. These would be set by CMake in a header file.
  std::string const compiler = "/usr/tce/packages/clang/clang-10.0.0/bin/clang++";
  std::string const compilerFlags = "-Wall -Wextra -Werror -Wpedantic -pedantic-errors -Wshadow -Wfloat-equal "
                                   "-Wcast-align -Wcast-qual -fcolor-diagnostics -g -fstandalone-debug "
                                   "-fopenmp=libomp -std=c++14";

  Compiler clang( compiler, compilerFlags );

  // The source file to compile.
  std::string const sourceFile = "/usr/WS2/corbett5/cxx-utilities/unitTests/squareAll.cpp";

  // The output library name.
  std::string const outputLib = "/usr/WS2/corbett5/cxx-utilities/build-quartz-clang@10.0.0-debug/lib/libsquareAll.so";

  // Include directories for the target.
  std::vector< std::string > const includeDirs = {
    "/usr/WS2/corbett5/cxx-utilities/src",
    "/usr/WS2/corbett5/cxx-utilities/build-quartz-clang@10.0.0-debug/include"
  };

  // System include directories for the target.
  std::vector< std::string > const systemIncludeDirs = {
    "/usr/tce/packages/mvapich2/mvapich2-2.3-clang-10.0.0/include",
    "/usr/gapps/GEOSX/thirdPartyLibs/2020-06-17/install-quartz-clang@10.0.0-release/raja/include",
    "/usr/gapps/GEOSX/thirdPartyLibs/2020-06-17/install-quartz-clang@10.0.0-release/caliper/include",
    "/usr/gapps/GEOSX/thirdPartyLibs/2020-06-17/install-quartz-clang@10.0.0-release/chai/include",
  };

  // Libraries required for the target.
  std::vector< std::string > const libs = {
    "/usr/gapps/GEOSX/thirdPartyLibs/2020-06-17/install-quartz-clang@10.0.0-release/raja/lib/libRAJA.a"
  };

  // In this case the policy is passed as a define.
  std::vector< std::string > const & defines = {
    "POLICY_DEF=RAJA::loop_exec"
  };
  
  // Compile the source and load it as a dynamic library.
  DynamicLibrary dl = clang.createDynamicLibrary( sourceFile, outputLib, includeDirs, systemIncludeDirs, libs, defines );

  // Prepare the input.
  LvArray::Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, LvArray::MallocBuffer > output( 10 );
  LvArray::Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, LvArray::MallocBuffer > input( 10 );

  for ( std::ptrdiff_t i = 0; i < input.size(); ++i )
  { input[ i ] = i; }

  using SquareAllType = void (*)( LvArray::ArraySlice< int, 1, 0, std::ptrdiff_t > const,
                                  LvArray::ArraySlice< int, 1, 0, std::ptrdiff_t > const );

  // Load the function from the library.
  SquareAllType squareAll = dl.getSymbol< SquareAllType >( "squareAll" );

  // Call the function.
  squareAll( output, input );

  // Check the output.
  for ( std::ptrdiff_t i = 0; i < output.size(); ++i )
  { EXPECT_EQ( output[ i ], i * i ); }

  int & x = *dl.getSymbol< int * >( "x" );
  std::string (*getStringX)() = dl.getSymbol< std::string (*)() >( "getStringX" );

  EXPECT_EQ( x, 0 );
  EXPECT_EQ( getStringX(), "0" );

  x = 1672;
  EXPECT_EQ( getStringX(), "1672" );

  // Supplying the wrong type for a symbol is an error.
  EXPECT_DEATH_IF_SUPPORTED( dl.getSymbol< void (*)() >( "squareAll" ), "" );

  // Asking for a non-existant symbol is an error.
  EXPECT_DEATH_IF_SUPPORTED( dl.getSymbol< void (*)() >( "bar" ), "" );
}

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}

