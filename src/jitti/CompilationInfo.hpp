#pragma once

#include <string>

namespace jitti
{

namespace internal
{

time_t getCompileTime( char const * const data, char const * const time );

} // namespace internal

/**
 * 
 */
struct CompilationInfo
{
  time_t const compilationTime = internal::getCompileTime( __DATE__, __TIME__ );
  std::string compileCommand;
  bool compilerIsNVCC;
  std::string linker;
  std::string linkArgs;
  std::string templateFunction;
  std::string templateParams;
  std::string headerFile;
  std::string outputObject;
  std::string outputLibrary;
};

} // namespace jitti
