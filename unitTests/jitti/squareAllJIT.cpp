#include "squareAllJIT.hpp"
#include "squareAllCompileCommands.hpp"
#include "squareAll.hpp"

jitti::CompilationInfo getCompilationInfo()
{
  jitti::CompilationInfo info;

  info.compileCommand = squareAllJIT_COMPILE_COMMAND;
#if defined( LVARRAY_USE_CUDA )
  info.compilerIsNVCC = true;
#else
  info.compilerIsNVCC = false;
#endif

  info.linker = squareAllJIT_LINKER;
  info.linkArgs = squareAllJIT_LINK_ARGS;

  info.templateFunction = "squareAll";

  std::string const currentFile = __FILE__;
  info.headerFile = currentFile.substr( 0, currentFile.size() - ( sizeof( "squareAllJIT.cpp" ) - 1 ) )
                                 + "squareAll.hpp";

  return info;
}
