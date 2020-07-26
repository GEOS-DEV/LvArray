#include "squareAll.hpp"
#include "squareAllCompileCommands.hpp"

jitti::CompilationInfo getCompilationInfo()
{
  jitti::CompilationInfo info;

  info.compileCommand = squareAll_COMPILE_COMMAND;
  info.linker = squareAll_LINKER;
  info.linkArgs = squareAll_LINK_ARGS;
  info.header = "/usr/WS2/corbett5/LvArray/unitTests/jitti/squareAll.hpp";
  info.function = "squareAll";

  return info;
}
