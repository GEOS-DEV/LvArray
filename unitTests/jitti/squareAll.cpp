#include "squareAll.hpp"
#include "squareAllCompileCommands.hpp"

jitti::CompilationInfo getCompilationInfo()
{
  jitti::CompilationInfo info;

  info.compilerPath = squareAll_COMPILE_COMMANDS;

  info.header = "/usr/WS2/corbett5/LvArray/unitTests/jitti/squareAll.hpp";

  info.libs = {
    JITTI_RAJA_LIBRARY
  };

  return info;
}
