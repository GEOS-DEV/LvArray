#include "Macros.hpp"

void eprint(char const * format, ...)
{
  va_list args;
  va_start(args, format);
  vprintf(format, args);
  va_end(args);
}
