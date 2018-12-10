
#include <iostream>
#include "SetFPE.hpp"
#include "SetSignalHandling.hpp"
#include "stackTrace.hpp"
#include "testHelperFunctions.hpp"

namespace testFloatingPointExceptionsHelpers
{

void func3(double divisor)
{
  double a = 1.0 / divisor;
  std::cout << "1.0/0.0 didn't kill program, result is " << a;
}

void func2(double divisor)
{
  func3(divisor);
}

void func1(double divisor)
{
  func2(divisor);
}

void func0(double divisor)
{
  func1(divisor);
}

void testStackTrace(double divisor)
{
  cxx_utilities::setSignalHandling(cxx_utilities::handler1);
  func0(divisor);
}

//TEST(testStackTrace_DeathTest, stackTrace)
//{
//   EXPECT_DEATH_IF_SUPPORTED(testStackTrace(0), IGNORE_OUTPUT);
//}




#pragma STDC FENV_ACCESS ON
void show_fe_exceptions(void)
{
    printf("exceptions raised:");
    if(fetestexcept(FE_DIVBYZERO)) printf(" FE_DIVBYZERO");
    if(fetestexcept(FE_INEXACT))   printf(" FE_INEXACT");
    if(fetestexcept(FE_INVALID))   printf(" FE_INVALID");
    if(fetestexcept(FE_OVERFLOW))  printf(" FE_OVERFLOW");
    if(fetestexcept(FE_UNDERFLOW)) printf(" FE_UNDERFLOW");
    feclearexcept(FE_ALL_EXCEPT);
    printf("\n");
}

double uf_test(double x, double denominator)
{
  return x/denominator;
}

double of_test( double x, double y )
{
  return x*y;
}
}
