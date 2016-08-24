CPP=g++
CPP_FLAGS=-std=c++11


test: main.o Derived1.o StringUtilities.o
	${CPP} ${CPP_FLAGS} -o test    main.o StringUtilities.o Derived1.o

testlib: main.o libDerived2.a StringUtilities.o
	${CPP} ${CPP_FLAGS} -o testlib main.o StringUtilities.o  Derived1.o -Wl,-all_load -L. -lDerived2 
#	${CPP} ${CPP_FLAGS} -o testlib main.o -Wl,--whole-archive -L. -lDerived 


main.o: main.cpp
	${CPP} ${CPP_FLAGS} -I../../src -c main.cpp 

Base.o: Base.cpp Base.hpp
	${CPP} ${CPP_FLAGS} -I../../src -c Base.cpp 

Derived1.o: Derived1.cpp Derived1.hpp Base.hpp
	${CPP} ${CPP_FLAGS} -I../../src  -c Derived1.cpp 

StringUtilities.o: ../../src/StringUtilities.cpp ../../src/StringUtilities.hpp
	${CPP} ${CPP_FLAGS} -I../../src  -c ../../src/StringUtilities.cpp 

libDerived2.a: Derived2.cpp Derived2.hpp Base.hpp
	${CPP} ${CPP_FLAGS} -I../../src  -c Derived2.cpp 
	ar qcsv libDerived2.a  Derived2.o
	ranlib libDerived2.a
	
	
	
clean:
	rm *.o *.a test testlib

all: test testlib