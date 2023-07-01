set(CONFIG_NAME "quartz-gcc-12" CACHE PATH "") 

set(COMPILER_DIR /usr/tce/packages/gcc/gcc-12.1.1-magic)

# C
set(CMAKE_C_COMPILER ${COMPILER_DIR}/bin/gcc CACHE PATH "")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native" CACHE STRING "")

# C++
set(CMAKE_CXX_COMPILER ${COMPILER_DIR}/bin/g++ CACHE PATH "")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native" CACHE STRING "")

include(${CMAKE_CURRENT_LIST_DIR}/quartz-base.cmake)
