set(COMPILER_DIR /usr/tce/packages/clang/clang-14.0.6-magic)

# C
set(CMAKE_C_COMPILER ${COMPILER_DIR}/bin/clang CACHE PATH "")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native" CACHE STRING "")

# C++
set(CMAKE_CXX_COMPILER ${COMPILER_DIR}/bin/clang++ CACHE PATH "")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native" CACHE STRING "")

include(${CMAKE_CURRENT_LIST_DIR}/llnl-cpu-base.cmake)
