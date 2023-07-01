set(CONFIG_NAME "quartz-icc@19.0.4" CACHE PATH "")

set(COMPILER_DIR /usr/tce/packages/intel/intel-19.0.4/compilers_and_libraries_2019.4.227/linux)

# C
set(CMAKE_C_COMPILER ${COMPILER_DIR}/bin/intel64/icc CACHE PATH "")
set(CMAKE_C_FLAGS_RELEASE "-DNDEBUG -march=native -mtune=native -qoverride-limits" CACHE STRING "")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-g ${CMAKE_C_FLAGS_RELEASE}" CACHE STRING "")

# C++
set(CMAKE_CXX_COMPILER ${COMPILER_DIR}/bin/intel64/icpc CACHE PATH "")
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -march=native -mtune=native -qoverride-limits" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g ${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "")

include(${CMAKE_CURRENT_LIST_DIR}/quartz-base.cmake)
