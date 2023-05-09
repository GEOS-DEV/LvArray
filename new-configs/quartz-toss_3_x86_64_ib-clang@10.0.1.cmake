#################################################################################
# Generated host-config - Edit at own risk!
#################################################################################
#--------------------------------------------------------------------------------
# SYS_TYPE: toss_3_x86_64_ib
# Compiler Spec: clang@10.0.1
# CMake executable path: /usr/tce/packages/cmake/cmake-3.14.5/bin/cmake
#--------------------------------------------------------------------------------

set(BLT_SOURCE_DIR "/usr/WS2/corbett5/LvArray/uberenv-libs/linux-rhel7-broadwell/clang-10.0.1/blt-0.5.2-6nztad6saell6ikor6wtxp6qycxtfwh4" CACHE PATH "")

#--------------------------------------------------------------------------------
# Compilers
#--------------------------------------------------------------------------------

set(CMAKE_C_COMPILER "/usr/tce/bin/clang-10.0.1" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/tce/bin/clang++-10.0.1" CACHE PATH "")

set(CMAKE_C_FLAGS "-march=native -mtune=native --gcc-toolchain=/usr/tce/packages/gcc/gcc-8.1.0" CACHE PATH "")

set(CMAKE_CXX_FLAGS "-march=native -mtune=native --gcc-toolchain=/usr/tce/packages/gcc/gcc-8.1.0" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "")

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG" CACHE STRING "")

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")

set(ENABLE_CUDA OFF CACHE BOOL "")

#--------------------------------------------------------------------------------
# CAMP
#--------------------------------------------------------------------------------

set(CAMP_DIR "/usr/WS2/corbett5/LvArray/uberenv-libs/linux-rhel7-broadwell/clang-10.0.1/camp-2022.03.2-2q75xbq2h4ykcyvasoqg55torawlabkw" CACHE PATH "")

#--------------------------------------------------------------------------------
# RAJA
#--------------------------------------------------------------------------------

set(RAJA_DIR "/usr/WS2/corbett5/LvArray/uberenv-libs/linux-rhel7-broadwell/clang-10.0.1/raja-2022.03.0-jkp4hp7ifyxkxzkbho5ngdnk4x3opaoy" CACHE PATH "")

#--------------------------------------------------------------------------------
# Umpire
#--------------------------------------------------------------------------------

set(ENABLE_UMPIRE ON CACHE BOOL "")

set(UMPIRE_DIR "/usr/WS2/corbett5/LvArray/uberenv-libs/linux-rhel7-broadwell/clang-10.0.1/umpire-2022.03.1-aerit7injc3hmn2ripnsxtnlwxicjmuu" CACHE PATH "")

#--------------------------------------------------------------------------------
# CHAI
#--------------------------------------------------------------------------------

set(ENABLE_CHAI ON CACHE BOOL "")

set(CHAI_DIR "/usr/WS2/corbett5/LvArray/uberenv-libs/linux-rhel7-broadwell/clang-10.0.1/chai-2022.03.0-s6w2gsrreu7krgzboekmlukmfestpg7k" CACHE PATH "")

#--------------------------------------------------------------------------------
# Caliper
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# Caliper
#--------------------------------------------------------------------------------

set(ENABLE_CALIPER ON CACHE BOOL "")

set(CALIPER_DIR "/usr/WS2/corbett5/LvArray/uberenv-libs/linux-rhel7-broadwell/clang-10.0.1/caliper-2.8.0-3fwkrbu4bhnc4bqvhrqcydrzxslq6ryz" CACHE PATH "")

#--------------------------------------------------------------------------------
# Python
#--------------------------------------------------------------------------------

set(ENABLE_PYLVARRAY OFF CACHE BOOL "")

#--------------------------------------------------------------------------------
# Documentation
#--------------------------------------------------------------------------------

set(ENABLE_DOCS OFF CACHE BOOL "")

#--------------------------------------------------------------------------------
# addr2line
#--------------------------------------------------------------------------------

set(ENABLE_ADDR2LINE ON CACHE BOOL "")

#--------------------------------------------------------------------------------
# Other
#--------------------------------------------------------------------------------

