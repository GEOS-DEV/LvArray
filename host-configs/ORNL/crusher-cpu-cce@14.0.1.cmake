
set(CONFIG_NAME "crusher-cpu-cce@14.0.1" CACHE PATH "")
include( ${CMAKE_CURRENT_LIST_DIR}/crusher-base.cmake )

# Set up the tpls
set(SYSTEM_TPL_DIR "/sw/crusher/spack-envs/base/opt/cray-sles15-zen3/cce-14.0.0" CACHE PATH "")
set(GEOSX_TPL_DIR "/gpfs/alpine/geo127/world-shared/cray-sles15-zen3/cce-14.0.0" CACHE PATH "")
set(GEOSX_TPL_DIR ${GEOSX_TPL_ROOT_DIR} CACHE PATH "")

set(CAMP_DIR "${GEOSX_TPL_DIR}/camp-0.2.2-r3wj7xcb7xycpqhda7rf5b2lekrsqx4w" CACHE PATH "" )

set(RAJA_DIR "${GEOSX_TPL_DIR}/raja-2022.03.0-33qhciagztrs7zbnyk3uufxwjllerqer" CACHE PATH "" )

set(ENABLE_UMPIRE TRUE CACHE BOOL "" )
set(UMPIRE_DIR "${GEOSX_TPL_DIR}/umpire-2022.03.0-72z2rwrcmmojgau6wgfmcckrnzbvi4aa" CACHE PATH "" )

set(ENABLE_CHAI TRUE CACHE BOOL "" )
set(CHAI_DIR "${GEOSX_TPL_DIR}/chai-2022.03.0-da6y22vj4iy3uru2bne7df7tpngckt3n" CACHE PATH "" )

set(METIS_DIR "${SYSTEM_TPL_DIR}/metis-5.1.0-q4xwj6aya5ooq4owhrhh2qllanjmyuew/" CACHE PATH "" )
set(PARMETIS_DIR "${SYSTEM_TPL_DIR}/parmetis-4.0.3-fyed2lqdzmulw6d4cl3oxt6r6jaqdtwg/" CACHE PATH "" )

# C++ options
set(CRAYPE_VERSION "2.7.15")
set(CMAKE_C_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/cc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/CC" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/ftn" CACHE PATH "")

# HIP Options
set( ENABLE_HIP OFF CACHE BOOL "" FORCE )