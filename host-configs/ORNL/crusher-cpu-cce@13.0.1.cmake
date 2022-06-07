
set(CONFIG_NAME "crusher-cpu-cce@13.0.1" CACHE PATH "")
include( crusher-base.cmake )

# Set up the tpls
set(GEOSX_TPL_ROOT_DIR "/gpfs/alpine/geo127/world-shared/cray-sles15-zen2/cce-13.0.1" CACHE PATH "")
set(GEOSX_TPL_DIR ${GEOSX_TPL_ROOT_DIR} CACHE PATH "")
set(GEOSX_TPL_DIR2 "/gpfs/alpine/geo127/world-shared/cray-sles15-zen3/cce-13.0.1" CACHE PATH "")

set(CAMP_DIR "${GEOSX_TPL_DIR2}/camp-0.2.2-mej6trivmy7o5vlr6a52cml6tzxb5fvk" CACHE PATH "" )

set(RAJA_DIR "${GEOSX_TPL_DIR2}/raja-2022.03.0-tmukf35ms7f2pkfswpejbnt3jtnpkakc" CACHE PATH "" )

set(ENABLE_UMPIRE TRUE CACHE BOOL "" )
set(UMPIRE_DIR "${GEOSX_TPL_DIR2}/umpire-2022.03.0-unirfq5er4vtyr2koymgi3xxq6h2f5l5" CACHE PATH "" )

set(ENABLE_CHAI TRUE CACHE BOOL "" )
set(CHAI_DIR "${GEOSX_TPL_DIR2}/chai-2022.03.0-aggyh463v2rz6s44laqshylc4xeeg4h7" CACHE PATH "" )

set(METIS_DIR "${GEOSX_TPL_DIR}/metis-5.1.0-zcfkawg5ifqpzcihrc3i6cdrrijusc2p/" CACHE PATH "" )
set(PARMETIS_DIR "${GEOSX_TPL_DIR}/parmetis-4.0.3-t2amifl5hh7yewre24gn2x3mlrz7qkl5/" CACHE PATH "" )

# C++ options
set(CMAKE_C_COMPILER "/opt/cray/pe/craype/2.7.13/bin/cc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/opt/cray/pe/craype/2.7.13/bin/CC" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/opt/cray/pe/craype/2.7.13/bin/ftn" CACHE PATH "")

# HIP Options
set( ENABLE_HIP OFF CACHE BOOL "" FORCE )
