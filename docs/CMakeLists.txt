###################################################################################################
# Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
# All rights reserved.
# See the LICENSE file for details.
# SPDX-License-Identifier: (BSD-3-Clause)
###################################################################################################

################################
# documentation targets
################################

if( DOXYGEN_FOUND )
    add_subdirectory(doxygen)
endif()

if( SPHINX_FOUND )
  add_subdirectory(sphinx)
endif()
