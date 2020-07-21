/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2019, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * LLNL-CODE-746361
 *
 * All rights reserved. See COPYRIGHT for details.
 *
 * This file is part of the GEOSX Simulation Framework.
 *
 * GEOSX is a free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License (as published by the
 * Free Software Foundation) version 2.1 dated February 1999.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

#include "benchmarkEigendecompositionKernels.hpp"

namespace LvArray
{
namespace benchmarking
{

template< int M, typename PERM_2D, typename PERM_3D, typename POLICY >
void Eigendecomposition< M, PERM_2D, PERM_3D, POLICY >::
eigenvaluesKernel( ArrayViewT< VALUE_TYPE const, PERM_2D > const & matrices,
                   ArrayViewT< FLOAT, PERM_2D > const & eigenvalues )
{
  forall< POLICY >( matrices.size( 0 ), [matrices, eigenvalues] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
      {
        tensorOps::symEigenvalues< M >( eigenvalues[ i ], matrices[ i ] );
      } );
}

template< int M, typename PERM_2D, typename PERM_3D, typename POLICY >
void Eigendecomposition< M, PERM_2D, PERM_3D, POLICY >::
eigenvectorsKernel( ArrayViewT< VALUE_TYPE const, PERM_2D > const & matrices,
                    ArrayViewT< FLOAT, PERM_2D > const & eigenvalues,
                    ArrayViewT< FLOAT, PERM_3D > const & eigenvectors )
{
  forall< POLICY >( matrices.size( 0 ), [matrices, eigenvalues, eigenvectors] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
      {
        tensorOps::symEigenvectors< M >( eigenvalues[ i ], eigenvectors[ i ], matrices[ i ] );
      } );
}


template class Eigendecomposition< 2, RAJA::PERM_IJ, RAJA::PERM_IJK, serialPolicy >;
template class Eigendecomposition< 2, RAJA::PERM_JI, RAJA::PERM_KJI, serialPolicy >;
template class Eigendecomposition< 3, RAJA::PERM_IJ, RAJA::PERM_IJK, serialPolicy >;
template class Eigendecomposition< 3, RAJA::PERM_JI, RAJA::PERM_KJI, serialPolicy >;

#if defined(USE_OPENMP)
template class Eigendecomposition< 2, RAJA::PERM_IJ, RAJA::PERM_IJK, parallelHostPolicy >;
template class Eigendecomposition< 2, RAJA::PERM_JI, RAJA::PERM_KJI, parallelHostPolicy >;
template class Eigendecomposition< 3, RAJA::PERM_IJ, RAJA::PERM_IJK, parallelHostPolicy >;
template class Eigendecomposition< 3, RAJA::PERM_JI, RAJA::PERM_KJI, parallelHostPolicy >;
#endif

#if defined(USE_CUDA) && defined(USE_CHAI)
template class Eigendecomposition< 2, RAJA::PERM_IJ, RAJA::PERM_IJK, parallelDevicePolicy< THREADS_PER_BLOCK > >;
template class Eigendecomposition< 2, RAJA::PERM_JI, RAJA::PERM_KJI, parallelDevicePolicy< THREADS_PER_BLOCK > >;
template class Eigendecomposition< 3, RAJA::PERM_IJ, RAJA::PERM_IJK, parallelDevicePolicy< THREADS_PER_BLOCK > >;
template class Eigendecomposition< 3, RAJA::PERM_JI, RAJA::PERM_KJI, parallelDevicePolicy< THREADS_PER_BLOCK > >;
#endif

} // namespace benchmarking
} // namespace LvArray
