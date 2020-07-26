rm lib/libsquareAllJIT.so

CXX=/usr/tce/packages/clang/clang-upstream-2019.03.26/bin/clang++
OBJECT_OUTPUT=unitTests/jitti/CMakeFiles/squareAllDummy.dir/squareAll.cpp.o

/usr/tce/packages/cuda/cuda-10.1.243/bin/nvcc -ccbin=$CXX -restrict -arch sm_70 --expt-extended-lambda -Werror cross-execution-space-call,reorder,deprecated-declarations -g -G -O0 -Xcompiler -O0 -Xcompiler=-fPIC -Xcompiler=-fopenmp=libomp -std=c++14 -Iinclude -isystem=/usr/tce/packages/cuda/cuda-10.1.243/include -isystem=/usr/gapps/GEOSX/thirdPartyLibs/2020-07-08/install-lassen-clang@upstream-release/raja/include -x cu -c /usr/WS2/corbett5/LvArray/src/jitti/templateSource.cpp -o $OBJECT_OUTPUT -D JITTI_TEMPLATE_HEADER_FILE='"/usr/WS2/corbett5/LvArray/unitTests/jitti/squareAll.hpp"' -D JITTI_TEMPLATE_FUNCTION=squareAll -D JITTI_TEMPLATE_PARAMS='RAJA::omp_parallel_for_exec'

$CXX -fPIC -shared -o lib/libsquareAllJIT.so $OBJECT_OUTPUT lib/liblvarray.a /usr/gapps/GEOSX/thirdPartyLibs/2020-07-08/install-lassen-clang@upstream-release/raja/lib/libRAJA.a /usr/tce/packages/cuda/cuda-10.1.243/lib64/libcudart_static.a -pthread -ldl /usr/lib64/librt.so lib/libjitti.a

ls lib/libsquareAllJIT.so


# /usr/tce/packages/cuda/cuda-10.1.243/lib64
# /usr/tce/packages/cuda/cuda-10.1.243/nvidia/targets/ppc64le-linux/lib/stubs
# /usr/tce/packages/cuda/cuda-10.1.243/nvidia/targets/ppc64le-linux/lib
