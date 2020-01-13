//bondsKernelsGpu.cuh
//Scott Grauer-Gray
//Header for bonds kernels to run on the GPU w/ CUDA

#ifndef BONDS_KERNELS_H
#define BONDS_KERNELS_H

#include <hip/hip_runtime.h>
#include "bondsFuncs.h"


__global__ void getBondsResultsGpu(inArgsStruct inArgs,
                                   resultsStruct results,
                                   int n);

#endif //BONDS_KERNELS_H
