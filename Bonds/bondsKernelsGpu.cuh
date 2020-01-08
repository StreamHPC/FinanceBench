//bondsKernelsGpu.cuh
//Scott Grauer-Gray
//Header for bonds kernels to run on the GPU w/ CUDA

#ifndef BONDS_KERNELS_GPU_CUH
#define BONDS_KERNELS_GPU_CUH

#include "bondsFuncs.h"

__global__ void getBondsResultsGpu(inArgsStruct inArgs,
                                   resultsStruct results,
                                   int n);

#endif //BONDS_KERNELS_GPU_CUH
