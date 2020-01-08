//repoKernels.cuh
//Scott Grauer-Gray
//Headers for kernels for running repo on the GPU

#ifndef REPO_KERNELS_CUH
#define REPO_KERNELS_CUH

#include "repoFuncs.h"

__global__ void getRepoResultsGpu(inArgsStruct inArgs,
					              resultsStruct results,
					              int n);

#endif //REPO_KERNELS_CUH
