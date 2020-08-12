//repoKernels.cuh
//Scott Grauer-Gray
//Headers for kernels for running repo on the GPU

#ifndef REPO_KERNELS_H
#define REPO_KERNELS_H

#include <hip/hip_runtime.h>
#include "repoStructs.h"

__global__ void getRepoResultsGpu(inArgsStruct inArgs,
					              resultsStruct results,
					              int n);

#endif //REPO_KERNELS_H
