//repoKernelsCpu.cuh
//Scott Grauer-Gray
//July 6, 2012
//Header for repo kernels on the CPU

#ifndef REPO_KERNELS_CPU_H
#define REPO_KERNELS_CPU_H

#include "repoFuncs.h"

void getRepoResultsGpuCpu(inArgsStruct inArgs,
                          resultsStruct results,
                          int totNumRuns);

#endif //REPO_KERNELS_CPU_H
