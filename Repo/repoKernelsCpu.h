//repoKernelsCpu.cuh
//Scott Grauer-Gray
//July 6, 2012
//Header for repo kernels on the CPU

#ifndef REPO_KERNELS_CPU_H
#define REPO_KERNELS_CPU_H

#include "repoFuncs.h"
#include <omp.h>

void getRepoResultsCpu(inArgsStruct inArgs,
                       resultsStruct results,
                       int totNumRuns);

void getRepoResultsOpenMP(inArgsStruct inArgs,
                          resultsStruct results,
                          int totNumRuns);

#endif //REPO_KERNELS_CPU_H
