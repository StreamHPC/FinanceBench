//bondsKernelsCpu.cuh
//Scott Grauer-Gray
//July 6, 2012
//Header for repo kernels to run on the CPU

#ifndef BONDS_KERNELS_CPU_H
#define BONDS_KERNELS_CPU_H

#include "bondsFuncs.h"
#include <omp.h>

void getBondsResultsCpu(inArgsStruct inArgs,
                        resultsStruct results,
                        int totNumRuns);

void getBondsResultsOpenMP(inArgsStruct inArgs,
                           resultsStruct results,
                           int totNumRuns);

#endif //BONDS_KERNELS_CPU_H
