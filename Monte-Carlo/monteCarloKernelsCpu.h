//monteCarloKernelsCpu.cuh
//Scott Grauer-Gray
//May 10, 2012
//Headers for monte carlo function on the CPU

#ifndef MONTE_CARLO_KERNELS_CPU_H
#define MONTE_CARLO_KERNELS_CPU_H

//needed for constants related to monte carlo
#include "monteCarloFuncs.h"
//#include "monteCarloStructs.h"
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>

//function to set up the random states
void setup_kernelCpu();

void getPathCpu(dataType * path,
                size_t sampleNum,
                dataType dt,
                monteCarloOptionStruct optionStruct,
                unsigned int * seedp);

void monteCarloGpuKernelOpenMP(dataType * samplePrices,
                               dataType * sampleWeights,
                               dataType * times,
                               dataType dt,
                               monteCarloOptionStruct * optionStructs,
                               unsigned int seed,
                               int numSamples);

void monteCarloGpuKernelCpu(dataType * samplePrices,
                            dataType * sampleWeights,
                            dataType* times,
                            dataType dt,
                            monteCarloOptionStruct * optionStructs,
                            unsigned int seed,
                            int numSamples);

#endif //MONTE_CARLO_KERNELS_CPU_H
