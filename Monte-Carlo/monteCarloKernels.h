
//monteCarloKernels.cuh
//Scott Grauer-Gray
//May 10, 2012
//Kernel headers for running monte carlo on the GPU

#ifndef MONTE_CARLO_KERNELS_H
#define MONTE_CARLO_KERNELS_H

//needed for curand
#include <hip/hip_runtime.h>
#include <hiprand_kernel.h>
#include <type_traits>

//needed for constants related to monte carlo
#include "monteCarloFuncs.h"

//function to set up the random states
__global__ void setup_kernel(hiprandState * state,
                             int seedVal,
                             int numSamples);

/*template<class HiprandState>
__device__ inline void getPath(dataType * path,
                               size_t sampleNum,
                               dataType dt,
                               HiprandState * state,
                               monteCarloOptionStruct optionStruct);*/

//template<>
__device__ inline void getPath(dataType * path,
                               size_t sampleNum,
                               dataType dt,
                               hiprandState * state,
                               monteCarloOptionStruct optionStruct);

//template<>
__device__ inline void getPath(dataType * path,
                               size_t sampleNum,
                               dataType dt,
                               #if defined(__NVCC__)
                               hiprandStatePhilox4_32_10_t * state,
                               #else

                               #endif
                               monteCarloOptionStruct optionStruct);

__global__ void monteCarloGpuKernel(dataType * samplePrices,
                                    dataType * sampleWeights,
                                    dataType * times,
                                    dataType dt,
                                    hiprandState * state,
                                    monteCarloOptionStruct * optionStructs,
                                    int seedVal,
                                    int numSamples);

__global__ void monteCarloGpuKernel(dataType * samplePrices,
                                    dataType * sampleWeights,
                                    dataType * times,
                                    dataType dt,
                                    #if defined(__NVCC__)
                                    hiprandStatePhilox4_32_10_t * state,
                                    #else

                                    #endif
                                    monteCarloOptionStruct * optionStructs,
                                    int seedVal,
                                    int numSamples);

#endif //MONTE_CARLO_KERNELS_H
