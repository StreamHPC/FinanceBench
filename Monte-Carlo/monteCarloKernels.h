
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

template<class HiprandState>
__device__ inline void getPath(dataType * path,
                               size_t sampleNum,
                               dataType dt,
                               HiprandState * state,
                               monteCarloOptionStruct optionStruct);

template<>
__device__ inline void getPath(dataType * path,
                               size_t sampleNum,
                               dataType dt,
                               hiprandState * state,
                               monteCarloOptionStruct optionStruct)
{
    path[0] = getProcessValX0(optionStruct);

    for(size_t i = 1; i < SEQUENCE_LENGTH; ++i)
    {
        dataType t = i * dt;
        dataType randVal = hiprand_uniform(&(state[sampleNum]));
        dataType inverseCumRandVal = compInverseNormDist(randVal);
        path[i] = processEvolve(
                      t, path[i - 1], dt, inverseCumRandVal, optionStruct
                  );
    }
}

template<>
__device__ inline void getPath(dataType * path,
                               size_t sampleNum,
                               dataType dt,
                               hiprandStatePhilox4_32_10_t * state,
                               monteCarloOptionStruct optionStruct)
{
    path[0] = getProcessValX0(optionStruct);

    for(size_t i = 1; i < SEQUENCE_LENGTH; ++i)
    {
        dataType t = i * dt;
        //dataType randVal = curand_uniform(&(state[sampleNum]));
        //dataType inverseCumRandVal = compInverseNormDist(randVal);
        dataType inverseCumRandVal = hiprand_normal(&(state[sampleNum]));
        path[i] = processEvolve(
                      t, path[i - 1], dt, inverseCumRandVal, optionStruct
                  );
    }
}

template<class HiprandState>
__global__ void monteCarloGpuKernel(dataType * samplePrices,
                                    dataType * sampleWeights,
                                    dataType * times,
                                    dataType dt,
                                    HiprandState * state,
                                    monteCarloOptionStruct * optionStructs,
                                    int seedVal,
                                    int numSamples)
{
    //retrieve the thread number
    size_t numThread = blockIdx.x * blockDim.x + threadIdx.x;

    //retrieve the option number
    int numOption = 0;

    //retrieve the number of sample
    int numSample = numThread;

    size_t outputNum = numSample;

    //while (numSample < numSamples)
    if(numSample < numSamples)
    {
        if(std::is_same<HiprandState, hiprandStatePhilox4_32_10_t>::value)
        {
            hiprand_init(seedVal, numSample, 0, &(state[numSample]));
        }
        //declare and initialize the path
        dataType path[SEQUENCE_LENGTH];
        initializePath(path);

        getPath(path, numSample, dt, state, optionStructs[numOption]);
        dataType price = getPrice(path[SEQUENCE_LENGTH - 1]);

        samplePrices[outputNum] = price;
        sampleWeights[outputNum] = DEFAULT_SEQ_WEIGHT;

        //increase the sample and output number if processing multiple samples per thread
        //    numSample += NUM_THREADS_PER_OPTION;
        //    outputNum += NUM_THREADS_PER_OPTION;
    }
}

#endif //MONTE_CARLO_KERNELS_H
