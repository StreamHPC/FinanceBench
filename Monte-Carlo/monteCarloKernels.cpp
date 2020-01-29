//monteCarloKernels.cu
//Scott Grauer-Gray
//May 10, 2012
//GPU Kernels for running monte carlo

#include "monteCarloKernels.h"

//function to set up the random states
__global__ void setup_kernel(hiprandState * state,
                             int seedVal,
                             int numSamples)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < numSamples)
    {
        /* Each thread gets same seed , a different sequence
        number , no offset */
        hiprand_init(seedVal, id, 0, &(state[id]));
    }
}

__device__ inline void getPathV1(dataType * path,
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

__device__ inline void getPathV2(dataType * path,
                                 size_t sampleNum,
                                 dataType dt,
                                 hiprandState * state,
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

//template<>
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

__global__ void monteCarloGpuKernel(dataType * samplePrices,
                                    dataType * sampleWeights,
                                    dataType * times,
                                    dataType dt,
                                    hiprandState * state,
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
        #if defined(__HCC__)
        hiprand_init(seedVal, numSample, 0, &(state[numSample]));
        #endif
        //declare and initialize the path
        dataType path[SEQUENCE_LENGTH];
        initializePath(path);

        #if defined(__HCC__)
        getPathV2(path, numSample, dt, state, optionStructs[numOption]);
        #else
        getPathV1(path, numSample, dt, state, optionStructs[numOption]);
        #endif
        dataType price = getPrice(path[SEQUENCE_LENGTH - 1]);

        samplePrices[outputNum] = price;
        sampleWeights[outputNum] = DEFAULT_SEQ_WEIGHT;

        //increase the sample and output number if processing multiple samples per thread
        //    numSample += NUM_THREADS_PER_OPTION;
        //    outputNum += NUM_THREADS_PER_OPTION;
    }
}

__global__ void monteCarloGpuKernel(dataType * samplePrices,
                                    dataType * sampleWeights,
                                    dataType * times,
                                    dataType dt,
                                    hiprandStatePhilox4_32_10_t * state,
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
        hiprand_init(seedVal, numSample, 0, &(state[numSample]));
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
