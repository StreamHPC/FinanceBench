#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <utility>
#include <algorithm>

// Google Test
#include <gtest/gtest.h>

#include "monteCarloKernelsCpu.h"
#ifdef BUILD_CUDA
#include "monteCarloKernels.cuh"
#include <cuda_runtime.h>

#define CUDA_CALL(error)         \
    ASSERT_EQ(static_cast<cudaError_t>(error),cudaSuccess)
#endif

#define RISK_VAL 0.06f
#define DIV_VAL 0.0f
#define VOLT_VAL 0.200f
#define UNDERLYING_VAL 30.0f
#define STRIKE_VAL 40.0f
#define DISCOUNT_VAL 0.94176453358424872f



    void initOptions(monteCarloOptionStruct * optionStructs)
    {
        monteCarloOptionStruct optionStruct;
        optionStruct.riskVal = RISK_VAL;
        optionStruct.divVal = DIV_VAL;
        optionStruct.voltVal = VOLT_VAL;
        optionStruct.underlyingVal = UNDERLYING_VAL;
        optionStruct.strikeVal = STRIKE_VAL;
        optionStruct.discountVal = DISCOUNT_VAL;

        for(int optNum = 0; optNum < NUM_OPTIONS; ++optNum)
        {
            optionStructs[optNum] = optionStruct;
        }
    }

TEST(MonteCarlo, OpenMP)
{
    const int size = 1024;
    const int seed = 123;
    srand(seed);

    monteCarloOptionStruct * optionStructs;
    optionStructs = (monteCarloOptionStruct *)malloc(NUM_OPTIONS * sizeof(monteCarloOptionStruct));
    initOptions(optionStructs);

    dataType * samplePricesCpu;
    dataType * samplePricesMp;
    dataType * sampleWeights;
    dataType * times;

    samplePricesCpu = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    samplePricesMp = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    sampleWeights = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    times = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));

    monteCarloKernelCpu(
        samplePricesCpu, sampleWeights, times,
        (1.0f / (dataType)SEQUENCE_LENGTH), optionStructs,
        seed, size
    );

    monteCarloKernelOpenMP(
        samplePricesMp, sampleWeights, times,
        (1.0f / (dataType)SEQUENCE_LENGTH), optionStructs,
        seed, size
    );

    dataType cumPriceCpu = 0.0f;
    dataType cumPriceMp = 0.0f;

    for(int numSamp = 0; numSamp < size; ++numSamp)
    {
        cumPriceCpu += samplePricesCpu[numSamp];
        cumPriceMp += samplePricesMp[numSamp];
    }

    cumPriceCpu /= size;
    cumPriceMp /= size;

    ASSERT_NEAR(cumPriceCpu, cumPriceMp, 0.5f);

    free(samplePricesCpu);
    free(samplePricesMp);
    free(sampleWeights);
    free(times);
    free(optionStructs);
}

#ifdef BUILD_CUDA
TEST(MonteCarlo, Cuda)
{
    const int size = 1024;
    const int seed = 123;
    srand(seed);

    monteCarloOptionStruct * optionStructs;
    optionStructs = (monteCarloOptionStruct *)malloc(NUM_OPTIONS * sizeof(monteCarloOptionStruct));
    initOptions(optionStructs);

    dataType * samplePricesCpu;
    dataType * samplePricesMp;
    dataType * sampleWeights;
    dataType * times;

    curandState * devStates;
    dataType * samplePricesGpu;
    dataType * sampleWeightsGpu;
    dataType * timesGpu;
    monteCarloOptionStruct * optionStructsGpu;

    samplePricesCpu = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    samplePricesMp = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    sampleWeights = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    times = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));

    monteCarloKernelCpu(
        samplePricesMp, sampleWeights, times,
        (1.0f / (dataType)SEQUENCE_LENGTH), optionStructs,
        seed, size
    );

    CUDA_CALL(cudaMalloc((void **)&devStates, size * sizeof(curandState)));
    CUDA_CALL(cudaMalloc(&samplePricesGpu, NUM_OPTIONS * size * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&sampleWeightsGpu, NUM_OPTIONS * size * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&timesGpu, NUM_OPTIONS * size * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&optionStructsGpu, NUM_OPTIONS * sizeof(monteCarloOptionStruct)));

    dim3 grid((size_t)ceil((dataType)size / ((dataType)THREAD_BLOCK_SIZE)), 1, 1);
    dim3 threads(THREAD_BLOCK_SIZE, 1, 1);

    CUDA_CALL(cudaMemcpy(optionStructsGpu, optionStructs, NUM_OPTIONS * sizeof(monteCarloOptionStruct), cudaMemcpyHostToDevice));

    setup_kernel<<<grid, threads>>>(devStates, seed, size);
    CUDA_CALL(cudaPeekAtLastError());
    monteCarloGpuKernel<<<grid, threads>>>(
        samplePricesGpu, sampleWeightsGpu, timesGpu,
        (1.0f / (dataType)SEQUENCE_LENGTH), devStates, optionStructsGpu,
        seed, size
    );
    CUDA_CALL(cudaPeekAtLastError());

    CUDA_CALL(cudaMemcpy(samplePricesCpu, samplePricesGpu, size * sizeof(dataType), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(sampleWeights, sampleWeightsGpu, size * sizeof(dataType), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(times, timesGpu, size * sizeof(dataType), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaDeviceSynchronize());

    dataType cumPriceCpu = 0.0f;
    dataType cumPriceMp = 0.0f;

    for(int numSamp = 0; numSamp < size; ++numSamp)
    {
        cumPriceCpu += samplePricesCpu[numSamp];
        cumPriceMp += samplePricesMp[numSamp];
    }

    cumPriceCpu /= size;
    cumPriceMp /= size;

    ASSERT_NEAR(cumPriceCpu, cumPriceMp, 0.5f);

    CUDA_CALL(cudaFree(devStates));
    CUDA_CALL(cudaFree(samplePricesGpu));
    CUDA_CALL(cudaFree(sampleWeightsGpu));
    CUDA_CALL(cudaFree(timesGpu));
    CUDA_CALL(cudaFree(optionStructsGpu));

    free(samplePricesCpu);
    free(samplePricesMp);
    free(sampleWeights);
    free(times);
    free(optionStructs);
}

TEST(MonteCarlo, CudaOpt)
{
    const int size = 1024;
    const int seed = 123;
    srand(seed);

    monteCarloOptionStruct * optionStructs;
    optionStructs = (monteCarloOptionStruct *)malloc(NUM_OPTIONS * sizeof(monteCarloOptionStruct));
    initOptions(optionStructs);

    dataType * samplePricesCpu;
    dataType * samplePricesMp;
    dataType * sampleWeights;
    dataType * times;

    curandStatePhilox4_32_10_t * devStates;
    dataType * samplePricesGpu;
    dataType * sampleWeightsGpu;
    dataType * timesGpu;
    monteCarloOptionStruct * optionStructsGpu;

    samplePricesCpu = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    samplePricesMp = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    sampleWeights = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    times = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));

    monteCarloKernelCpu(
        samplePricesMp, sampleWeights, times,
        (1.0f / (dataType)SEQUENCE_LENGTH), optionStructs,
        seed, size
    );

    CUDA_CALL(cudaMalloc((void **)&devStates, size * sizeof(curandStatePhilox4_32_10_t)));
    CUDA_CALL(cudaMalloc(&samplePricesGpu, NUM_OPTIONS * size * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&sampleWeightsGpu, NUM_OPTIONS * size * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&timesGpu, NUM_OPTIONS * size * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&optionStructsGpu, NUM_OPTIONS * sizeof(monteCarloOptionStruct)));

    dim3 grid((size_t)ceil((dataType)size / ((dataType)THREAD_BLOCK_SIZE)), 1, 1);
    dim3 threads(THREAD_BLOCK_SIZE, 1, 1);

    CUDA_CALL(cudaMemcpy(optionStructsGpu, optionStructs, NUM_OPTIONS * sizeof(monteCarloOptionStruct), cudaMemcpyHostToDevice));

    monteCarloGpuKernel<<<grid, threads>>>(
        samplePricesGpu, sampleWeightsGpu, timesGpu,
        (1.0f / (dataType)SEQUENCE_LENGTH), devStates, optionStructsGpu,
        seed, size
    );
    CUDA_CALL(cudaPeekAtLastError());

    CUDA_CALL(cudaMemcpy(samplePricesCpu, samplePricesGpu, size * sizeof(dataType), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(sampleWeights, sampleWeightsGpu, size * sizeof(dataType), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(times, timesGpu, size * sizeof(dataType), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaDeviceSynchronize());

    dataType cumPriceCpu = 0.0f;
    dataType cumPriceMp = 0.0f;

    for(int numSamp = 0; numSamp < size; ++numSamp)
    {
        cumPriceCpu += samplePricesCpu[numSamp];
        cumPriceMp += samplePricesMp[numSamp];
    }

    cumPriceCpu /= size;
    cumPriceMp /= size;

    ASSERT_NEAR(cumPriceCpu, cumPriceMp, 0.5f);

    CUDA_CALL(cudaFree(devStates));
    CUDA_CALL(cudaFree(samplePricesGpu));
    CUDA_CALL(cudaFree(sampleWeightsGpu));
    CUDA_CALL(cudaFree(timesGpu));
    CUDA_CALL(cudaFree(optionStructsGpu));

    free(samplePricesCpu);
    free(samplePricesMp);
    free(sampleWeights);
    free(times);
    free(optionStructs);
}
#endif
