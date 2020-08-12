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
#ifdef BUILD_HIP
#include "monteCarloKernelsGpu.h"
#include <hip/hip_runtime.h>

#define HIP_CALL(error)         \
    ASSERT_EQ(static_cast<hipError_t>(error),hipSuccess)
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

#ifdef BUILD_HIP
TEST(MonteCarlo, Hip)
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

    hiprandState * devStates;
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

    HIP_CALL(hipMalloc((void **)&devStates, size * sizeof(hiprandState)));
    HIP_CALL(hipMalloc(&samplePricesGpu, NUM_OPTIONS * size * sizeof(dataType)));
    HIP_CALL(hipMalloc(&sampleWeightsGpu, NUM_OPTIONS * size * sizeof(dataType)));
    HIP_CALL(hipMalloc(&timesGpu, NUM_OPTIONS * size * sizeof(dataType)));
    HIP_CALL(hipMalloc(&optionStructsGpu, NUM_OPTIONS * sizeof(monteCarloOptionStruct)));

    dim3 grid((size_t)ceil((dataType)size / ((dataType)THREAD_BLOCK_SIZE)), 1, 1);
    dim3 threads(THREAD_BLOCK_SIZE, 1, 1);

    HIP_CALL(hipMemcpy(optionStructsGpu, optionStructs, NUM_OPTIONS * sizeof(monteCarloOptionStruct), hipMemcpyHostToDevice));

    hipLaunchKernelGGL((setup_kernel), dim3(grid), dim3(threads), 0, 0, devStates, seed, size);
    HIP_CALL(hipPeekAtLastError());
    hipLaunchKernelGGL((monteCarloGpuKernel), dim3(grid), dim3(threads), 0, 0,
        samplePricesGpu, sampleWeightsGpu, timesGpu,
        (1.0f / (dataType)SEQUENCE_LENGTH), devStates, optionStructsGpu,
        seed, size
    );
    HIP_CALL(hipPeekAtLastError());

    HIP_CALL(hipMemcpy(samplePricesCpu, samplePricesGpu, size * sizeof(dataType), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(sampleWeights, sampleWeightsGpu, size * sizeof(dataType), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(times, timesGpu, size * sizeof(dataType), hipMemcpyDeviceToHost));
    HIP_CALL(hipDeviceSynchronize());

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

    HIP_CALL(hipFree(devStates));
    HIP_CALL(hipFree(samplePricesGpu));
    HIP_CALL(hipFree(sampleWeightsGpu));
    HIP_CALL(hipFree(timesGpu));
    HIP_CALL(hipFree(optionStructsGpu));

    free(samplePricesCpu);
    free(samplePricesMp);
    free(sampleWeights);
    free(times);
    free(optionStructs);
}

TEST(MonteCarlo, HipOpt)
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

    #if defined(__NVCC__)
    hiprandStatePhilox4_32_10_t * devStates;
    #else
    hiprandState * devStates;
    #endif
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

    #if defined(__NVCC__)
    HIP_CALL(hipMalloc((void **)&devStates, size * sizeof(hiprandStatePhilox4_32_10_t)));
    #else
    HIP_CALL(hipMalloc((void **)&devStates, size * sizeof(hiprandState)));
    #endif
    HIP_CALL(hipMalloc(&samplePricesGpu, NUM_OPTIONS * size * sizeof(dataType)));
    HIP_CALL(hipMalloc(&sampleWeightsGpu, NUM_OPTIONS * size * sizeof(dataType)));
    HIP_CALL(hipMalloc(&timesGpu, NUM_OPTIONS * size * sizeof(dataType)));
    HIP_CALL(hipMalloc(&optionStructsGpu, NUM_OPTIONS * sizeof(monteCarloOptionStruct)));

    dim3 grid((size_t)ceil((dataType)size / ((dataType)THREAD_BLOCK_SIZE)), 1, 1);
    dim3 threads(THREAD_BLOCK_SIZE, 1, 1);

    HIP_CALL(hipMemcpy(optionStructsGpu, optionStructs, NUM_OPTIONS * sizeof(monteCarloOptionStruct), hipMemcpyHostToDevice));

    hipLaunchKernelGGL((monteCarloGpuKernel), dim3(grid), dim3(threads), 0, 0,
        samplePricesGpu, sampleWeightsGpu, timesGpu,
        (1.0f / (dataType)SEQUENCE_LENGTH), devStates, optionStructsGpu,
        seed, size
    );
    HIP_CALL(hipPeekAtLastError());

    HIP_CALL(hipMemcpy(samplePricesCpu, samplePricesGpu, size * sizeof(dataType), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(sampleWeights, sampleWeightsGpu, size * sizeof(dataType), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(times, timesGpu, size * sizeof(dataType), hipMemcpyDeviceToHost));
    HIP_CALL(hipDeviceSynchronize());

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

    HIP_CALL(hipFree(devStates));
    HIP_CALL(hipFree(samplePricesGpu));
    HIP_CALL(hipFree(sampleWeightsGpu));
    HIP_CALL(hipFree(timesGpu));
    HIP_CALL(hipFree(optionStructsGpu));

    free(samplePricesCpu);
    free(samplePricesMp);
    free(sampleWeights);
    free(times);
    free(optionStructs);
}
#endif
