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

#include "blackScholesAnalyticEngineKernelsCpu.h"
#ifdef BUILD_HIP
#include "blackScholesAnalyticEngineKernelsGpu.h"
#include <hip/hip_runtime.h>

#define HIP_CALL(error)         \
    ASSERT_EQ(static_cast<hipError_t>(error),hipSuccess)
#endif

#define NUM_DIFF_SETTINGS 37
#include "initBlackScholes.h"

TEST(BlackScholes, OpenMP)
{
    int numVals = 1024;
    optionInputStruct * values = new optionInputStruct[numVals];

    initOptions(values, numVals);

    float * outputCpu = new float[numVals];
    float * outputMp = new float[numVals];

    getOutValOptionCpu(values, outputCpu, numVals);
    getOutValOptionOpenMP(values, outputMp, numVals);

    for(int i = 0; i < numVals; ++i)
    {
        ASSERT_NEAR(outputCpu[i], outputMp[i], 0.001f) << "where index = " << i;;
    }

    delete [] values;
    delete [] outputCpu;
    delete [] outputMp;
}

#ifdef BUILD_HIP
TEST(BlackScholes, Hip)
{
    int numVals = 1024;
    optionInputStruct * values = new optionInputStruct[numVals];

    initOptions(values, numVals);

    float * outputCpu = new float[numVals];
    float * outputGpu = new float[numVals];
    optionInputStruct * optionsGpu;
    float * outputValsGpu;

    getOutValOptionCpu(values, outputCpu, numVals);

    HIP_CALL(hipMalloc(&optionsGpu, numVals * sizeof(optionInputStruct)));
    HIP_CALL(hipMalloc(&outputValsGpu, numVals * sizeof(float)));

    dim3 grid((size_t)ceil((float)numVals / (float)THREAD_BLOCK_SIZE), 1, 1);
    dim3 threads(THREAD_BLOCK_SIZE, 1, 1);

    HIP_CALL(hipMemcpy(optionsGpu, values, numVals * sizeof(optionInputStruct), hipMemcpyHostToDevice));
    hipLaunchKernelGGL((getOutValOption), dim3(grid), dim3(threads), 0, 0, optionsGpu, outputValsGpu, numVals);
    HIP_CALL(hipPeekAtLastError());
    HIP_CALL(hipMemcpy(outputGpu, outputValsGpu, numVals * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CALL(hipDeviceSynchronize());

    for(int i = 0; i < numVals; ++i)
    {
        ASSERT_NEAR(outputCpu[i], outputGpu[i], 0.001f) << "where index = " << i;;
    }

    HIP_CALL(hipFree(optionsGpu));
    HIP_CALL(hipFree(outputValsGpu));

    delete [] values;
    delete [] outputCpu;
    delete [] outputGpu;
}

TEST(BlackScholes, HipOpt)
{
    int numVals = 1024;
    optionInputStruct * values = new optionInputStruct[numVals];

    initOptions(values, numVals);

    float * outputCpu = new float[numVals];
    float * outputGpu = new float[numVals];
    optionInputStruct * optionsGpu;
    float * outputValsGpu;

    getOutValOptionCpu(values, outputCpu, numVals);

    HIP_CALL(hipMalloc(&optionsGpu, numVals * sizeof(optionInputStruct)));
    HIP_CALL(hipMalloc(&outputValsGpu, numVals * sizeof(float)));

    dim3 grid((size_t)ceil((float)numVals / (float)THREAD_BLOCK_SIZE), 1, 1);
    dim3 threads(THREAD_BLOCK_SIZE, 1, 1);

    HIP_CALL(hipMemcpy(optionsGpu, values, numVals * sizeof(optionInputStruct), hipMemcpyHostToDevice));
    hipLaunchKernelGGL((getOutValOptionOpt), dim3(grid), dim3(threads), 0, 0, optionsGpu, outputValsGpu, numVals);
    HIP_CALL(hipPeekAtLastError());
    HIP_CALL(hipMemcpy(outputGpu, outputValsGpu, numVals * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CALL(hipDeviceSynchronize());

    for(int i = 0; i < numVals; ++i)
    {
        ASSERT_NEAR(outputCpu[i], outputGpu[i], 0.001f) << "where index = " << i;;
    }

    HIP_CALL(hipFree(optionsGpu));
    HIP_CALL(hipFree(outputValsGpu));

    delete [] values;
    delete [] outputCpu;
    delete [] outputGpu;
}

TEST(BlackScholes, HipOpt2)
{
    int numVals = 1024;
    char * type = new char[numVals];
    float * strike = new float[numVals];
    float * spot = new float[numVals];
    float * q = new float[numVals];
    float * r = new float[numVals];
    float * t = new float[numVals];
    float * vol = new float[numVals];
    float * value = new float[numVals];
    float * tol = new float[numVals];
    float * values = new float[6 * numVals];

    initOptions(type, strike, spot, q, r, t, vol, value, tol, numVals);

    optionInputStruct * v = new optionInputStruct[numVals];

    initOptions(v, numVals);
    float * outputCpu = new float[numVals];
    float * outputGpu = new float[numVals];

    getOutValOptionCpu(v, outputCpu, numVals);

    memcpy(values, strike, numVals * sizeof(float));
    memcpy(values + numVals, spot, numVals * sizeof(float));
    memcpy(values + (numVals * 2), q, numVals * sizeof(float));
    memcpy(values + (numVals * 3), r, numVals * sizeof(float));
    memcpy(values + (numVals * 4), t, numVals * sizeof(float));
    memcpy(values + (numVals * 5), vol, numVals * sizeof(float));

    char * typeGpu;
    float * valuesGpu;
    float * outputValsGpu;

    HIP_CALL(hipMalloc(&typeGpu, numVals * sizeof(char)));
    HIP_CALL(hipMalloc(&valuesGpu, 6 * numVals * sizeof(float)));
    HIP_CALL(hipMalloc(&outputValsGpu, numVals * sizeof(float)));

    dim3 grid((size_t)ceil((float)numVals / (float)THREAD_BLOCK_SIZE), 1, 1);
    dim3 threads( THREAD_BLOCK_SIZE, 1, 1);

    HIP_CALL(hipMemcpy(typeGpu, type, numVals * sizeof(char), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(valuesGpu, values, 6 * numVals * sizeof(float), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(
        (getOutValOptionOpt), dim3(grid), dim3(threads), 0, 0,
        typeGpu, valuesGpu, outputValsGpu, numVals
    );
    HIP_CALL(hipPeekAtLastError());
    HIP_CALL(hipMemcpy(outputGpu, outputValsGpu, numVals * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CALL(hipDeviceSynchronize());

    for(int i = 0; i < numVals; ++i)
    {
        ASSERT_NEAR(outputCpu[i], outputGpu[i], 0.001f) << "where index = " << i;;
    }

    HIP_CALL(hipFree(typeGpu));
    HIP_CALL(hipFree(valuesGpu));
    HIP_CALL(hipFree(outputValsGpu));

    delete [] v;
    delete [] type;
    delete [] strike;
    delete [] spot;
    delete [] q;
    delete [] r;
    delete [] t;
    delete [] vol;
    delete [] value;
    delete [] tol;
    delete [] values;
    delete [] outputCpu;
    delete [] outputGpu;
}
#endif
