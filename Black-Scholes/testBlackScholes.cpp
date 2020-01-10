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
#ifdef BUILD_CUDA
#include "blackScholesAnalyticEngineKernels.cuh"
#include <cuda_runtime.h>

#define CUDA_CALL(error)         \
    ASSERT_EQ(static_cast<cudaError_t>(error),cudaSuccess)
#endif

#define NUM_DIFF_SETTINGS 37


void initOptions(optionInputStruct * values,
                 int numVals)
{
    for(int numOption = 0; numOption < numVals; ++numOption)
    {
        if((numOption % NUM_DIFF_SETTINGS) == 0)
        {
            optionInputStruct currVal = { CALL,  40.00,  42.00, 0.08, 0.04, 0.75, 0.35,  5.0975, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 1)
        {
            optionInputStruct currVal = { CALL, 100.00,  90.00, 0.10, 0.10, 0.10, 0.15,  0.0205, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 2)
        {
            optionInputStruct currVal = { CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15,  1.8734, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 3)
        {
            optionInputStruct currVal = { CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15,  9.9413, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 4)
        {
            optionInputStruct currVal = { CALL, 100.00,  90.00, 0.10, 0.10, 0.10, 0.25,  0.3150, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 5)
        {
            optionInputStruct currVal = { CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.25,  3.1217, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 6)
        {
            optionInputStruct currVal = { CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.25, 10.3556, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 7)
        {
            optionInputStruct currVal =  { CALL, 100.00,  90.00, 0.10, 0.10, 0.10, 0.35,  0.9474, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 8)
        {
            optionInputStruct currVal = { CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.35,  4.3693, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 9)
        {
            optionInputStruct currVal = { CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.35, 11.1381, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 10)
        {
            optionInputStruct currVal =  { CALL, 100.00,  90.00, 0.10, 0.10, 0.50, 0.15,  0.8069, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 11)
        {
            optionInputStruct currVal =  { CALL, 100.00, 100.00, 0.10, 0.10, 0.50, 0.15,  4.0232, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 12)
        {
            optionInputStruct currVal =  { CALL, 100.00, 110.00, 0.10, 0.10, 0.50, 0.15, 10.5769, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 13)
        {
            optionInputStruct currVal =   { CALL, 100.00,  90.00, 0.10, 0.10, 0.50, 0.25,  2.7026, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 14)
        {
            optionInputStruct currVal =   { CALL, 100.00, 100.00, 0.10, 0.10, 0.50, 0.25,  6.6997, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 15)
        {
            optionInputStruct currVal =   { CALL, 100.00, 110.00, 0.10, 0.10, 0.50, 0.25, 12.7857, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 16)
        {
            optionInputStruct currVal =   { CALL, 100.00,  90.00, 0.10, 0.10, 0.50, 0.35,  4.9329, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 17)
        {
            optionInputStruct currVal =  { CALL, 100.00, 100.00, 0.10, 0.10, 0.50, 0.35,  9.3679, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 18)
        {
            optionInputStruct currVal = { CALL, 100.00, 110.00, 0.10, 0.10, 0.50, 0.35, 15.3086, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 19)
        {
            optionInputStruct currVal =  { PUT,  100.00,  90.00, 0.10, 0.10, 0.10, 0.15,  9.9210, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 20)
        {
            optionInputStruct currVal =   { PUT,  100.00, 100.00, 0.10, 0.10, 0.10, 0.15,  1.8734, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 21)
        {
            optionInputStruct currVal =   { PUT,  100.00, 110.00, 0.10, 0.10, 0.10, 0.15,  0.0408, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 22)
        {
            optionInputStruct currVal =  { PUT,  100.00,  90.00, 0.10, 0.10, 0.10, 0.25, 10.2155, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 23)
        {
            optionInputStruct currVal =   { PUT,  100.00, 100.00, 0.10, 0.10, 0.10, 0.25,  3.1217, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 24)
        {
            optionInputStruct currVal =    { PUT,  100.00, 110.00, 0.10, 0.10, 0.10, 0.25,  0.4551, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 25)
        {
            optionInputStruct currVal =  { PUT,  100.00,  90.00, 0.10, 0.10, 0.10, 0.35, 10.8479, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 26)
        {
            optionInputStruct currVal =   { PUT,  100.00, 100.00, 0.10, 0.10, 0.10, 0.35,  4.3693, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 27)
        {
            optionInputStruct currVal =  { PUT,  100.00, 110.00, 0.10, 0.10, 0.10, 0.35,  1.2376, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 28)
        {
            optionInputStruct currVal =  { PUT,  100.00,  90.00, 0.10, 0.10, 0.50, 0.15, 10.3192, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 29)
        {
            optionInputStruct currVal =   { PUT,  100.00, 100.00, 0.10, 0.10, 0.50, 0.15,  4.0232, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 30)
        {
            optionInputStruct currVal =  { PUT,  100.00, 110.00, 0.10, 0.10, 0.50, 0.15,  1.0646, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 31)
        {
            optionInputStruct currVal =  { PUT,  100.00,  90.00, 0.10, 0.10, 0.50, 0.25, 12.2149, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 32)
        {
            optionInputStruct currVal =   { PUT,  100.00, 100.00, 0.10, 0.10, 0.50, 0.25,  6.6997, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 33)
        {
            optionInputStruct currVal =   { PUT,  100.00, 110.00, 0.10, 0.10, 0.50, 0.25,  3.2734, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 34)
        {
            optionInputStruct currVal =   { PUT,  100.00,  90.00, 0.10, 0.10, 0.50, 0.35, 14.4452, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 35)
        {
            optionInputStruct currVal =  { PUT,  100.00, 100.00, 0.10, 0.10, 0.50, 0.35,  9.3679, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 36)
        {
            optionInputStruct currVal =   { PUT,  100.00, 110.00, 0.10, 0.10, 0.50, 0.35,  5.7963, 1.0e-4};
            values[numOption] = currVal;
        }
    }
}

TEST(BlackScholes, OpenMP)
{
    int numVals = 1024;
    optionInputStruct * values = new optionInputStruct[numVals];

    initOptions(values, numVals);

    float * outputCpu = (float *)malloc(numVals * sizeof(float));
    float * outputMp = (float *)malloc(numVals * sizeof(float));

    getOutValOptionCpu(values, outputCpu, numVals);
    getOutValOptionOpenMP(values, outputMp, numVals);

    for(int i = 0; i < numVals; ++i)
    {
        ASSERT_NEAR(outputCpu[i], outputMp[i], 0.001f) << "where index = " << i;;
    }

    delete [] values;
    free(outputCpu);
    free(outputMp);
}

#ifdef BUILD_CUDA
TEST(BlackScholes, Cuda)
{
    int numVals = 1024;
    optionInputStruct * values = new optionInputStruct[numVals];

    initOptions(values, numVals);

    float * outputCpu = (float *)malloc(numVals * sizeof(float));
    float * outputGpu = (float *)malloc(numVals * sizeof(float));
    optionInputStruct * optionsGpu;
    float * outputValsGpu;

    getOutValOptionCpu(values, outputCpu, numVals);

    CUDA_CALL(cudaMalloc(&optionsGpu, numVals * sizeof(optionInputStruct)));
    CUDA_CALL(cudaMalloc(&outputValsGpu, numVals * sizeof(float)));

    dim3 grid((size_t)ceil((float)numVals / (float)THREAD_BLOCK_SIZE), 1, 1);
    dim3 threads(THREAD_BLOCK_SIZE, 1, 1);

    CUDA_CALL(cudaMemcpy(optionsGpu, values, numVals * sizeof(optionInputStruct), cudaMemcpyHostToDevice));
    getOutValOption<<<grid, threads>>>(optionsGpu, outputValsGpu, numVals);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaMemcpy(outputGpu, outputValsGpu, numVals * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaDeviceSynchronize());

    for(int i = 0; i < numVals; ++i)
    {
        ASSERT_NEAR(outputCpu[i], outputGpu[i], 0.001f) << "where index = " << i;;
    }

    CUDA_CALL(cudaFree(optionsGpu));
    CUDA_CALL(cudaFree(outputValsGpu));

    delete [] values;
    free(outputCpu);
    free(outputGpu);
}

TEST(BlackScholes, CudaOpt)
{
    int numVals = 1024;
    optionInputStruct * values = new optionInputStruct[numVals];

    initOptions(values, numVals);

    float * outputCpu = (float *)malloc(numVals * sizeof(float));
    float * outputGpu = (float *)malloc(numVals * sizeof(float));
    optionInputStruct * optionsGpu;
    float * outputValsGpu;

    getOutValOptionCpu(values, outputCpu, numVals);

    CUDA_CALL(cudaMalloc(&optionsGpu, numVals * sizeof(optionInputStruct)));
    CUDA_CALL(cudaMalloc(&outputValsGpu, numVals * sizeof(float)));

    dim3 grid((size_t)ceil((float)numVals / (float)THREAD_BLOCK_SIZE), 1, 1);
    dim3 threads(THREAD_BLOCK_SIZE, 1, 1);

    CUDA_CALL(cudaMemcpy(optionsGpu, values, numVals * sizeof(optionInputStruct), cudaMemcpyHostToDevice));
    getOutValOptionOpt<<<grid, threads>>>(optionsGpu, outputValsGpu, numVals);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaMemcpy(outputGpu, outputValsGpu, numVals * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaDeviceSynchronize());

    for(int i = 0; i < numVals; ++i)
    {
        ASSERT_NEAR(outputCpu[i], outputGpu[i], 0.001f) << "where index = " << i;;
    }

    CUDA_CALL(cudaFree(optionsGpu));
    CUDA_CALL(cudaFree(outputValsGpu));

    delete [] values;
    free(outputCpu);
    free(outputGpu);
}
#endif
