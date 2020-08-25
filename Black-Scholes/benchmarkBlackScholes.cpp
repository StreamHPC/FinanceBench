#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <utility>
#include <algorithm>

// Google Benchmark
#include "benchmark/benchmark.h"
#include "cmdparser.hpp"

#include "blackScholesAnalyticEngineKernelsCpu.h"
#ifdef BUILD_HIP
#include "blackScholesAnalyticEngineKernelsGpu.h"
#include <hip/hip_runtime.h>

#define HIP_CALL(x) do { if((x)!=hipSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)
#endif

#define NUM_DIFF_SETTINGS 37
#include "initBlackScholes.h"

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 5;
#endif

const unsigned int warmup_size = 5;

void runBenchmarkCpu(benchmark::State& state,
                     size_t size)
{
    int numVals = size;
    optionInputStruct * values = new optionInputStruct[size];

    initOptions(values, size);

    float * outputVals = new float[numVals];

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        getOutValOptionCpu(values, outputVals, numVals);
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        getOutValOptionCpu(values, outputVals, numVals);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    delete [] values;
    delete [] outputVals;
}

void runBenchmarkOpenMP(benchmark::State& state,
                        size_t size)
{
    int numVals = size;
    optionInputStruct * values = new optionInputStruct[size];

    initOptions(values, size);

    float * outputVals = new float[numVals];

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        getOutValOptionOpenMP(values, outputVals, numVals);
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        getOutValOptionOpenMP(values, outputVals, numVals);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    delete [] values;
    delete [] outputVals;
}

#ifdef BUILD_HIP
void runBenchmarkHipV1(benchmark::State& state,
                        size_t size)
{
    int numVals = size;
    optionInputStruct * values = new optionInputStruct[size];

    initOptions(values, size);

    float * outputVals = new float[numVals];
    optionInputStruct * optionsGpu;
    float * outputValsGpu;

    hipStream_t stream;
    HIP_CALL(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP_CALL(hipMalloc(&optionsGpu, numVals * sizeof(optionInputStruct)));
    HIP_CALL(hipMalloc(&outputValsGpu, numVals * sizeof(float)));

    HIP_CALL(hipMemcpyAsync(optionsGpu, values, numVals * sizeof(optionInputStruct), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipStreamSynchronize(stream));

    dim3 grid((size_t)ceil((float)numVals / (float)THREAD_BLOCK_SIZE), 1, 1);
    dim3 threads( THREAD_BLOCK_SIZE, 1, 1);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        hipLaunchKernelGGL((getOutValOption), dim3(grid), dim3(threads), 0, stream, optionsGpu, outputValsGpu, numVals);
        HIP_CALL(hipPeekAtLastError());
        HIP_CALL(hipStreamSynchronize(stream));
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        hipLaunchKernelGGL((getOutValOption), dim3(grid), dim3(threads), 0, stream, optionsGpu, outputValsGpu, numVals);
        HIP_CALL(hipPeekAtLastError());
        HIP_CALL(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    HIP_CALL(hipFree(optionsGpu));
    HIP_CALL(hipFree(outputValsGpu));
    HIP_CALL(hipStreamDestroy(stream));

    delete [] values;
    delete [] outputVals;
}

void runBenchmarkHipV2(benchmark::State& state,
                        size_t size)
{
    int numVals = size;
    optionInputStruct * values = new optionInputStruct[size];

    initOptions(values, size);

    float * outputVals = new float[numVals];
    optionInputStruct * optionsGpu;
    float * outputValsGpu;

    hipStream_t stream;
    HIP_CALL(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP_CALL(hipMalloc(&optionsGpu, numVals * sizeof(optionInputStruct)));
    HIP_CALL(hipMalloc(&outputValsGpu, numVals * sizeof(float)));

    dim3 grid((size_t)ceil((float)numVals / (float)THREAD_BLOCK_SIZE), 1, 1);
    dim3 threads( THREAD_BLOCK_SIZE, 1, 1);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        hipLaunchKernelGGL((getOutValOption), dim3(grid), dim3(threads), 0, stream, optionsGpu, outputValsGpu, numVals);
        HIP_CALL(hipStreamSynchronize(stream));
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        HIP_CALL(hipMemcpyAsync(optionsGpu, values, numVals * sizeof(optionInputStruct), hipMemcpyHostToDevice, stream));
        hipLaunchKernelGGL((getOutValOption), dim3(grid), dim3(threads), 0, stream, optionsGpu, outputValsGpu, numVals);
        HIP_CALL(hipPeekAtLastError());
        HIP_CALL(hipMemcpyAsync(outputVals, outputValsGpu, numVals * sizeof(float), hipMemcpyDeviceToHost, stream));
        HIP_CALL(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    HIP_CALL(hipFree(optionsGpu));
    HIP_CALL(hipFree(outputValsGpu));
    HIP_CALL(hipStreamDestroy(stream));

    delete [] values;
    delete [] outputVals;
}

void runBenchmarkHipV3(benchmark::State& state,
                        size_t size)
{
    int numVals = size;
    optionInputStruct * values = new optionInputStruct[size];

    initOptions(values, size);

    float * outputVals = new float[numVals];
    optionInputStruct * optionsGpu;
    float * outputValsGpu;

    hipStream_t stream;
    HIP_CALL(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP_CALL(hipMalloc(&optionsGpu, numVals * sizeof(optionInputStruct)));
    HIP_CALL(hipMalloc(&outputValsGpu, numVals * sizeof(float)));

    HIP_CALL(hipMemcpyAsync(optionsGpu, values, numVals * sizeof(optionInputStruct), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipStreamSynchronize(stream));

    dim3 grid((size_t)ceil((float)numVals / (float)THREAD_BLOCK_SIZE), 1, 1);
    dim3 threads( THREAD_BLOCK_SIZE, 1, 1);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        hipLaunchKernelGGL((getOutValOptionOpt), dim3(grid), dim3(threads), 0, stream, optionsGpu, outputValsGpu, numVals);
        HIP_CALL(hipPeekAtLastError());
        HIP_CALL(hipStreamSynchronize(stream));
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        hipLaunchKernelGGL((getOutValOptionOpt), dim3(grid), dim3(threads), 0, stream, optionsGpu, outputValsGpu, numVals);
        HIP_CALL(hipPeekAtLastError());
        HIP_CALL(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    HIP_CALL(hipFree(optionsGpu));
    HIP_CALL(hipFree(outputValsGpu));
    HIP_CALL(hipStreamDestroy(stream));

    delete [] values;
    delete [] outputVals;
}

void runBenchmarkHipV4(benchmark::State& state,
                        size_t size)
{
    int numVals = size;
    optionInputStruct * values = new optionInputStruct[size];

    initOptions(values, size);

    float * outputVals = new float[numVals];
    optionInputStruct * optionsGpu;
    float * outputValsGpu;

    hipStream_t stream;
    HIP_CALL(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP_CALL(hipMalloc(&optionsGpu, numVals * sizeof(optionInputStruct)));
    HIP_CALL(hipMalloc(&outputValsGpu, numVals * sizeof(float)));

    dim3 grid((size_t)ceil((float)numVals / (float)THREAD_BLOCK_SIZE), 1, 1);
    dim3 threads( THREAD_BLOCK_SIZE, 1, 1);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        hipLaunchKernelGGL((getOutValOptionOpt), dim3(grid), dim3(threads), 0, stream, optionsGpu, outputValsGpu, numVals);
        HIP_CALL(hipStreamSynchronize(stream));
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        HIP_CALL(hipMemcpyAsync(optionsGpu, values, numVals * sizeof(optionInputStruct), hipMemcpyHostToDevice, stream));
        hipLaunchKernelGGL((getOutValOptionOpt), dim3(grid), dim3(threads), 0, stream, optionsGpu, outputValsGpu, numVals);
        HIP_CALL(hipPeekAtLastError());
        HIP_CALL(hipMemcpyAsync(outputVals, outputValsGpu, numVals * sizeof(float), hipMemcpyDeviceToHost, stream));
        HIP_CALL(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    HIP_CALL(hipFree(optionsGpu));
    HIP_CALL(hipFree(outputValsGpu));
    HIP_CALL(hipStreamDestroy(stream));

    delete [] values;
    delete [] outputVals;
}

void runBenchmarkHipV5(benchmark::State& state,
                        size_t size)
{
    int numVals = size;
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
    float * outputVals = new float[numVals];

    initOptions(type, strike, spot, q, r, t, vol, value, tol, size);

    memcpy(values, strike, numVals * sizeof(float));
    memcpy(values + numVals, spot, numVals * sizeof(float));
    memcpy(values + (numVals * 2), q, numVals * sizeof(float));
    memcpy(values + (numVals * 3), r, numVals * sizeof(float));
    memcpy(values + (numVals * 4), t, numVals * sizeof(float));
    memcpy(values + (numVals * 5), vol, numVals * sizeof(float));

    char * typeGpu;
    float * valuesGpu;
    float * outputValsGpu;

    hipStream_t stream;
    HIP_CALL(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP_CALL(hipMalloc(&typeGpu, numVals * sizeof(char)));
    HIP_CALL(hipMalloc(&valuesGpu, 6 * numVals * sizeof(float)));
    HIP_CALL(hipMalloc(&outputValsGpu, numVals * sizeof(float)));

    dim3 grid((size_t)ceil((float)numVals / (float)THREAD_BLOCK_SIZE), 1, 1);
    dim3 threads( THREAD_BLOCK_SIZE, 1, 1);

    HIP_CALL(hipMemcpyAsync(typeGpu, type, numVals * sizeof(char), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync(valuesGpu, values, 6 * numVals * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipStreamSynchronize(stream));

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        hipLaunchKernelGGL(
            (getOutValOptionOpt), dim3(grid), dim3(threads), 0, stream,
            typeGpu, valuesGpu, outputValsGpu, numVals
        );
        HIP_CALL(hipStreamSynchronize(stream));
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        hipLaunchKernelGGL(
            (getOutValOptionOpt), dim3(grid), dim3(threads), 0, stream,
            typeGpu, valuesGpu, outputValsGpu, numVals
        );
        HIP_CALL(hipPeekAtLastError());
        HIP_CALL(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    HIP_CALL(hipFree(typeGpu));
    HIP_CALL(hipFree(valuesGpu));
    HIP_CALL(hipFree(outputValsGpu));
    HIP_CALL(hipStreamDestroy(stream));

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
    delete [] outputVals;
}

void runBenchmarkHipV6(benchmark::State& state,
                        size_t size)
{
    int numVals = size;
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
    float * outputVals = new float[numVals];

    initOptions(type, strike, spot, q, r, t, vol, value, tol, size);

    memcpy(values, strike, numVals * sizeof(float));
    memcpy(values + numVals, spot, numVals * sizeof(float));
    memcpy(values + (numVals * 2), q, numVals * sizeof(float));
    memcpy(values + (numVals * 3), r, numVals * sizeof(float));
    memcpy(values + (numVals * 4), t, numVals * sizeof(float));
    memcpy(values + (numVals * 5), vol, numVals * sizeof(float));

    char * typeGpu;
    float * valuesGpu;
    float * outputValsGpu;

    hipStream_t stream;
    HIP_CALL(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP_CALL(hipMalloc(&typeGpu, numVals * sizeof(char)));
    HIP_CALL(hipMalloc(&valuesGpu, 6 * numVals * sizeof(float)));
    HIP_CALL(hipMalloc(&outputValsGpu, numVals * sizeof(float)));

    dim3 grid((size_t)ceil((float)numVals / (float)THREAD_BLOCK_SIZE), 1, 1);
    dim3 threads( THREAD_BLOCK_SIZE, 1, 1);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        hipLaunchKernelGGL(
            (getOutValOptionOpt), dim3(grid), dim3(threads), 0, stream,
            typeGpu, valuesGpu, outputValsGpu, numVals
        );
        HIP_CALL(hipStreamSynchronize(stream));
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        HIP_CALL(hipMemcpyAsync(typeGpu, type, numVals * sizeof(char), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(valuesGpu, values, 6 * numVals * sizeof(float), hipMemcpyHostToDevice, stream));
        hipLaunchKernelGGL(
            (getOutValOptionOpt), dim3(grid), dim3(threads), 0, stream,
            typeGpu, valuesGpu, outputValsGpu, numVals
        );
        HIP_CALL(hipPeekAtLastError());
        HIP_CALL(hipMemcpyAsync(outputVals, outputValsGpu, numVals * sizeof(float), hipMemcpyDeviceToHost, stream));
        HIP_CALL(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    HIP_CALL(hipFree(typeGpu));
    HIP_CALL(hipFree(valuesGpu));
    HIP_CALL(hipFree(outputValsGpu));
    HIP_CALL(hipStreamDestroy(stream));

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
    delete [] outputVals;
}

void runBenchmarkHipV7(benchmark::State& state,
                        size_t size)
{
    int numVals = size;
    char * type = new char[numVals];
    float * strike = new float[numVals];
    float * spot = new float[numVals];
    float * q = new float[numVals];
    float * r = new float[numVals];
    float * t = new float[numVals];
    float * vol = new float[numVals];
    float * value = new float[numVals];
    float * tol = new float[numVals];
    float * outputVals = new float[numVals];

    initOptions(type, strike, spot, q, r, t, vol, value, tol, size);

    char * typeGpu;
    float * strikeGpu;
    float * spotGpu;
    float * qGpu;
    float * rGpu;
    float * tGpu;
    float * volGpu;
    float * valueGpu;
    float * tolGpu;
    float * outputValsGpu;

    hipStream_t stream;
    HIP_CALL(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP_CALL(hipMalloc(&typeGpu, numVals * sizeof(char)));
    HIP_CALL(hipMalloc(&strikeGpu, numVals * sizeof(float)));
    HIP_CALL(hipMalloc(&spotGpu, numVals * sizeof(float)));
    HIP_CALL(hipMalloc(&qGpu, numVals * sizeof(float)));
    HIP_CALL(hipMalloc(&rGpu, numVals * sizeof(float)));
    HIP_CALL(hipMalloc(&tGpu, numVals * sizeof(float)));
    HIP_CALL(hipMalloc(&volGpu, numVals * sizeof(float)));
    HIP_CALL(hipMalloc(&valueGpu, numVals * sizeof(float)));
    HIP_CALL(hipMalloc(&tolGpu, numVals * sizeof(float)));
    HIP_CALL(hipMalloc(&outputValsGpu, numVals * sizeof(float)));

    optionInputStruct_ optionsGpu;
    optionsGpu.type    = typeGpu;
    optionsGpu.strike  = strikeGpu;
    optionsGpu.spot    = spotGpu;
    optionsGpu.q       = qGpu;
    optionsGpu.r       = rGpu;
    optionsGpu.t       = tGpu;
    optionsGpu.vol     = volGpu;
    optionsGpu.value   = valueGpu;
    optionsGpu.tol     = tolGpu;

    dim3 grid((size_t)ceil((float)numVals / (float)THREAD_BLOCK_SIZE), 1, 1);
    dim3 threads( THREAD_BLOCK_SIZE, 1, 1);

    HIP_CALL(hipMemcpyAsync(typeGpu, type, numVals * sizeof(char), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync(strikeGpu, strike, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync(spotGpu, spot, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync(qGpu, q, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync(rGpu, r, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync(tGpu, t, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync(volGpu, vol, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync(valueGpu, value, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync(tolGpu, tol, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipStreamSynchronize(stream));

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        hipLaunchKernelGGL(
            (getOutValOptionOpt), dim3(grid), dim3(threads), 0, stream,
            optionsGpu, outputValsGpu, numVals
        );
        HIP_CALL(hipStreamSynchronize(stream));
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        hipLaunchKernelGGL(
            (getOutValOptionOpt), dim3(grid), dim3(threads), 0, stream,
            optionsGpu, outputValsGpu, numVals
        );
        HIP_CALL(hipPeekAtLastError());
        HIP_CALL(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    HIP_CALL(hipFree(typeGpu));
    HIP_CALL(hipFree(strikeGpu));
    HIP_CALL(hipFree(spotGpu));
    HIP_CALL(hipFree(qGpu));
    HIP_CALL(hipFree(rGpu));
    HIP_CALL(hipFree(tGpu));
    HIP_CALL(hipFree(volGpu));
    HIP_CALL(hipFree(valueGpu));
    HIP_CALL(hipFree(tolGpu));
    HIP_CALL(hipFree(outputValsGpu));
    HIP_CALL(hipStreamDestroy(stream));

    delete [] type;
    delete [] strike;
    delete [] spot;
    delete [] q;
    delete [] r;
    delete [] t;
    delete [] vol;
    delete [] value;
    delete [] tol;
    delete [] outputVals;
}

void runBenchmarkHipV8(benchmark::State& state,
                        size_t size)
{
    int numVals = size;
    char * type = new char[numVals];
    float * strike = new float[numVals];
    float * spot = new float[numVals];
    float * q = new float[numVals];
    float * r = new float[numVals];
    float * t = new float[numVals];
    float * vol = new float[numVals];
    float * value = new float[numVals];
    float * tol = new float[numVals];
    float * outputVals = new float[numVals];

    initOptions(type, strike, spot, q, r, t, vol, value, tol, size);

    char * typeGpu;
    float * strikeGpu;
    float * spotGpu;
    float * qGpu;
    float * rGpu;
    float * tGpu;
    float * volGpu;
    float * valueGpu = nullptr;
    float * tolGpu = nullptr;
    float * outputValsGpu;

    hipStream_t stream;
    HIP_CALL(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP_CALL(hipMalloc(&typeGpu, numVals * sizeof(char)));
    HIP_CALL(hipMalloc(&strikeGpu, numVals * sizeof(float)));
    HIP_CALL(hipMalloc(&spotGpu, numVals * sizeof(float)));
    HIP_CALL(hipMalloc(&qGpu, numVals * sizeof(float)));
    HIP_CALL(hipMalloc(&rGpu, numVals * sizeof(float)));
    HIP_CALL(hipMalloc(&tGpu, numVals * sizeof(float)));
    HIP_CALL(hipMalloc(&volGpu, numVals * sizeof(float)));
    //HIP_CALL(hipMalloc(&valueGpu, numVals * sizeof(float)));
    //HIP_CALL(hipMalloc(&tolGpu, numVals * sizeof(float)));
    HIP_CALL(hipMalloc(&outputValsGpu, numVals * sizeof(float)));

    optionInputStruct_ optionsGpu;
    optionsGpu.type    = typeGpu;
    optionsGpu.strike  = strikeGpu;
    optionsGpu.spot    = spotGpu;
    optionsGpu.q       = qGpu;
    optionsGpu.r       = rGpu;
    optionsGpu.t       = tGpu;
    optionsGpu.vol     = volGpu;
    optionsGpu.value   = valueGpu;
    optionsGpu.tol     = tolGpu;

    dim3 grid((size_t)ceil((float)numVals / (float)THREAD_BLOCK_SIZE), 1, 1);
    dim3 threads( THREAD_BLOCK_SIZE, 1, 1);

    HIP_CALL(hipMemcpyAsync(typeGpu, type, numVals * sizeof(char), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync(strikeGpu, strike, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync(spotGpu, spot, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync(qGpu, q, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync(rGpu, r, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync(tGpu, t, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync(volGpu, vol, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
    //HIP_CALL(hipMemcpyAsync(valueGpu, value, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
    //HIP_CALL(hipMemcpyAsync(tolGpu, tol, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipStreamSynchronize(stream));

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        hipLaunchKernelGGL(
            (getOutValOptionOpt), dim3(grid), dim3(threads), 0, stream,
            optionsGpu, outputValsGpu, numVals
        );
        HIP_CALL(hipStreamSynchronize(stream));
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        HIP_CALL(hipMemcpyAsync(typeGpu, type, numVals * sizeof(char), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(strikeGpu, strike, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(spotGpu, spot, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(qGpu, q, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(rGpu, r, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(tGpu, t, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(volGpu, vol, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
        //HIP_CALL(hipMemcpyAsync(valueGpu, value, numVals * sizeof(float), hipMemcpyHostToDevice, stream));
        //HIP_CALL(hipMemcpyAsync(tolGpu, tol, numVals * sizeof(float), hipMemcpyHostToDevice, stream));

        hipLaunchKernelGGL(
            (getOutValOptionOpt), dim3(grid), dim3(threads), 0, stream,
            optionsGpu, outputValsGpu, numVals
        );
        HIP_CALL(hipMemcpyAsync(outputVals, outputValsGpu, numVals * sizeof(float), hipMemcpyDeviceToHost, stream));
        HIP_CALL(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    HIP_CALL(hipFree(typeGpu));
    HIP_CALL(hipFree(strikeGpu));
    HIP_CALL(hipFree(spotGpu));
    HIP_CALL(hipFree(qGpu));
    HIP_CALL(hipFree(rGpu));
    HIP_CALL(hipFree(tGpu));
    HIP_CALL(hipFree(volGpu));
    //HIP_CALL(hipFree(valueGpu));
    //HIP_CALL(hipFree(tolGpu));
    HIP_CALL(hipFree(outputValsGpu));
    HIP_CALL(hipStreamDestroy(stream));

    delete [] type;
    delete [] strike;
    delete [] spot;
    delete [] q;
    delete [] r;
    delete [] t;
    delete [] vol;
    delete [] value;
    delete [] tol;
    delete [] outputVals;
}
#endif

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);

    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", 10, "number of iterations");
    parser.set_optional<int>("device_id", "device_id", 0, "ID of GPU to run");
    parser.set_optional<bool>("run_cpu", "run_cpu", false, "Run single-threaded CPU version (slow)");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size = parser.get<size_t>("size");
    const int trials = parser.get<int>("trials");
    const int device_id = parser.get<int>("device_id");
    const bool run_cpu = parser.get<bool>("run_cpu");

    #ifdef BUILD_HIP
    //int device_id;
    //HIP_CALL(hipGetDevice(&device_id));
    hipDeviceProp_t props;
    HIP_CALL(hipGetDeviceProperties(&props, device_id));
    HIP_CALL(hipSetDevice(device_id));

    //std::cout << "Runtime: " << runtime_version << " ";
    std::cout << "Device: " << props.name;
    std::cout << std::endl << std::endl;
    #endif

    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        benchmark::RegisterBenchmark(
            ("blackScholes (OpenMP)"),
            [=](benchmark::State& state) { runBenchmarkOpenMP(state, size); }
        ),
        #ifdef BUILD_HIP
        benchmark::RegisterBenchmark(
            ("blackScholesHip (Compute Only)"),
            [=](benchmark::State& state) { runBenchmarkHipV1(state, size); }
        ),
        benchmark::RegisterBenchmark(
            ("blackScholesHip (+ Transfers)"),
            [=](benchmark::State& state) { runBenchmarkHipV2(state, size); }
        ),
        benchmark::RegisterBenchmark(
            ("blackScholesHipOpt (Compute Only)"),
            [=](benchmark::State& state) { runBenchmarkHipV3(state, size); }
        ),
        benchmark::RegisterBenchmark(
            ("blackScholesHipOpt (+ Transfers)"),
            [=](benchmark::State& state) { runBenchmarkHipV4(state, size); }
        ),
        benchmark::RegisterBenchmark(
            ("blackScholesHipOpt [Linear input array] (Compute Only)"),
            [=](benchmark::State& state) { runBenchmarkHipV5(state, size); }
        ),
        benchmark::RegisterBenchmark(
            ("blackScholesHipOpt [Linear input array] (+ Transfers)"),
            [=](benchmark::State& state) { runBenchmarkHipV6(state, size); }
        ),
        benchmark::RegisterBenchmark(
            ("blackScholesHipOpt [Struct of Arrays] (Compute Only)"),
            [=](benchmark::State& state) { runBenchmarkHipV7(state, size); }
        ),
        benchmark::RegisterBenchmark(
            ("blackScholesHipOpt [Struct of Arrays] (+ Transfers)"),
            [=](benchmark::State& state) { runBenchmarkHipV8(state, size); }
        ),
        #endif
    };

    if(run_cpu)
    {
        benchmarks.insert(
            benchmarks.begin(),
            benchmark::RegisterBenchmark(
            ("blackScholes (CPU)"),
            [=](benchmark::State& state) { runBenchmarkCpu(state, size); }
            )
        );
    }

    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMicrosecond);
    }

    // Force number of iterations
    if(trials > 0)
    {
        for(auto& b : benchmarks)
        {
            b->Iterations(trials);
        }
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
