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
#include "blackScholesAnalyticEngineKernels.cuh"

#include <cuda_runtime.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 5;
#endif

#define NUM_DIFF_SETTINGS 37

const unsigned int warmup_size = 5;

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

void runBenchmarkCpu(benchmark::State& state,
                     size_t size)
{
    int numVals = size;
    optionInputStruct * values = new optionInputStruct[size];

    initOptions(values, size);

    float * outputVals = (float *)malloc(numVals * sizeof(float));

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
    free(outputVals);
}

void runBenchmarkOpenMP(benchmark::State& state,
                        size_t size)
{
    int numVals = size;
    optionInputStruct * values = new optionInputStruct[size];

    initOptions(values, size);

    float * outputVals = (float *)malloc(numVals * sizeof(float));

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
    free(outputVals);
}

void runBenchmarkCudaV1(benchmark::State& state,
                        size_t size)
{
    int numVals = size;
    optionInputStruct * values = new optionInputStruct[size];

    initOptions(values, size);

    float * outputVals = (float *)malloc(numVals * sizeof(float));
    optionInputStruct * optionsGpu;
    float * outputValsGpu;

    CUDA_CALL(cudaMalloc(&optionsGpu, numVals * sizeof(optionInputStruct)));
    CUDA_CALL(cudaMalloc(&outputValsGpu, numVals * sizeof(float)));

    CUDA_CALL(cudaMemcpy(optionsGpu, values, numVals * sizeof(optionInputStruct), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaDeviceSynchronize());

    dim3 grid((size_t)ceil((float)numVals / (float)THREAD_BLOCK_SIZE), 1, 1);
    dim3 threads( THREAD_BLOCK_SIZE, 1, 1);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        getOutValOption<<<grid, threads>>>(optionsGpu, outputValsGpu, numVals);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        getOutValOption<<<grid, threads>>>(optionsGpu, outputValsGpu, numVals);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    CUDA_CALL(cudaFree(optionsGpu));
    CUDA_CALL(cudaFree(outputValsGpu));

    delete [] values;
    free(outputVals);
}

void runBenchmarkCudaV2(benchmark::State& state,
                        size_t size)
{
    int numVals = size;
    optionInputStruct * values = new optionInputStruct[size];

    initOptions(values, size);

    float * outputVals = (float *)malloc(numVals * sizeof(float));
    optionInputStruct * optionsGpu;
    float * outputValsGpu;

    CUDA_CALL(cudaMalloc(&optionsGpu, numVals * sizeof(optionInputStruct)));
    CUDA_CALL(cudaMalloc(&outputValsGpu, numVals * sizeof(float)));

    dim3 grid((size_t)ceil((float)numVals / (float)THREAD_BLOCK_SIZE), 1, 1);
    dim3 threads( THREAD_BLOCK_SIZE, 1, 1);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        getOutValOption<<<grid, threads>>>(optionsGpu, outputValsGpu, numVals);
        CUDA_CALL(cudaDeviceSynchronize());
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        CUDA_CALL(cudaMemcpy(optionsGpu, values, numVals * sizeof(optionInputStruct), cudaMemcpyHostToDevice));
        getOutValOption<<<grid, threads>>>(optionsGpu, outputValsGpu, numVals);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaMemcpy(outputVals, outputValsGpu, numVals * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    CUDA_CALL(cudaFree(optionsGpu));
    CUDA_CALL(cudaFree(outputValsGpu));

    delete [] values;
    free(outputVals);
}

void runBenchmarkCudaV3(benchmark::State& state,
                        size_t size)
{
    int numVals = size;
    optionInputStruct * values = new optionInputStruct[size];

    initOptions(values, size);

    float * outputVals = (float *)malloc(numVals * sizeof(float));
    optionInputStruct * optionsGpu;
    float * outputValsGpu;

    CUDA_CALL(cudaMalloc(&optionsGpu, numVals * sizeof(optionInputStruct)));
    CUDA_CALL(cudaMalloc(&outputValsGpu, numVals * sizeof(float)));

    CUDA_CALL(cudaMemcpy(optionsGpu, values, numVals * sizeof(optionInputStruct), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaDeviceSynchronize());

    dim3 grid((size_t)ceil((float)numVals / (float)THREAD_BLOCK_SIZE), 1, 1);
    dim3 threads( THREAD_BLOCK_SIZE, 1, 1);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        getOutValOptionOpt<<<grid, threads>>>(optionsGpu, outputValsGpu, numVals);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        getOutValOptionOpt<<<grid, threads>>>(optionsGpu, outputValsGpu, numVals);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    CUDA_CALL(cudaFree(optionsGpu));
    CUDA_CALL(cudaFree(outputValsGpu));

    delete [] values;
    free(outputVals);
}

void runBenchmarkCudaV4(benchmark::State& state,
                        size_t size)
{
    int numVals = size;
    optionInputStruct * values = new optionInputStruct[size];

    initOptions(values, size);

    float * outputVals = (float *)malloc(numVals * sizeof(float));
    optionInputStruct * optionsGpu;
    float * outputValsGpu;

    CUDA_CALL(cudaMalloc(&optionsGpu, numVals * sizeof(optionInputStruct)));
    CUDA_CALL(cudaMalloc(&outputValsGpu, numVals * sizeof(float)));

    dim3 grid((size_t)ceil((float)numVals / (float)THREAD_BLOCK_SIZE), 1, 1);
    dim3 threads( THREAD_BLOCK_SIZE, 1, 1);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        getOutValOptionOpt<<<grid, threads>>>(optionsGpu, outputValsGpu, numVals);
        CUDA_CALL(cudaDeviceSynchronize());
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        CUDA_CALL(cudaMemcpy(optionsGpu, values, numVals * sizeof(optionInputStruct), cudaMemcpyHostToDevice));
        getOutValOptionOpt<<<grid, threads>>>(optionsGpu, outputValsGpu, numVals);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaMemcpy(outputVals, outputValsGpu, numVals * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    CUDA_CALL(cudaFree(optionsGpu));
    CUDA_CALL(cudaFree(outputValsGpu));

    delete [] values;
    free(outputVals);
}

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);

    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", 10, "number of iterations");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size = parser.get<size_t>("size");
    const int trials = parser.get<int>("trials");

    int runtime_version;
    CUDA_CALL(cudaRuntimeGetVersion(&runtime_version));
    int device_id;
    CUDA_CALL(cudaGetDevice(&device_id));
    cudaDeviceProp props;
    CUDA_CALL(cudaGetDeviceProperties(&props, device_id));

    std::cout << "Runtime: " << runtime_version << " ";
    std::cout << "Device: " << props.name;
    std::cout << std::endl << std::endl;

    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        benchmark::RegisterBenchmark(
            ("blackScholes (CPU)"),
            [=](benchmark::State& state) { runBenchmarkCpu(state, size); }
        ),
        benchmark::RegisterBenchmark(
            ("blackScholes (OpenMP)"),
            [=](benchmark::State& state) { runBenchmarkOpenMP(state, size); }
        ),
        benchmark::RegisterBenchmark(
            ("blackScholesCuda (Compute Only)"),
            [=](benchmark::State& state) { runBenchmarkCudaV1(state, size); }
        ),
        benchmark::RegisterBenchmark(
            ("blackScholesCuda (+ Transfers)"),
            [=](benchmark::State& state) { runBenchmarkCudaV2(state, size); }
        ),
        benchmark::RegisterBenchmark(
            ("blackScholesCudaOpt (Compute Only)"),
            [=](benchmark::State& state) { runBenchmarkCudaV3(state, size); }
        ),
        benchmark::RegisterBenchmark(
            ("blackScholesCudaOpt (+ Transfers)"),
            [=](benchmark::State& state) { runBenchmarkCudaV4(state, size); }
        ),
    };

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
