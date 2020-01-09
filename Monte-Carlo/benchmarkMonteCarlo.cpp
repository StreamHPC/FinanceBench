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

#include "monteCarloKernelsCpu.h"
#include "monteCarloKernels.cuh"

#include <cuda_runtime.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 512;
#endif

#define RISK_VAL 0.06f
#define DIV_VAL 0.0f
#define VOLT_VAL 0.200f
#define UNDERLYING_VAL 30.0f
#define STRIKE_VAL 40.0f
#define DISCOUNT_VAL 0.94176453358424872f

const unsigned int warmup_size = 5;

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

void runBenchmarkCpu(benchmark::State& state,
                     int seed,
                     size_t size)
{
    monteCarloOptionStruct * optionStructs;
    optionStructs = (monteCarloOptionStruct *)malloc(NUM_OPTIONS * sizeof(monteCarloOptionStruct));
    initOptions(optionStructs);

    dataType * samplePrices;
    dataType * sampleWeights;
    dataType * times;

    samplePrices = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    sampleWeights = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    times = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        monteCarloKernelCpu(
            samplePrices, sampleWeights, times,
            (1.0f / (dataType)SEQUENCE_LENGTH), optionStructs,
            seed, size
        );
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        monteCarloKernelCpu(
            samplePrices, sampleWeights, times,
            (1.0f / (dataType)SEQUENCE_LENGTH), optionStructs,
            seed, size
        );

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    free(samplePrices);
    free(sampleWeights);
    free(times);
    free(optionStructs);
}

void runBenchmarkOpenMP(benchmark::State& state,
                        int seed,
                        size_t size)
{
    monteCarloOptionStruct * optionStructs;
    optionStructs = (monteCarloOptionStruct *)malloc(NUM_OPTIONS * sizeof(monteCarloOptionStruct));
    initOptions(optionStructs);

    dataType * samplePrices;
    dataType * sampleWeights;
    dataType * times;

    //allocate space for data on CPU
    samplePrices = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    sampleWeights = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    times = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        monteCarloKernelOpenMP(
            samplePrices, sampleWeights, times,
            (1.0f / (dataType)SEQUENCE_LENGTH), optionStructs,
            seed, size
        );
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        monteCarloKernelOpenMP(
            samplePrices, sampleWeights, times,
            (1.0f / (dataType)SEQUENCE_LENGTH), optionStructs,
            seed, size
        );

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    free(samplePrices);
    free(sampleWeights);
    free(times);
    free(optionStructs);
}

void runBenchmarkCudaV1(benchmark::State& state,
                        int seed,
                        size_t size)
{
    monteCarloOptionStruct * optionStructs;
    optionStructs = (monteCarloOptionStruct *)malloc(NUM_OPTIONS * sizeof(monteCarloOptionStruct));
    initOptions(optionStructs);

    dataType * samplePrices;
    dataType * sampleWeights;
    dataType * times;

    curandState * devStates;
    dataType * samplePricesGpu;
    dataType * sampleWeightsGpu;
    dataType * timesGpu;
    monteCarloOptionStruct * optionStructsGpu;

    //allocate space for data on CPU
    samplePrices = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    sampleWeights = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    times = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));

    CUDA_CALL(cudaMalloc((void **)&devStates, size * sizeof(curandState)));
    CUDA_CALL(cudaMalloc(&samplePricesGpu, NUM_OPTIONS * size * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&sampleWeightsGpu, NUM_OPTIONS * size * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&timesGpu, NUM_OPTIONS * size * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&optionStructsGpu, NUM_OPTIONS * sizeof(monteCarloOptionStruct)));

    CUDA_CALL(cudaMemcpy(samplePricesGpu, samplePrices, NUM_OPTIONS * size * sizeof(dataType), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(sampleWeightsGpu, sampleWeights, NUM_OPTIONS * size * sizeof(dataType), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(timesGpu, times, NUM_OPTIONS * size * sizeof(dataType), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(optionStructsGpu, optionStructs, NUM_OPTIONS * sizeof(monteCarloOptionStruct), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaDeviceSynchronize());

    dim3 grid((size_t)ceil((dataType)size / ((dataType)THREAD_BLOCK_SIZE)), 1, 1);
    dim3 threads(THREAD_BLOCK_SIZE, 1, 1);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        setup_kernel<<<grid, threads>>>(devStates, seed, size);
        CUDA_CALL(cudaPeekAtLastError());
        monteCarloGpuKernel<<<grid, threads>>>(
            samplePricesGpu, sampleWeightsGpu, timesGpu,
            (1.0f / (dataType)SEQUENCE_LENGTH), devStates, optionStructsGpu,
            seed, size
        );
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        setup_kernel<<<grid, threads>>>(devStates, seed, size);
        CUDA_CALL(cudaPeekAtLastError());
        monteCarloGpuKernel<<<grid, threads>>>(
            samplePricesGpu, sampleWeightsGpu, timesGpu,
            (1.0f / (dataType)SEQUENCE_LENGTH), devStates, optionStructsGpu,
            seed, size
        );
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    CUDA_CALL(cudaFree(samplePricesGpu));
    CUDA_CALL(cudaFree(sampleWeightsGpu));
    CUDA_CALL(cudaFree(timesGpu));
    CUDA_CALL(cudaFree(optionStructsGpu));

    free(samplePrices);
    free(sampleWeights);
    free(times);
    free(optionStructs);
}

void runBenchmarkCudaV2(benchmark::State& state,
                        int seed,
                        size_t size)
{
    monteCarloOptionStruct * optionStructs;
    optionStructs = (monteCarloOptionStruct *)malloc(NUM_OPTIONS * sizeof(monteCarloOptionStruct));
    initOptions(optionStructs);

    dataType * samplePrices;
    dataType * sampleWeights;
    dataType * times;

    curandState * devStates;
    dataType * samplePricesGpu;
    dataType * sampleWeightsGpu;
    dataType * timesGpu;
    monteCarloOptionStruct * optionStructsGpu;

    //allocate space for data on CPU
    samplePrices = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    sampleWeights = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    times = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));

    CUDA_CALL(cudaMalloc((void **)&devStates, size * sizeof(curandState)));
    CUDA_CALL(cudaMalloc(&samplePricesGpu, NUM_OPTIONS * size * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&sampleWeightsGpu, NUM_OPTIONS * size * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&timesGpu, NUM_OPTIONS * size * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&optionStructsGpu, NUM_OPTIONS * sizeof(monteCarloOptionStruct)));

    dim3 grid((size_t)ceil((dataType)size / ((dataType)THREAD_BLOCK_SIZE)), 1, 1);
    dim3 threads(THREAD_BLOCK_SIZE, 1, 1);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        setup_kernel<<<grid, threads>>>(devStates, seed, size);
        //cudaDeviceSynchronize();
        monteCarloGpuKernel<<<grid, threads>>>(
            samplePricesGpu, sampleWeightsGpu, timesGpu,
            (1.0f / (dataType)SEQUENCE_LENGTH), devStates, optionStructsGpu,
            seed, size
        );
        CUDA_CALL(cudaDeviceSynchronize());
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        CUDA_CALL(cudaMemcpy(samplePricesGpu, samplePrices, NUM_OPTIONS * size * sizeof(dataType), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(sampleWeightsGpu, sampleWeights, NUM_OPTIONS * size * sizeof(dataType), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(timesGpu, times, NUM_OPTIONS * size * sizeof(dataType), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(optionStructsGpu, optionStructs, NUM_OPTIONS * sizeof(monteCarloOptionStruct), cudaMemcpyHostToDevice));

        setup_kernel<<<grid, threads>>>(devStates, seed, size);
        CUDA_CALL(cudaPeekAtLastError());
        monteCarloGpuKernel<<<grid, threads>>>(
            samplePricesGpu, sampleWeightsGpu, timesGpu,
            (1.0f / (dataType)SEQUENCE_LENGTH), devStates, optionStructsGpu,
            seed, size
        );
        CUDA_CALL(cudaPeekAtLastError());

        CUDA_CALL(cudaMemcpy(samplePrices, samplePricesGpu, size * sizeof(dataType), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(sampleWeights, sampleWeightsGpu, size * sizeof(dataType), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(times, timesGpu, size * sizeof(dataType), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    CUDA_CALL(cudaFree(samplePricesGpu));
    CUDA_CALL(cudaFree(sampleWeightsGpu));
    CUDA_CALL(cudaFree(timesGpu));
    CUDA_CALL(cudaFree(optionStructsGpu));

    free(samplePrices);
    free(sampleWeights);
    free(times);
    free(optionStructs);
}

void runBenchmarkCudaV3(benchmark::State& state,
                        int seed,
                        size_t size)
{
    monteCarloOptionStruct * optionStructs;
    optionStructs = (monteCarloOptionStruct *)malloc(NUM_OPTIONS * sizeof(monteCarloOptionStruct));
    initOptions(optionStructs);

    dataType * samplePrices;
    dataType * sampleWeights;
    dataType * times;

    curandStatePhilox4_32_10_t * devStates;
    dataType * samplePricesGpu;
    dataType * sampleWeightsGpu;
    dataType * timesGpu;
    monteCarloOptionStruct * optionStructsGpu;

    //allocate space for data on CPU
    samplePrices = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    sampleWeights = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    times = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));

    CUDA_CALL(cudaMalloc((void **)&devStates, size * sizeof(curandStatePhilox4_32_10_t)));
    CUDA_CALL(cudaMalloc(&samplePricesGpu, NUM_OPTIONS * size * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&sampleWeightsGpu, NUM_OPTIONS * size * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&timesGpu, NUM_OPTIONS * size * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&optionStructsGpu, NUM_OPTIONS * sizeof(monteCarloOptionStruct)));

    CUDA_CALL(cudaMemcpy(samplePricesGpu, samplePrices, NUM_OPTIONS * size * sizeof(dataType), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(sampleWeightsGpu, sampleWeights, NUM_OPTIONS * size * sizeof(dataType), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(timesGpu, times, NUM_OPTIONS * size * sizeof(dataType), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(optionStructsGpu, optionStructs, NUM_OPTIONS * sizeof(monteCarloOptionStruct), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaDeviceSynchronize());

    dim3 grid((size_t)ceil((dataType)size / ((dataType)THREAD_BLOCK_SIZE)), 1, 1);
    dim3 threads(THREAD_BLOCK_SIZE, 1, 1);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        monteCarloGpuKernel<<<grid, threads>>>(
            samplePricesGpu, sampleWeightsGpu, timesGpu,
            (1.0f / (dataType)SEQUENCE_LENGTH), devStates, optionStructsGpu,
            seed, size
        );
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        monteCarloGpuKernel<<<grid, threads>>>(
            samplePricesGpu, sampleWeightsGpu, timesGpu,
            (1.0f / (dataType)SEQUENCE_LENGTH), devStates, optionStructsGpu,
            seed, size
        );
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    CUDA_CALL(cudaFree(samplePricesGpu));
    CUDA_CALL(cudaFree(sampleWeightsGpu));
    CUDA_CALL(cudaFree(timesGpu));
    CUDA_CALL(cudaFree(optionStructsGpu));

    free(samplePrices);
    free(sampleWeights);
    free(times);
    free(optionStructs);
}

void runBenchmarkCudaV4(benchmark::State& state,
                        int seed,
                        size_t size)
{
    monteCarloOptionStruct * optionStructs;
    optionStructs = (monteCarloOptionStruct *)malloc(NUM_OPTIONS * sizeof(monteCarloOptionStruct));
    initOptions(optionStructs);

    dataType * samplePrices;
    dataType * sampleWeights;
    dataType * times;

    curandStatePhilox4_32_10_t * devStates;
    dataType * samplePricesGpu;
    dataType * sampleWeightsGpu;
    dataType * timesGpu;
    monteCarloOptionStruct * optionStructsGpu;

    //allocate space for data on CPU
    samplePrices = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    sampleWeights = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    times = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));

    CUDA_CALL(cudaMalloc((void **)&devStates, size * sizeof(curandStatePhilox4_32_10_t)));
    CUDA_CALL(cudaMalloc(&samplePricesGpu, NUM_OPTIONS * size * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&sampleWeightsGpu, NUM_OPTIONS * size * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&timesGpu, NUM_OPTIONS * size * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&optionStructsGpu, NUM_OPTIONS * sizeof(monteCarloOptionStruct)));

    dim3 grid((size_t)ceil((dataType)size / ((dataType)THREAD_BLOCK_SIZE)), 1, 1);
    dim3 threads(THREAD_BLOCK_SIZE, 1, 1);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        monteCarloGpuKernel<<<grid, threads>>>(
            samplePricesGpu, sampleWeightsGpu, timesGpu,
            (1.0f / (dataType)SEQUENCE_LENGTH), devStates, optionStructsGpu,
            seed, size
        );
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        CUDA_CALL(cudaMemcpy(samplePricesGpu, samplePrices, NUM_OPTIONS * size * sizeof(dataType), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(sampleWeightsGpu, sampleWeights, NUM_OPTIONS * size * sizeof(dataType), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(timesGpu, times, NUM_OPTIONS * size * sizeof(dataType), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(optionStructsGpu, optionStructs, NUM_OPTIONS * sizeof(monteCarloOptionStruct), cudaMemcpyHostToDevice));

        monteCarloGpuKernel<<<grid, threads>>>(
            samplePricesGpu, sampleWeightsGpu, timesGpu,
            (1.0f / (dataType)SEQUENCE_LENGTH), devStates, optionStructsGpu,
            seed, size
        );
        CUDA_CALL(cudaPeekAtLastError());

        CUDA_CALL(cudaMemcpy(samplePrices, samplePricesGpu, size * sizeof(dataType), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(sampleWeights, sampleWeightsGpu, size * sizeof(dataType), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(times, timesGpu, size * sizeof(dataType), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    CUDA_CALL(cudaFree(samplePricesGpu));
    CUDA_CALL(cudaFree(sampleWeightsGpu));
    CUDA_CALL(cudaFree(timesGpu));
    CUDA_CALL(cudaFree(optionStructsGpu));

    free(samplePrices);
    free(sampleWeights);
    free(times);
    free(optionStructs);
}

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);

    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", 10, "number of iterations");
    parser.set_optional<int>("seed", "seed", 123, "seed for RNG");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size = parser.get<size_t>("size");
    const int trials = parser.get<int>("trials");
    const int seed = parser.get<int>("seed");
    srand(seed);

    int runtime_version;
    CUDA_CALL(cudaRuntimeGetVersion(&runtime_version));
    int device_id;
    CUDA_CALL(cudaGetDevice(&device_id));
    cudaDeviceProp props;
    CUDA_CALL(cudaGetDeviceProperties(&props, device_id));

    //std::cout << "cuRAND: " << version << " ";
    std::cout << "Runtime: " << runtime_version << " ";
    std::cout << "Device: " << props.name;
    std::cout << std::endl << std::endl;

    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        benchmark::RegisterBenchmark(
            ("monteCarlo (CPU)"),
            [=](benchmark::State& state) { runBenchmarkCpu(state, seed, size); }
        ),
        benchmark::RegisterBenchmark(
            ("monteCarlo (OpenMP)"),
            [=](benchmark::State& state) { runBenchmarkOpenMP(state, seed, size); }
        ),
        benchmark::RegisterBenchmark(
            ("monteCarloCuda (Compute Only)"),
            [=](benchmark::State& state) { runBenchmarkCudaV1(state, seed, size); }
        ),
        benchmark::RegisterBenchmark(
            ("monteCarloCuda (+ Transfers)"),
            [=](benchmark::State& state) { runBenchmarkCudaV2(state, seed, size); }
        ),
        benchmark::RegisterBenchmark(
            ("monteCarloCudaOpt (Compute Only)"),
            [=](benchmark::State& state) { runBenchmarkCudaV3(state, seed, size); }
        ),
        benchmark::RegisterBenchmark(
            ("monteCarloCudaOpt (+ Transfers)"),
            [=](benchmark::State& state) { runBenchmarkCudaV4(state, seed, size); }
        ),
    };

    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
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
