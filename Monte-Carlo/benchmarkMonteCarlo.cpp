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
#ifdef BUILD_HIP
#include "monteCarloKernelsGpu.h"
#include <hip/hip_runtime.h>

#define HIP_CALL(x) do { if((x)!=hipSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)
#endif

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

#ifdef BUILD_HIP
void runBenchmarkHipV1(benchmark::State& state,
                        int seed,
                        size_t size)
{
    monteCarloOptionStruct * optionStructs;
    optionStructs = (monteCarloOptionStruct *)malloc(NUM_OPTIONS * sizeof(monteCarloOptionStruct));
    initOptions(optionStructs);

    dataType * samplePrices;
    dataType * sampleWeights;
    dataType * times;

    hiprandState * devStates;
    dataType * samplePricesGpu;
    dataType * sampleWeightsGpu;
    dataType * timesGpu;
    monteCarloOptionStruct * optionStructsGpu;

    //allocate space for data on CPU
    samplePrices = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    sampleWeights = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    times = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));

    HIP_CALL(hipMalloc((void **)&devStates, size * sizeof(hiprandState)));
    HIP_CALL(hipMalloc(&samplePricesGpu, NUM_OPTIONS * size * sizeof(dataType)));
    HIP_CALL(hipMalloc(&sampleWeightsGpu, NUM_OPTIONS * size * sizeof(dataType)));
    HIP_CALL(hipMalloc(&timesGpu, NUM_OPTIONS * size * sizeof(dataType)));
    HIP_CALL(hipMalloc(&optionStructsGpu, NUM_OPTIONS * sizeof(monteCarloOptionStruct)));

    HIP_CALL(hipMemcpy(optionStructsGpu, optionStructs, NUM_OPTIONS * sizeof(monteCarloOptionStruct), hipMemcpyHostToDevice));
    HIP_CALL(hipDeviceSynchronize());

    dim3 grid((size_t)ceil((dataType)size / ((dataType)THREAD_BLOCK_SIZE)), 1, 1);
    dim3 threads(THREAD_BLOCK_SIZE, 1, 1);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        hipLaunchKernelGGL((setup_kernel), dim3(grid), dim3(threads), 0, 0, devStates, seed, size);
        HIP_CALL(hipPeekAtLastError());
        hipLaunchKernelGGL((monteCarloGpuKernel), dim3(grid), dim3(threads), 0, 0,
            samplePricesGpu, sampleWeightsGpu, timesGpu,
            (1.0f / (dataType)SEQUENCE_LENGTH), devStates, optionStructsGpu,
            seed, size
        );
        HIP_CALL(hipPeekAtLastError());
        HIP_CALL(hipDeviceSynchronize());
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        hipLaunchKernelGGL((setup_kernel), dim3(grid), dim3(threads), 0, 0, devStates, seed, size);
        HIP_CALL(hipPeekAtLastError());
        hipLaunchKernelGGL((monteCarloGpuKernel), dim3(grid), dim3(threads), 0, 0,
            samplePricesGpu, sampleWeightsGpu, timesGpu,
            (1.0f / (dataType)SEQUENCE_LENGTH), devStates, optionStructsGpu,
            seed, size
        );
        HIP_CALL(hipPeekAtLastError());
        HIP_CALL(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    HIP_CALL(hipFree(devStates));
    HIP_CALL(hipFree(samplePricesGpu));
    HIP_CALL(hipFree(sampleWeightsGpu));
    HIP_CALL(hipFree(timesGpu));
    HIP_CALL(hipFree(optionStructsGpu));

    free(samplePrices);
    free(sampleWeights);
    free(times);
    free(optionStructs);
}

void runBenchmarkHipV2(benchmark::State& state,
                        int seed,
                        size_t size)
{
    monteCarloOptionStruct * optionStructs;
    optionStructs = (monteCarloOptionStruct *)malloc(NUM_OPTIONS * sizeof(monteCarloOptionStruct));
    initOptions(optionStructs);

    dataType * samplePrices;
    dataType * sampleWeights;
    dataType * times;

    hiprandState * devStates;
    dataType * samplePricesGpu;
    dataType * sampleWeightsGpu;
    dataType * timesGpu;
    monteCarloOptionStruct * optionStructsGpu;

    //allocate space for data on CPU
    samplePrices = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    sampleWeights = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    times = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));

    HIP_CALL(hipMalloc((void **)&devStates, size * sizeof(hiprandState)));
    HIP_CALL(hipMalloc(&samplePricesGpu, NUM_OPTIONS * size * sizeof(dataType)));
    HIP_CALL(hipMalloc(&sampleWeightsGpu, NUM_OPTIONS * size * sizeof(dataType)));
    HIP_CALL(hipMalloc(&timesGpu, NUM_OPTIONS * size * sizeof(dataType)));
    HIP_CALL(hipMalloc(&optionStructsGpu, NUM_OPTIONS * sizeof(monteCarloOptionStruct)));

    dim3 grid((size_t)ceil((dataType)size / ((dataType)THREAD_BLOCK_SIZE)), 1, 1);
    dim3 threads(THREAD_BLOCK_SIZE, 1, 1);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        hipLaunchKernelGGL((setup_kernel), dim3(grid), dim3(threads), 0, 0, devStates, seed, size);
        //hipDeviceSynchronize();
        hipLaunchKernelGGL((monteCarloGpuKernel), dim3(grid), dim3(threads), 0, 0,
            samplePricesGpu, sampleWeightsGpu, timesGpu,
            (1.0f / (dataType)SEQUENCE_LENGTH), devStates, optionStructsGpu,
            seed, size
        );
        HIP_CALL(hipDeviceSynchronize());
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        HIP_CALL(hipMemcpy(optionStructsGpu, optionStructs, NUM_OPTIONS * sizeof(monteCarloOptionStruct), hipMemcpyHostToDevice));

        hipLaunchKernelGGL((setup_kernel), dim3(grid), dim3(threads), 0, 0, devStates, seed, size);
        HIP_CALL(hipPeekAtLastError());
        hipLaunchKernelGGL((monteCarloGpuKernel), dim3(grid), dim3(threads), 0, 0,
            samplePricesGpu, sampleWeightsGpu, timesGpu,
            (1.0f / (dataType)SEQUENCE_LENGTH), devStates, optionStructsGpu,
            seed, size
        );
        HIP_CALL(hipPeekAtLastError());

        HIP_CALL(hipMemcpy(samplePrices, samplePricesGpu, size * sizeof(dataType), hipMemcpyDeviceToHost));
        HIP_CALL(hipMemcpy(sampleWeights, sampleWeightsGpu, size * sizeof(dataType), hipMemcpyDeviceToHost));
        HIP_CALL(hipMemcpy(times, timesGpu, size * sizeof(dataType), hipMemcpyDeviceToHost));
        HIP_CALL(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    HIP_CALL(hipFree(devStates));
    HIP_CALL(hipFree(samplePricesGpu));
    HIP_CALL(hipFree(sampleWeightsGpu));
    HIP_CALL(hipFree(timesGpu));
    HIP_CALL(hipFree(optionStructsGpu));

    free(samplePrices);
    free(sampleWeights);
    free(times);
    free(optionStructs);
}

void runBenchmarkHipV3(benchmark::State& state,
                        int seed,
                        size_t size)
{
    monteCarloOptionStruct * optionStructs;
    optionStructs = (monteCarloOptionStruct *)malloc(NUM_OPTIONS * sizeof(monteCarloOptionStruct));
    initOptions(optionStructs);

    dataType * samplePrices;
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

    //allocate space for data on CPU
    samplePrices = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    sampleWeights = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    times = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));

    #if defined(__NVCC__)
    HIP_CALL(hipMalloc((void **)&devStates, size * sizeof(hiprandStatePhilox4_32_10_t)));
    #else
    HIP_CALL(hipMalloc((void **)&devStates, size * sizeof(hiprandState)));
    #endif
    HIP_CALL(hipMalloc(&samplePricesGpu, NUM_OPTIONS * size * sizeof(dataType)));
    HIP_CALL(hipMalloc(&sampleWeightsGpu, NUM_OPTIONS * size * sizeof(dataType)));
    HIP_CALL(hipMalloc(&timesGpu, NUM_OPTIONS * size * sizeof(dataType)));
    HIP_CALL(hipMalloc(&optionStructsGpu, NUM_OPTIONS * sizeof(monteCarloOptionStruct)));

    HIP_CALL(hipMemcpy(optionStructsGpu, optionStructs, NUM_OPTIONS * sizeof(monteCarloOptionStruct), hipMemcpyHostToDevice));
    HIP_CALL(hipDeviceSynchronize());

    dim3 grid((size_t)ceil((dataType)size / ((dataType)THREAD_BLOCK_SIZE)), 1, 1);
    dim3 threads(THREAD_BLOCK_SIZE, 1, 1);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        hipLaunchKernelGGL((monteCarloGpuKernel), dim3(grid), dim3(threads), 0, 0,
            samplePricesGpu, sampleWeightsGpu, timesGpu,
            (1.0f / (dataType)SEQUENCE_LENGTH), devStates, optionStructsGpu,
            seed, size
        );
        HIP_CALL(hipPeekAtLastError());
        HIP_CALL(hipDeviceSynchronize());
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        hipLaunchKernelGGL((monteCarloGpuKernel), dim3(grid), dim3(threads), 0, 0,
            samplePricesGpu, sampleWeightsGpu, timesGpu,
            (1.0f / (dataType)SEQUENCE_LENGTH), devStates, optionStructsGpu,
            seed, size
        );
        HIP_CALL(hipPeekAtLastError());
        HIP_CALL(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    HIP_CALL(hipFree(devStates));
    HIP_CALL(hipFree(samplePricesGpu));
    HIP_CALL(hipFree(sampleWeightsGpu));
    HIP_CALL(hipFree(timesGpu));
    HIP_CALL(hipFree(optionStructsGpu));

    free(samplePrices);
    free(sampleWeights);
    free(times);
    free(optionStructs);
}

void runBenchmarkHipV4(benchmark::State& state,
                        int seed,
                        size_t size)
{
    monteCarloOptionStruct * optionStructs;
    optionStructs = (monteCarloOptionStruct *)malloc(NUM_OPTIONS * sizeof(monteCarloOptionStruct));
    initOptions(optionStructs);

    dataType * samplePrices;
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

    //allocate space for data on CPU
    samplePrices = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    sampleWeights = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));
    times = (dataType *)malloc(NUM_OPTIONS * size * sizeof(dataType));

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

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        hipLaunchKernelGGL((monteCarloGpuKernel), dim3(grid), dim3(threads), 0, 0,
            samplePricesGpu, sampleWeightsGpu, timesGpu,
            (1.0f / (dataType)SEQUENCE_LENGTH), devStates, optionStructsGpu,
            seed, size
        );
        HIP_CALL(hipPeekAtLastError());
        HIP_CALL(hipDeviceSynchronize());
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        HIP_CALL(hipMemcpy(optionStructsGpu, optionStructs, NUM_OPTIONS * sizeof(monteCarloOptionStruct), hipMemcpyHostToDevice));

        hipLaunchKernelGGL((monteCarloGpuKernel), dim3(grid), dim3(threads), 0, 0,
            samplePricesGpu, sampleWeightsGpu, timesGpu,
            (1.0f / (dataType)SEQUENCE_LENGTH), devStates, optionStructsGpu,
            seed, size
        );
        HIP_CALL(hipPeekAtLastError());

        HIP_CALL(hipMemcpy(samplePrices, samplePricesGpu, size * sizeof(dataType), hipMemcpyDeviceToHost));
        HIP_CALL(hipMemcpy(sampleWeights, sampleWeightsGpu, size * sizeof(dataType), hipMemcpyDeviceToHost));
        HIP_CALL(hipMemcpy(times, timesGpu, size * sizeof(dataType), hipMemcpyDeviceToHost));
        HIP_CALL(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    HIP_CALL(hipFree(devStates));
    HIP_CALL(hipFree(samplePricesGpu));
    HIP_CALL(hipFree(sampleWeightsGpu));
    HIP_CALL(hipFree(timesGpu));
    HIP_CALL(hipFree(optionStructsGpu));

    free(samplePrices);
    free(sampleWeights);
    free(times);
    free(optionStructs);
}
#endif

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);

    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", 10, "number of iterations");
    parser.set_optional<int>("seed", "seed", 123, "seed for RNG");
    parser.set_optional<int>("device_id", "device_id", 0, "ID of GPU to run");
    parser.set_optional<bool>("run_cpu", "run_cpu", false, "Run single-threaded CPU version (slow)");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size = parser.get<size_t>("size");
    const int trials = parser.get<int>("trials");
    const int seed = parser.get<int>("seed");
    const int device_id = parser.get<int>("device_id");
    const bool run_cpu = parser.get<bool>("run_cpu");
    srand(seed);

    #ifdef BUILD_HIP
    //int device_id;
    //HIP_CALL(hipGetDevice(&device_id));
    hipDeviceProp_t props;
    HIP_CALL(hipGetDeviceProperties(&props, device_id));
    HIP_CALL(hipSetDevice(device_id));

    std::cout << "Device: " << props.name;
    std::cout << std::endl << std::endl;
    #endif

    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        benchmark::RegisterBenchmark(
            ("monteCarlo (OpenMP)"),
            [=](benchmark::State& state) { runBenchmarkOpenMP(state, seed, size); }
        ),
        #ifdef BUILD_HIP
        benchmark::RegisterBenchmark(
            ("monteCarloHip (Compute Only)"),
            [=](benchmark::State& state) { runBenchmarkHipV1(state, seed, size); }
        ),
        benchmark::RegisterBenchmark(
            ("monteCarloHip (+ Transfers)"),
            [=](benchmark::State& state) { runBenchmarkHipV2(state, seed, size); }
        ),
        benchmark::RegisterBenchmark(
            ("monteCarloHipOpt (Compute Only)"),
            [=](benchmark::State& state) { runBenchmarkHipV3(state, seed, size); }
        ),
        benchmark::RegisterBenchmark(
            ("monteCarloHipOpt (+ Transfers)"),
            [=](benchmark::State& state) { runBenchmarkHipV4(state, seed, size); }
        ),
        #endif
    };

    if(run_cpu)
    {
        benchmarks.insert(
            benchmarks.begin(),
            benchmark::RegisterBenchmark(
                ("monteCarlo (CPU)"),
                [=](benchmark::State& state) { runBenchmarkCpu(state, seed, size); }
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
