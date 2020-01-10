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

#include "bondsKernelsCpu.h"
#ifdef BUILD_CUDA
#include "bondsKernelsGpu.cuh"
#include <cuda_runtime.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)
#endif

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024;
#endif

const unsigned int warmup_size = 5;

bondsDateStruct intializeDate(int d,
                              int m,
                              int y)
{
	bondsDateStruct currDate;

	currDate.day = d;
	currDate.month = m;
	currDate.year = y;
	bool leap = isLeapKernel(y);
	int offset = monthOffsetKernel(m,leap);
	currDate.dateSerialNum = d + offset + yearOffsetKernel(y);

	return currDate;
}

void initArgs(inArgsStruct& inArgsHost,
              int numBonds)
{
    int numBond;
    for(numBond = 0; numBond < numBonds; ++numBond)
    {
        dataType repoRate = 0.07;
        int repoCompounding = SIMPLE_INTEREST;
        dataType repoCompoundFreq = 1;

        bondsDateStruct bondIssueDate =  intializeDate(rand() % 28 + 1, rand() % 12 + 1, 1999 - (rand() % 2));
        bondsDateStruct bondMaturityDate = intializeDate(rand() % 28 + 1, rand() % 12 + 1, 2000 + (rand() % 2));
        bondsDateStruct todaysDate = intializeDate(bondMaturityDate.day-1,bondMaturityDate.month,bondMaturityDate.year);

        bondStruct bond;
        bond.startDate = bondIssueDate;
        bond.maturityDate = bondMaturityDate;
        bond.rate = 0.08 + ((float)rand()/(float)RAND_MAX - 0.5)*0.1;

        dataType bondCouponFrequency = 2;
        dataType bondCleanPrice = 89.97693786;

        bondsYieldTermStruct bondCurve;
        bondCurve.refDate = todaysDate;
        bondCurve.calDate = todaysDate;
        bondCurve.forward = -0.1f;  // dummy rate
        bondCurve.compounding = COMPOUNDED_INTEREST;
        bondCurve.frequency = bondCouponFrequency;
        bondCurve.dayCounter = USE_EXACT_DAY;
        bondCurve.refDate = todaysDate;
        bondCurve.calDate = todaysDate;
        bondCurve.compounding = COMPOUNDED_INTEREST;
        bondCurve.frequency = bondCouponFrequency;

        dataType dummyStrike = 91.5745;
        bondsYieldTermStruct repoCurve;
        repoCurve.refDate = todaysDate;
        repoCurve.calDate = todaysDate;
        repoCurve.forward = repoRate;
        repoCurve.compounding = repoCompounding;
        repoCurve.frequency = repoCompoundFreq;
        repoCurve.dayCounter = USE_SERIAL_NUMS;
        inArgsHost.discountCurve[numBond] = bondCurve;
        inArgsHost.repoCurve[numBond] = repoCurve;
        inArgsHost.currDate[numBond] = todaysDate;
        inArgsHost.maturityDate[numBond] = bondMaturityDate;
        inArgsHost.bondCleanPrice[numBond] = bondCleanPrice;
        inArgsHost.bond[numBond] = bond;
        inArgsHost.dummyStrike[numBond] = dummyStrike;
    }
}

void runBenchmarkCpu(benchmark::State& state,
                     size_t size)
{
    int numBonds = size;

    inArgsStruct inArgsHost;
    inArgsHost.discountCurve = (bondsYieldTermStruct *)malloc(numBonds * sizeof(bondsYieldTermStruct));
    inArgsHost.repoCurve = (bondsYieldTermStruct *)malloc(numBonds * sizeof(bondsYieldTermStruct));
    inArgsHost.currDate = (bondsDateStruct *)malloc(numBonds * sizeof(bondsDateStruct));
    inArgsHost.maturityDate = (bondsDateStruct *)malloc(numBonds * sizeof(bondsDateStruct));
    inArgsHost.bondCleanPrice = (dataType *)malloc(numBonds * sizeof(dataType));
    inArgsHost.bond = (bondStruct *)malloc(numBonds * sizeof(bondStruct));
    inArgsHost.dummyStrike = (dataType *)malloc(numBonds * sizeof(dataType));

    initArgs(inArgsHost, numBonds);

    resultsStruct resultsHost;
    resultsHost.dirtyPrice = (dataType *)malloc(numBonds * sizeof(dataType));
    resultsHost.accruedAmountCurrDate = (dataType *)malloc(numBonds * sizeof(dataType));
    resultsHost.cleanPrice = (dataType *)malloc(numBonds * sizeof(dataType));
    resultsHost.bondForwardVal = (dataType *)malloc(numBonds * sizeof(dataType));

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        getBondsResultsCpu(inArgsHost, resultsHost, numBonds);
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        getBondsResultsCpu(inArgsHost, resultsHost, numBonds);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    free(resultsHost.dirtyPrice);
    free(resultsHost.accruedAmountCurrDate);;
    free(resultsHost.cleanPrice);;
    free(resultsHost.bondForwardVal);;

    free(inArgsHost.discountCurve);
    free(inArgsHost.repoCurve);
    free(inArgsHost.currDate);
    free(inArgsHost.maturityDate);
    free(inArgsHost.bondCleanPrice);
    free(inArgsHost.bond);
    free(inArgsHost.dummyStrike);
}

void runBenchmarkOpenMP(benchmark::State& state,
                        size_t size)
{
    int numBonds = size;

    inArgsStruct inArgsHost;
    inArgsHost.discountCurve = (bondsYieldTermStruct *)malloc(numBonds * sizeof(bondsYieldTermStruct));
    inArgsHost.repoCurve = (bondsYieldTermStruct *)malloc(numBonds * sizeof(bondsYieldTermStruct));
    inArgsHost.currDate = (bondsDateStruct *)malloc(numBonds * sizeof(bondsDateStruct));
    inArgsHost.maturityDate = (bondsDateStruct *)malloc(numBonds * sizeof(bondsDateStruct));
    inArgsHost.bondCleanPrice = (dataType *)malloc(numBonds * sizeof(dataType));
    inArgsHost.bond = (bondStruct *)malloc(numBonds * sizeof(bondStruct));
    inArgsHost.dummyStrike = (dataType *)malloc(numBonds * sizeof(dataType));

    initArgs(inArgsHost, numBonds);

    resultsStruct resultsHost;
    resultsHost.dirtyPrice = (dataType *)malloc(numBonds * sizeof(dataType));
    resultsHost.accruedAmountCurrDate = (dataType *)malloc(numBonds * sizeof(dataType));
    resultsHost.cleanPrice = (dataType *)malloc(numBonds * sizeof(dataType));
    resultsHost.bondForwardVal = (dataType *)malloc(numBonds * sizeof(dataType));

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        getBondsResultsOpenMP(inArgsHost, resultsHost, numBonds);
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        getBondsResultsOpenMP(inArgsHost, resultsHost, numBonds);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    free(resultsHost.dirtyPrice);
    free(resultsHost.accruedAmountCurrDate);;
    free(resultsHost.cleanPrice);;
    free(resultsHost.bondForwardVal);;
    free(inArgsHost.discountCurve);
    free(inArgsHost.repoCurve);
    free(inArgsHost.currDate);
    free(inArgsHost.maturityDate);
    free(inArgsHost.bondCleanPrice);
    free(inArgsHost.bond);
    free(inArgsHost.dummyStrike);
}

#ifdef BUILD_CUDA
void runBenchmarkCudaV1(benchmark::State& state,
                        size_t size)
{
    int numBonds = size;

    inArgsStruct inArgsHost;
    inArgsHost.discountCurve = (bondsYieldTermStruct *)malloc(numBonds * sizeof(bondsYieldTermStruct));
    inArgsHost.repoCurve = (bondsYieldTermStruct *)malloc(numBonds * sizeof(bondsYieldTermStruct));
    inArgsHost.currDate = (bondsDateStruct *)malloc(numBonds * sizeof(bondsDateStruct));
    inArgsHost.maturityDate = (bondsDateStruct *)malloc(numBonds * sizeof(bondsDateStruct));
    inArgsHost.bondCleanPrice = (dataType *)malloc(numBonds * sizeof(dataType));
    inArgsHost.bond = (bondStruct *)malloc(numBonds * sizeof(bondStruct));
    inArgsHost.dummyStrike = (dataType *)malloc(numBonds * sizeof(dataType));

    initArgs(inArgsHost, numBonds);

    bondsYieldTermStruct * discountCurveGpu;
    bondsYieldTermStruct * repoCurveGpu;
    bondsDateStruct * currDateGpu;
    bondsDateStruct * maturityDateGpu;
    dataType * bondCleanPriceGpu;
    bondStruct * bondGpu;
    dataType * dummyStrikeGpu;
    dataType * dirtyPriceGpu;
    dataType * accruedAmountCurrDateGpu;
    dataType * cleanPriceGpu;
    dataType * bondForwardValGpu;

    CUDA_CALL(cudaMalloc(&discountCurveGpu, numBonds * sizeof(bondsYieldTermStruct)));
    CUDA_CALL(cudaMalloc(&repoCurveGpu, numBonds * sizeof(bondsYieldTermStruct)));
    CUDA_CALL(cudaMalloc(&currDateGpu, numBonds * sizeof(bondsDateStruct)));
    CUDA_CALL(cudaMalloc(&maturityDateGpu, numBonds * sizeof(bondsDateStruct)));
    CUDA_CALL(cudaMalloc(&bondCleanPriceGpu, numBonds * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&bondGpu, numBonds * sizeof(bondStruct)));
    CUDA_CALL(cudaMalloc(&dummyStrikeGpu, numBonds * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&dirtyPriceGpu, numBonds * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&accruedAmountCurrDateGpu, numBonds * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&cleanPriceGpu, numBonds * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&bondForwardValGpu, numBonds * sizeof(dataType)));

    CUDA_CALL(cudaMemcpy(discountCurveGpu, inArgsHost.discountCurve, numBonds * sizeof(bondsYieldTermStruct), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(repoCurveGpu, inArgsHost.repoCurve, numBonds * sizeof(bondsYieldTermStruct), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(currDateGpu, inArgsHost.currDate, numBonds * sizeof(bondsDateStruct), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(maturityDateGpu, inArgsHost.maturityDate, numBonds * sizeof(bondsDateStruct), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(bondCleanPriceGpu, inArgsHost.bondCleanPrice, numBonds * sizeof(dataType), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(bondGpu, inArgsHost.bond, numBonds * sizeof(bondStruct), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dummyStrikeGpu, inArgsHost.dummyStrike, numBonds * sizeof(dataType), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaDeviceSynchronize());

    inArgsStruct inArgs;
    inArgs.discountCurve    = discountCurveGpu;
    inArgs.repoCurve        = repoCurveGpu;
    inArgs.currDate   = currDateGpu;
    inArgs.maturityDate     = maturityDateGpu;
    inArgs.bondCleanPrice   = bondCleanPriceGpu;
    inArgs.bond             = bondGpu;
    inArgs.dummyStrike      = dummyStrikeGpu;

    resultsStruct results;
    results.dirtyPrice                = dirtyPriceGpu;
    results.accruedAmountCurrDate  = accruedAmountCurrDateGpu;
    results.cleanPrice                = cleanPriceGpu;
    results.bondForwardVal         = bondForwardValGpu;

    dim3 grid((ceil(((float)numBonds)/((float)256))), 1, 1);
    dim3 threads(256, 1, 1);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        getBondsResultsGpu<<<grid, threads>>>(inArgs, results, numBonds);
        CUDA_CALL(cudaDeviceSynchronize());
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        getBondsResultsGpu<<<grid, threads>>>(inArgs, results, numBonds);
        CUDA_CALL(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    CUDA_CALL(cudaFree(discountCurveGpu));
    CUDA_CALL(cudaFree(repoCurveGpu));
    CUDA_CALL(cudaFree(currDateGpu));
    CUDA_CALL(cudaFree(maturityDateGpu));
    CUDA_CALL(cudaFree(bondCleanPriceGpu));
    CUDA_CALL(cudaFree(bondGpu));
    CUDA_CALL(cudaFree(dummyStrikeGpu));
    CUDA_CALL(cudaFree(dirtyPriceGpu));
    CUDA_CALL(cudaFree(accruedAmountCurrDateGpu));
    CUDA_CALL(cudaFree(cleanPriceGpu));
    CUDA_CALL(cudaFree(bondForwardValGpu));

    free(inArgsHost.discountCurve);
    free(inArgsHost.repoCurve);
    free(inArgsHost.currDate);
    free(inArgsHost.maturityDate);
    free(inArgsHost.bondCleanPrice);
    free(inArgsHost.bond);
    free(inArgsHost.dummyStrike);
}

void runBenchmarkCudaV2(benchmark::State& state,
                        size_t size)
{
    int numBonds = size;

    inArgsStruct inArgsHost;
    inArgsHost.discountCurve = (bondsYieldTermStruct *)malloc(numBonds * sizeof(bondsYieldTermStruct));
    inArgsHost.repoCurve = (bondsYieldTermStruct *)malloc(numBonds * sizeof(bondsYieldTermStruct));
    inArgsHost.currDate = (bondsDateStruct *)malloc(numBonds * sizeof(bondsDateStruct));
    inArgsHost.maturityDate = (bondsDateStruct *)malloc(numBonds * sizeof(bondsDateStruct));
    inArgsHost.bondCleanPrice = (dataType *)malloc(numBonds * sizeof(dataType));
    inArgsHost.bond = (bondStruct *)malloc(numBonds * sizeof(bondStruct));
    inArgsHost.dummyStrike = (dataType *)malloc(numBonds * sizeof(dataType));

    resultsStruct resultsFromGpu;
    resultsFromGpu.dirtyPrice = (dataType *)malloc(numBonds * sizeof(dataType));
    resultsFromGpu.accruedAmountCurrDate = (dataType *)malloc(numBonds * sizeof(dataType));;
    resultsFromGpu.cleanPrice = (dataType *)malloc(numBonds * sizeof(dataType));;
    resultsFromGpu.bondForwardVal = (dataType *)malloc(numBonds * sizeof(dataType));;

    initArgs(inArgsHost, numBonds);

    bondsYieldTermStruct * discountCurveGpu;
    bondsYieldTermStruct * repoCurveGpu;
    bondsDateStruct * currDateGpu;
    bondsDateStruct * maturityDateGpu;
    dataType * bondCleanPriceGpu;
    bondStruct * bondGpu;
    dataType * dummyStrikeGpu;
    dataType * dirtyPriceGpu;
    dataType * accruedAmountCurrDateGpu;
    dataType * cleanPriceGpu;
    dataType * bondForwardValGpu;

    CUDA_CALL(cudaMalloc(&discountCurveGpu, numBonds * sizeof(bondsYieldTermStruct)));
    CUDA_CALL(cudaMalloc(&repoCurveGpu, numBonds * sizeof(bondsYieldTermStruct)));
    CUDA_CALL(cudaMalloc(&currDateGpu, numBonds * sizeof(bondsDateStruct)));
    CUDA_CALL(cudaMalloc(&maturityDateGpu, numBonds * sizeof(bondsDateStruct)));
    CUDA_CALL(cudaMalloc(&bondCleanPriceGpu, numBonds * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&bondGpu, numBonds * sizeof(bondStruct)));
    CUDA_CALL(cudaMalloc(&dummyStrikeGpu, numBonds * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&dirtyPriceGpu, numBonds * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&accruedAmountCurrDateGpu, numBonds * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&cleanPriceGpu, numBonds * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&bondForwardValGpu, numBonds * sizeof(dataType)));

    inArgsStruct inArgs;
    inArgs.discountCurve    = discountCurveGpu;
    inArgs.repoCurve        = repoCurveGpu;
    inArgs.currDate   = currDateGpu;
    inArgs.maturityDate     = maturityDateGpu;
    inArgs.bondCleanPrice   = bondCleanPriceGpu;
    inArgs.bond             = bondGpu;
    inArgs.dummyStrike      = dummyStrikeGpu;

    resultsStruct results;
    results.dirtyPrice                = dirtyPriceGpu;
    results.accruedAmountCurrDate  = accruedAmountCurrDateGpu;
    results.cleanPrice                = cleanPriceGpu;
    results.bondForwardVal         = bondForwardValGpu;

    dim3 grid((ceil(((float)numBonds)/((float)256))), 1, 1);
    dim3 threads(256, 1, 1);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        getBondsResultsGpu<<<grid, threads>>>(inArgs, results, numBonds);
        CUDA_CALL(cudaDeviceSynchronize());
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        CUDA_CALL(cudaMemcpy(discountCurveGpu, inArgsHost.discountCurve, numBonds * sizeof(bondsYieldTermStruct), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(repoCurveGpu, inArgsHost.repoCurve, numBonds * sizeof(bondsYieldTermStruct), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(currDateGpu, inArgsHost.currDate, numBonds * sizeof(bondsDateStruct), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(maturityDateGpu, inArgsHost.maturityDate, numBonds * sizeof(bondsDateStruct), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(bondCleanPriceGpu, inArgsHost.bondCleanPrice, numBonds * sizeof(dataType), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(bondGpu, inArgsHost.bond, numBonds * sizeof(bondStruct), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(dummyStrikeGpu, inArgsHost.dummyStrike, numBonds * sizeof(dataType), cudaMemcpyHostToDevice));

        getBondsResultsGpu<<<grid, threads>>>(inArgs, results, numBonds);

        CUDA_CALL(cudaMemcpy(resultsFromGpu.dirtyPrice, dirtyPriceGpu, numBonds * sizeof(dataType), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(resultsFromGpu.accruedAmountCurrDate, accruedAmountCurrDateGpu, numBonds * sizeof(dataType), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(resultsFromGpu.cleanPrice, cleanPriceGpu, numBonds * sizeof(dataType), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(resultsFromGpu.bondForwardVal, bondForwardValGpu, numBonds * sizeof(dataType), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    CUDA_CALL(cudaFree(discountCurveGpu));
    CUDA_CALL(cudaFree(repoCurveGpu));
    CUDA_CALL(cudaFree(currDateGpu));
    CUDA_CALL(cudaFree(maturityDateGpu));
    CUDA_CALL(cudaFree(bondCleanPriceGpu));
    CUDA_CALL(cudaFree(bondGpu));
    CUDA_CALL(cudaFree(dummyStrikeGpu));
    CUDA_CALL(cudaFree(dirtyPriceGpu));
    CUDA_CALL(cudaFree(accruedAmountCurrDateGpu));
    CUDA_CALL(cudaFree(cleanPriceGpu));
    CUDA_CALL(cudaFree(bondForwardValGpu));

    free(resultsFromGpu.dirtyPrice);
    free(resultsFromGpu.accruedAmountCurrDate);;
    free(resultsFromGpu.cleanPrice);;
    free(resultsFromGpu.bondForwardVal);
    free(inArgsHost.discountCurve);
    free(inArgsHost.repoCurve);
    free(inArgsHost.currDate);
    free(inArgsHost.maturityDate);
    free(inArgsHost.bondCleanPrice);
    free(inArgsHost.bond);
    free(inArgsHost.dummyStrike);
}
#endif

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

    #ifdef BUILD_CUDA
    int runtime_version;
    CUDA_CALL(cudaRuntimeGetVersion(&runtime_version));
    int device_id;
    CUDA_CALL(cudaGetDevice(&device_id));
    cudaDeviceProp props;
    CUDA_CALL(cudaGetDeviceProperties(&props, device_id));

    std::cout << "Runtime: " << runtime_version << " ";
    std::cout << "Device: " << props.name;
    std::cout << std::endl << std::endl;
    #endif

    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        benchmark::RegisterBenchmark(
            ("bonds (CPU)"),
            [=](benchmark::State& state) { runBenchmarkCpu(state, size); }
        ),
        benchmark::RegisterBenchmark(
            ("bonds (OpenMP)"),
            [=](benchmark::State& state) { runBenchmarkOpenMP(state, size); }
        ),
        #ifdef BUILD_CUDA
        benchmark::RegisterBenchmark(
            ("bondsCuda (Compute Only)"),
            [=](benchmark::State& state) { runBenchmarkCudaV1(state, size); }
        ),
        benchmark::RegisterBenchmark(
            ("bondsCuda (+ Transfers)"),
            [=](benchmark::State& state) { runBenchmarkCudaV2(state, size); }
        ),
        #endif
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
