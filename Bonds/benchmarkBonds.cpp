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
#ifdef BUILD_HIP
#include "bondsKernelsGpu.h"
#include <hip/hip_runtime.h>

#define HIP_CALL(x) do { if((x)!=hipSuccess) { \
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

#ifdef BUILD_HIP
void runBenchmarkHipV1(benchmark::State& state,
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

    hipStream_t stream;
    HIP_CALL(hipStreamCreate(&stream));

    HIP_CALL(hipMalloc(&discountCurveGpu, numBonds * sizeof(bondsYieldTermStruct)));
    HIP_CALL(hipMalloc(&repoCurveGpu, numBonds * sizeof(bondsYieldTermStruct)));
    HIP_CALL(hipMalloc(&currDateGpu, numBonds * sizeof(bondsDateStruct)));
    HIP_CALL(hipMalloc(&maturityDateGpu, numBonds * sizeof(bondsDateStruct)));
    HIP_CALL(hipMalloc(&bondCleanPriceGpu, numBonds * sizeof(dataType)));
    HIP_CALL(hipMalloc(&bondGpu, numBonds * sizeof(bondStruct)));
    HIP_CALL(hipMalloc(&dummyStrikeGpu, numBonds * sizeof(dataType)));
    HIP_CALL(hipMalloc(&dirtyPriceGpu, numBonds * sizeof(dataType)));
    HIP_CALL(hipMalloc(&accruedAmountCurrDateGpu, numBonds * sizeof(dataType)));
    HIP_CALL(hipMalloc(&cleanPriceGpu, numBonds * sizeof(dataType)));
    HIP_CALL(hipMalloc(&bondForwardValGpu, numBonds * sizeof(dataType)));

    HIP_CALL(hipMemcpyAsync(discountCurveGpu, inArgsHost.discountCurve, numBonds * sizeof(bondsYieldTermStruct), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync(repoCurveGpu, inArgsHost.repoCurve, numBonds * sizeof(bondsYieldTermStruct), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync(currDateGpu, inArgsHost.currDate, numBonds * sizeof(bondsDateStruct), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync(maturityDateGpu, inArgsHost.maturityDate, numBonds * sizeof(bondsDateStruct), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync(bondCleanPriceGpu, inArgsHost.bondCleanPrice, numBonds * sizeof(dataType), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync(bondGpu, inArgsHost.bond, numBonds * sizeof(bondStruct), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync(dummyStrikeGpu, inArgsHost.dummyStrike, numBonds * sizeof(dataType), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipStreamSynchronize(stream));

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
        hipLaunchKernelGGL((getBondsResultsGpu), dim3(grid), dim3(threads), 0, stream, inArgs, results, numBonds);
        HIP_CALL(hipStreamSynchronize(stream));
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        hipLaunchKernelGGL((getBondsResultsGpu), dim3(grid), dim3(threads), 0, stream, inArgs, results, numBonds);
        HIP_CALL(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    HIP_CALL(hipFree(discountCurveGpu));
    HIP_CALL(hipFree(repoCurveGpu));
    HIP_CALL(hipFree(currDateGpu));
    HIP_CALL(hipFree(maturityDateGpu));
    HIP_CALL(hipFree(bondCleanPriceGpu));
    HIP_CALL(hipFree(bondGpu));
    HIP_CALL(hipFree(dummyStrikeGpu));
    HIP_CALL(hipFree(dirtyPriceGpu));
    HIP_CALL(hipFree(accruedAmountCurrDateGpu));
    HIP_CALL(hipFree(cleanPriceGpu));
    HIP_CALL(hipFree(bondForwardValGpu));
    HIP_CALL(hipStreamDestroy(stream));

    free(inArgsHost.discountCurve);
    free(inArgsHost.repoCurve);
    free(inArgsHost.currDate);
    free(inArgsHost.maturityDate);
    free(inArgsHost.bondCleanPrice);
    free(inArgsHost.bond);
    free(inArgsHost.dummyStrike);
}

void runBenchmarkHipV2(benchmark::State& state,
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

    hipStream_t stream;
    HIP_CALL(hipStreamCreate(&stream));

    HIP_CALL(hipMalloc(&discountCurveGpu, numBonds * sizeof(bondsYieldTermStruct)));
    HIP_CALL(hipMalloc(&repoCurveGpu, numBonds * sizeof(bondsYieldTermStruct)));
    HIP_CALL(hipMalloc(&currDateGpu, numBonds * sizeof(bondsDateStruct)));
    HIP_CALL(hipMalloc(&maturityDateGpu, numBonds * sizeof(bondsDateStruct)));
    HIP_CALL(hipMalloc(&bondCleanPriceGpu, numBonds * sizeof(dataType)));
    HIP_CALL(hipMalloc(&bondGpu, numBonds * sizeof(bondStruct)));
    HIP_CALL(hipMalloc(&dummyStrikeGpu, numBonds * sizeof(dataType)));
    HIP_CALL(hipMalloc(&dirtyPriceGpu, numBonds * sizeof(dataType)));
    HIP_CALL(hipMalloc(&accruedAmountCurrDateGpu, numBonds * sizeof(dataType)));
    HIP_CALL(hipMalloc(&cleanPriceGpu, numBonds * sizeof(dataType)));
    HIP_CALL(hipMalloc(&bondForwardValGpu, numBonds * sizeof(dataType)));

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
        HIP_CALL(hipMemcpyAsync(discountCurveGpu, inArgsHost.discountCurve, numBonds * sizeof(bondsYieldTermStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(repoCurveGpu, inArgsHost.repoCurve, numBonds * sizeof(bondsYieldTermStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(currDateGpu, inArgsHost.currDate, numBonds * sizeof(bondsDateStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(maturityDateGpu, inArgsHost.maturityDate, numBonds * sizeof(bondsDateStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(bondCleanPriceGpu, inArgsHost.bondCleanPrice, numBonds * sizeof(dataType), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(bondGpu, inArgsHost.bond, numBonds * sizeof(bondStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(dummyStrikeGpu, inArgsHost.dummyStrike, numBonds * sizeof(dataType), hipMemcpyHostToDevice, stream));

        hipLaunchKernelGGL((getBondsResultsGpu), dim3(grid), dim3(threads), 0, 0, inArgs, results, numBonds);
        HIP_CALL(hipStreamSynchronize(stream));
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        HIP_CALL(hipMemcpyAsync(discountCurveGpu, inArgsHost.discountCurve, numBonds * sizeof(bondsYieldTermStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(repoCurveGpu, inArgsHost.repoCurve, numBonds * sizeof(bondsYieldTermStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(currDateGpu, inArgsHost.currDate, numBonds * sizeof(bondsDateStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(maturityDateGpu, inArgsHost.maturityDate, numBonds * sizeof(bondsDateStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(bondCleanPriceGpu, inArgsHost.bondCleanPrice, numBonds * sizeof(dataType), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(bondGpu, inArgsHost.bond, numBonds * sizeof(bondStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(dummyStrikeGpu, inArgsHost.dummyStrike, numBonds * sizeof(dataType), hipMemcpyHostToDevice, stream));

        hipLaunchKernelGGL((getBondsResultsGpu), dim3(grid), dim3(threads), 0, stream, inArgs, results, numBonds);

        HIP_CALL(hipMemcpyAsync(resultsFromGpu.dirtyPrice, dirtyPriceGpu, numBonds * sizeof(dataType), hipMemcpyDeviceToHost, stream));
		HIP_CALL(hipMemcpyAsync(resultsFromGpu.accruedAmountCurrDate, accruedAmountCurrDateGpu, numBonds * sizeof(dataType), hipMemcpyDeviceToHost, stream));
		HIP_CALL(hipMemcpyAsync(resultsFromGpu.cleanPrice, cleanPriceGpu, numBonds * sizeof(dataType), hipMemcpyDeviceToHost, stream));
		HIP_CALL(hipMemcpyAsync(resultsFromGpu.bondForwardVal, bondForwardValGpu, numBonds * sizeof(dataType), hipMemcpyDeviceToHost, stream));
        HIP_CALL(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    HIP_CALL(hipFree(discountCurveGpu));
    HIP_CALL(hipFree(repoCurveGpu));
    HIP_CALL(hipFree(currDateGpu));
    HIP_CALL(hipFree(maturityDateGpu));
    HIP_CALL(hipFree(bondCleanPriceGpu));
    HIP_CALL(hipFree(bondGpu));
    HIP_CALL(hipFree(dummyStrikeGpu));
    HIP_CALL(hipFree(dirtyPriceGpu));
    HIP_CALL(hipFree(accruedAmountCurrDateGpu));
    HIP_CALL(hipFree(cleanPriceGpu));
    HIP_CALL(hipFree(bondForwardValGpu));
    HIP_CALL(hipStreamDestroy(stream));

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
            ("bonds (OpenMP)"),
            [=](benchmark::State& state) { runBenchmarkOpenMP(state, size); }
        ),
        #ifdef BUILD_HIP
        benchmark::RegisterBenchmark(
            ("bondsHip (Compute Only)"),
            [=](benchmark::State& state) { runBenchmarkHipV1(state, size); }
        ),
        benchmark::RegisterBenchmark(
            ("bondsHip (+ Transfers)"),
            [=](benchmark::State& state) { runBenchmarkHipV2(state, size); }
        ),
        #endif
    };

    if(run_cpu)
    {
        benchmarks.insert(
            benchmarks.begin(),
            benchmark::RegisterBenchmark(
                ("bonds (CPU)"),
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
